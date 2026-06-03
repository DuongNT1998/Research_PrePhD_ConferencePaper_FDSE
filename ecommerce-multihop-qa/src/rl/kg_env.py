"""
src/rl/kg_env.py

Knowledge Graph Reinforcement Learning Environment.

Implements an OpenAI-Gym-style environment where:
  - observation = dense state vector (total_dim,)
  - action      = integer index into the dynamic action list at current node
  - reward      = multi-objective scalar (defined in reward.py)
  - done        = True when policy chooses STOP or safety hop limit reached

The environment wraps Neo4j via KGConnector and delegates state construction
to StateBuilder.  Dynamic action spaces are built fresh at each step from the
live graph neighbourhood — no hard-coded traversal rules.

Episode lifecycle
-----------------
    obs, info = env.reset(query_text, anchor_node_id)
    while not done:
        action_list = env.get_valid_actions()   # dynamic
        action_idx  = policy(obs, action_list)
        obs, reward, done, info = env.step(action_idx)
    trajectory  = env.get_trajectory()
    evidence    = env.get_evidence()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from src.config.settings import Config, DEFAULT_CONFIG
from src.kg.neo4j_connector import KGConnector
from src.retrieval.query_encoder import QueryEncoder, QueryRepresentation
from src.retrieval.node_encoder import (
    NodeEncoder, NodeRepresentation, EdgeRepresentation,
)
from src.retrieval.state_builder import StateBuilder, EpisodeState
from src.retrieval.semantic_scorer import SemanticScorer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Action dataclass
# ---------------------------------------------------------------------------

@dataclass
class Action:
    """Represents one element of the dynamic action space at a given step."""
    index: int                         # position in current action list
    is_stop: bool                      # True if this is the STOP action
    target_node_id: Optional[str]      # None if is_stop
    target_node_type: Optional[str]
    relation_type: Optional[str]
    edge_weight: float = 1.0
    # Pre-computed action embedding for the policy network (projected node+edge)
    action_embedding: Optional[torch.Tensor] = None
    # Target node BASE embedding (embedding_dim, shared sentence-transformer
    # space) — the only embedding comparable to the query via cosine similarity.
    target_base_embedding: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# StepInfo
# ---------------------------------------------------------------------------

@dataclass
class StepInfo:
    """Metadata returned alongside (obs, reward, done) at each step."""
    hop: int
    action_taken: Action
    node_id: str
    node_type: str
    relation_type: str
    evidence_text: str
    uncertainty: float
    reward_breakdown: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# KGEnvironment
# ---------------------------------------------------------------------------

class KGEnvironment:
    """
    RL environment for adaptive multi-hop retrieval over a Neo4j KG.

    Parameters
    ----------
    connector  : KGConnector    — live Neo4j connection
    query_enc  : QueryEncoder
    node_enc   : NodeEncoder
    state_builder : StateBuilder
    scorer     : SemanticScorer — computes relevance scores
    reward_fn  : callable       — reward_fn(env_state) → float
    config     : Config
    """

    def __init__(
        self,
        connector: KGConnector,
        query_enc: QueryEncoder,
        node_enc: NodeEncoder,
        state_builder: StateBuilder,
        scorer: SemanticScorer,
        reward_fn,                      # imported lazily to avoid circular
        config: Config = DEFAULT_CONFIG,
    ) -> None:
        self.connector = connector
        self.query_enc = query_enc
        self.node_enc = node_enc
        self.state_builder = state_builder
        self.scorer = scorer
        self.reward_fn = reward_fn
        self.config = config
        self.device = torch.device(config.encoder.device)

        # Episode state (set on reset)
        self._episode: Optional[EpisodeState] = None
        self._query_repr: Optional[QueryRepresentation] = None
        self._current_actions: List[Action] = []
        self._total_reward: float = 0.0

        logger.info("KGEnvironment initialised.")

    # ------------------------------------------------------------------
    # Gym-style API
    # ------------------------------------------------------------------

    def reset(
        self,
        query_text: str,
        anchor_node_id: Optional[str] = None,
        gold_answers: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Start a new retrieval episode.

        Parameters
        ----------
        query_text : str
            Raw English user query.
        anchor_node_id : str | None
            If None, the environment resolves the best anchor node via
            semantic search over product titles in Neo4j.
        gold_answers : list[str] | None
            Optional gold parent_asins (from the QA dataset). When provided, the
            reward function scores answer quality against them exactly. Leave as
            None for free inference.

        Returns
        -------
        obs : (total_dim,) tensor
        info : dict with query_repr and anchor metadata
        """
        logger.debug("KGEnvironment.reset() query='%s'", query_text[:80])

        # 1. Encode query
        self._query_repr = self.query_enc.encode(query_text)

        # 2. Resolve anchor node (robust chain: resolver → gold → any product)
        if anchor_node_id is None:
            try:
                anchor_node_id = self._resolve_anchor(self._query_repr)
            except Exception as exc:
                logger.warning("Anchor resolver raised: %s", exc)
                anchor_node_id = ""
            if not anchor_node_id and gold_answers:
                # Training fallback: start from a known in-graph gold product.
                anchor_node_id = gold_answers[0]
                logger.debug("Anchor fell back to gold product: %s", anchor_node_id)
            if not anchor_node_id:
                anchor_node_id = self._fallback_any_product()
            if not anchor_node_id:
                raise RuntimeError(
                    "Could not resolve any anchor node — the Product set appears "
                    "empty. Check that the KG is populated and NEO4J_DATABASE is correct."
                )
        logger.debug("Anchor node: %s", anchor_node_id)

        # 3. Fetch anchor node properties from Neo4j
        anchor_props, anchor_type = self.connector.get_node_by_id(anchor_node_id)
        anchor_repr = self.node_enc.encode_node(
            node_type=anchor_type,
            properties=anchor_props,
            node_id=anchor_node_id,
        )

        # 4. Reset state builder
        self._episode = self.state_builder.reset(self._query_repr, anchor_repr)
        self._episode.gold_answers = list(gold_answers or [])
        self._total_reward = 0.0

        # 5. Build initial action space
        self._current_actions = self._build_action_space(anchor_node_id, anchor_type)

        obs = self._episode.state_vector
        info = {
            "anchor_node_id": anchor_node_id,
            "anchor_type": anchor_type,
            "query_repr": self._query_repr,
            "num_actions": len(self._current_actions),
        }
        return obs, info

    def step(
        self, action_idx: int
    ) -> Tuple[torch.Tensor, float, bool, StepInfo]:
        """
        Execute one step in the retrieval episode.

        Parameters
        ----------
        action_idx : int
            Index into the current dynamic action list
            (including the STOP action which is always last).

        Returns
        -------
        obs    : (total_dim,)
        reward : float
        done   : bool
        info   : StepInfo
        """
        if self._episode is None:
            raise RuntimeError("Call reset() before step().")

        action = self._current_actions[action_idx]

        # --- STOP action ---
        if action.is_stop:
            self._episode.done = True
            reward, breakdown = self.reward_fn(
                episode=self._episode,
                action=action,
                is_terminal=True,
            )
            self._total_reward += reward
            info = StepInfo(
                hop=self._episode.hop_count,
                action_taken=action,
                node_id=self._episode.current_node.node_id,
                node_type=self._episode.current_node.node_type,
                relation_type="STOP",
                evidence_text="",
                uncertainty=self._episode.uncertainty_score,
                reward_breakdown=breakdown,
            )
            return self._episode.state_vector, reward, True, info

        # --- Safety hop limit (emergency brake only) ---
        if self._episode.hop_count >= self.config.environment.max_hops:
            self._episode.done = True
            logger.warning(
                "Safety max_hops=%d reached. Forcing stop.",
                self.config.environment.max_hops,
            )
            reward, breakdown = self.reward_fn(
                episode=self._episode,
                action=action,
                is_terminal=True,
            )
            return self._episode.state_vector, reward, True, StepInfo(
                hop=self._episode.hop_count,
                action_taken=action,
                node_id=self._episode.current_node.node_id,
                node_type=self._episode.current_node.node_type,
                relation_type="FORCED_STOP",
                evidence_text="",
                uncertainty=self._episode.uncertainty_score,
                reward_breakdown=breakdown,
            )

        # --- Move to neighbour ---
        target_id = action.target_node_id
        target_type = action.target_node_type
        relation_type = action.relation_type

        # Fetch target node
        target_props, _ = self.connector.get_node_by_id(target_id)
        target_repr = self.node_enc.encode_node(
            node_type=target_type,
            properties=target_props,
            node_id=target_id,
        )
        edge_repr = self.node_enc.encode_edge(
            relation_type=relation_type,
            weight=action.edge_weight,
            source_id=self._episode.current_node.node_id,
            target_id=target_id,
            target_type=target_type,
        )

        # Construct evidence text from node properties
        evidence_text = self._extract_evidence_text(target_props, target_type, relation_type)

        # Compute uncertainty BEFORE state update (stopping module will be
        # called with the new accumulated evidence)
        new_uncertainty = self._compute_uncertainty(target_repr)

        # Update episode state
        self._episode = self.state_builder.step(
            new_node=target_repr,
            edge=edge_repr,
            evidence_text=evidence_text,
            uncertainty=new_uncertainty,
        )

        # Compute reward
        reward, breakdown = self.reward_fn(
            episode=self._episode,
            action=action,
            is_terminal=False,
        )
        self._total_reward += reward

        # Build new action space from new position
        self._current_actions = self._build_action_space(target_id, target_type)

        done = self._episode.done or (len(self._current_actions) == 1)
        # (only STOP action left means dead end)

        info = StepInfo(
            hop=self._episode.hop_count,
            action_taken=action,
            node_id=target_id,
            node_type=target_type,
            relation_type=relation_type,
            evidence_text=evidence_text,
            uncertainty=new_uncertainty,
            reward_breakdown=breakdown,
        )
        return self._episode.state_vector, reward, done, info

    # ------------------------------------------------------------------
    # Dynamic action space
    # ------------------------------------------------------------------

    def get_valid_actions(self) -> List[Action]:
        """Return the current dynamic action list (read-only)."""
        return list(self._current_actions)

    def _build_action_space(
        self, node_id: str, node_type: str
    ) -> List[Action]:
        """
        Query Neo4j for neighbours and build the dynamic action list.

        Returns a list of Action objects.  The STOP action is always appended
        as the last element so the policy can always choose to halt.

        The action list is bounded by max_neighbours for efficiency.
        """
        max_n = self.config.environment.max_neighbours
        neighbours = self.connector.get_neighbours(
            node_id=node_id,
            node_type=node_type,
            max_results=max_n,
        )
        # Filter already-visited nodes (except if no unvisited remain)
        visited = self._episode.visited_node_ids if self._episode else set()
        unvisited = [n for n in neighbours if n["target_id"] not in visited]
        if not unvisited:
            unvisited = neighbours  # allow revisit if no fresh nodes

        actions: List[Action] = []
        for idx, nb in enumerate(unvisited):
            # Pre-compute action embedding: concat node_emb + edge_emb
            target_props = nb.get("target_props", {})
            target_repr = self.node_enc.encode_node(
                node_type=nb["target_type"],
                properties=target_props,
                node_id=nb["target_id"],
            )
            edge_repr = self.node_enc.encode_edge(
                relation_type=nb["relation_type"],
                weight=nb.get("weight", 1.0),
            )
            action_emb = torch.cat([
                target_repr.embedding,
                edge_repr.embedding,
            ], dim=0)   # (node_feature_dim + edge_type_dim,)

            actions.append(Action(
                index=idx,
                is_stop=False,
                target_node_id=nb["target_id"],
                target_node_type=nb["target_type"],
                relation_type=nb["relation_type"],
                edge_weight=nb.get("weight", 1.0),
                action_embedding=action_emb,
                target_base_embedding=target_repr.base_embedding,
            ))

        # STOP is always last
        stop_edge_emb = self.node_enc.encode_edge("STOP")
        stop_node_emb = torch.zeros(
            self.config.encoder.node_feature_dim, device=self.device
        )
        stop_action_emb = torch.cat([stop_node_emb, stop_edge_emb.embedding], dim=0)
        actions.append(Action(
            index=len(actions),
            is_stop=True,
            target_node_id=None,
            target_node_type=None,
            relation_type="STOP",
            edge_weight=0.0,
            action_embedding=stop_action_emb,
        ))
        return actions

    # ------------------------------------------------------------------
    # Anchor resolution
    # ------------------------------------------------------------------

    def _resolve_anchor(self, query_repr: QueryRepresentation) -> str:
        """
        Find the best anchor (starting) node in Neo4j for a query.

        Multi-tier, schema-aware, and robust: it never raises just because one
        search came back empty — it falls through to broader searches and finally
        to top products.  Returns "" only if no candidate could be ranked, in
        which case KGEnvironment.reset() applies its own gold/any-product fallback.

        Tier 1: Product.title CONTAINS each candidate keyword (product hints,
                singularised tokens).
        Tier 2: Category.name CONTAINS a keyword  →  products in that category.
        Tier 3: top products overall (keyword="").
        """
        candidates: List[Dict[str, Any]] = []

        # Tier 1 — product title search over candidate keywords
        for kw in self._anchor_keywords(query_repr):
            hits = self.connector.search_products(keyword=kw, limit=20)
            if hits:
                candidates = hits
                break

        # Tier 2 — category search (maps to Category.name in the schema)
        if not candidates:
            for kw in self._anchor_keywords(query_repr):
                try:
                    hits = self.connector.search_products_by_category(kw, limit=20)
                except Exception:
                    hits = []
                if hits:
                    candidates = hits
                    break

        # Tier 3 — top products overall
        if not candidates:
            candidates = self.connector.search_products(keyword="", limit=20)

        if not candidates:
            logger.warning("Anchor resolution found no candidates.")
            return ""

        best_id: str = ""
        best_score: float = -1.0
        query_base = query_repr.base_embedding
        for cand in candidates:
            asin = cand.get("parent_asin")
            if not asin:
                continue
            try:
                node_repr = self.node_enc.encode_node(
                    node_type="Product",
                    properties=cand,
                    node_id=asin,
                )
                # Compare in the shared base space (both embedding_dim).
                score = self.scorer.score(query_base, node_repr.base_embedding)
            except Exception as exc:
                logger.debug("Scoring candidate %s failed: %s", asin, exc)
                continue
            if score > best_score:
                best_score = score
                best_id = asin

        # If scoring failed for all, just take the first candidate.
        if not best_id:
            best_id = candidates[0].get("parent_asin", "")

        logger.debug("Anchor resolved: %s (score=%.4f)", best_id, best_score)
        return best_id

    def _fallback_any_product(self) -> str:
        """Return any product asin from the graph (last-resort anchor)."""
        hits = self.connector.search_products(keyword="", limit=1)
        return hits[0].get("parent_asin", "") if hits else ""

    @staticmethod
    def _anchor_keywords(query_repr: QueryRepresentation) -> List[str]:
        """
        Build an ordered list of candidate keywords for anchor search.
        Uses explicit product-type hints first, then singularised content tokens,
        then detected aspects. De-duplicated, preserving order.
        """
        kws: List[str] = []

        def _add(word: str) -> None:
            w = (word or "").strip().lower()
            if len(w) >= 3 and w not in kws:
                kws.append(w)

        for h in query_repr.product_type_hints:
            _add(h)
        # singularise simple plurals from tokens ("laptops" -> "laptop")
        stop = {"with", "and", "the", "for", "good", "best", "not", "but",
                "under", "over", "without", "great", "high", "low", "that",
                "has", "featuring", "offering", "though", "yet", "ideally",
                "dollars", "than", "too"}
        for t in query_repr.tokens:
            tl = t.lower()
            if tl in stop or not tl.isalpha():
                continue
            _add(tl)
            if tl.endswith("s") and len(tl) > 3:
                _add(tl[:-1])
        for a in query_repr.detected_aspects:
            _add(a)
        return kws

    # ------------------------------------------------------------------
    # Evidence & trajectory access
    # ------------------------------------------------------------------

    def get_trajectory(self) -> List[Dict[str, Any]]:
        return self.state_builder.extract_trajectory()

    def get_evidence(self) -> List[str]:
        return self.state_builder.extract_evidence_summary()

    def get_total_reward(self) -> float:
        return self._total_reward

    def get_episode(self) -> Optional[EpisodeState]:
        return self._episode

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _extract_evidence_text(
        self, props: Dict[str, Any], node_type: str, relation: str
    ) -> str:
        """
        Construct a human-readable evidence string from node properties.
        This text is later fed to the LLM for answer synthesis.
        """
        parts: List[str] = [f"[{node_type} via {relation}]"]
        if node_type == "Product":
            if props.get("title"):
                parts.append(f"Product: {props['title']}")
            if props.get("price"):
                parts.append(f"Price: ${props['price']:.2f}")
            if props.get("average_rating"):
                parts.append(f"Rating: {props['average_rating']:.1f}/5")
        elif node_type == "Aspect":
            if props.get("name"):
                sentiment = "positive" if "POSITIVE" in relation else "negative"
                parts.append(f"Aspect ({sentiment}): {props['name']}")
        elif node_type == "Feature":
            if props.get("text"):
                parts.append(f"Feature: {props['text']}")
        elif node_type == "Brand":
            if props.get("name"):
                parts.append(f"Brand: {props['name']}")
        elif node_type == "Category":
            if props.get("name"):
                parts.append(f"Category: {props['name']}")
        elif node_type == "Detail":
            parts.append(f"{props.get('key', '')}: {props.get('value', '')}")
        return " | ".join(parts)

    def _compute_uncertainty(self, node_repr: NodeRepresentation) -> float:
        """
        Simple uncertainty proxy: cosine similarity between current evidence
        centroid and the new node embedding.  High similarity → low uncertainty.
        This is overridden by the full UncertaintyEstimator in stopping.py.
        """
        if self._episode is None or not self._episode.collected_node_reprs:
            return 1.0
        stack = torch.stack(self._episode.collected_node_reprs, dim=0)
        centroid = stack.mean(dim=0)
        cos = torch.nn.functional.cosine_similarity(
            centroid.unsqueeze(0),
            node_repr.embedding.unsqueeze(0),
        ).item()
        # cos ∈ [-1,1] → uncertainty ∈ [0,1]; high similarity → low uncertainty
        return max(0.0, 1.0 - (cos + 1.0) / 2.0)