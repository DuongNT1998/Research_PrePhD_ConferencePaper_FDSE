"""
src/rl/reward.py

Multi-objective Reward Function.

Reward Components
-----------------
R_total = w1 * R_answer_quality
        + w2 * R_retrieval_relevance
        + w3 * R_path_quality
        + w4 * R_grounding
        + w5 * penalty_efficiency      (per-step cost, negative)
        + w6 * R_uncertainty_stop      (bonus for stopping at right time)

Each component is normalised to [0, 1] before weighting.

Design principle
----------------
- Non-terminal steps receive R_retrieval_relevance + R_path_quality + penalty.
- Terminal steps receive all components.
- The LLM judge score (R_answer_quality) is computed lazily only at terminal
  steps to avoid expensive API calls at every hop.
- The reward function is called by KGEnvironment.step() and by the PPO trainer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from src.config.settings import Config, DEFAULT_CONFIG
from src.retrieval.state_builder import EpisodeState
from src.rl.kg_env import Action

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reward breakdown dataclass
# ---------------------------------------------------------------------------

@dataclass
class RewardBreakdown:
    answer_quality: float = 0.0
    retrieval_relevance: float = 0.0
    path_quality: float = 0.0
    grounding: float = 0.0
    efficiency_penalty: float = 0.0
    uncertainty_stop: float = 0.0
    total: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "answer_quality": self.answer_quality,
            "retrieval_relevance": self.retrieval_relevance,
            "path_quality": self.path_quality,
            "grounding": self.grounding,
            "efficiency_penalty": self.efficiency_penalty,
            "uncertainty_stop": self.uncertainty_stop,
            "total": self.total,
        }


# ---------------------------------------------------------------------------
# Individual reward components
# ---------------------------------------------------------------------------

def compute_retrieval_relevance(
    query_base_emb: torch.Tensor,
    node_base_emb: torch.Tensor,
) -> float:
    """
    Cosine similarity between query and the newly retrieved node, computed in the
    SHARED sentence-transformer base space (embedding_dim).  Measures how relevant
    this hop was to the query.  Returns float in [0, 1].

    Note
    ----
    Both arguments must be *base* embeddings (``QueryRepresentation.base_embedding``
    and ``NodeRepresentation.base_embedding``).  The projected query/node vectors
    live in separate learned spaces of different dimensionality and are NOT
    comparable via cosine similarity.
    """
    cos = F.cosine_similarity(
        query_base_emb.unsqueeze(0), node_base_emb.unsqueeze(0)
    ).item()
    return (cos + 1.0) / 2.0


def compute_path_quality(episode: EpisodeState) -> float:
    """
    Evaluate the quality of the traversal path taken so far.
    Criteria:
    1. No cycles (revisit penalty).
    2. Edge type diversity (broad exploration rewarded).
    3. Aspect edges weighted higher (direct review signal).
    Returns float in [0, 1].
    """
    if not episode.visited_steps:
        return 0.5

    steps = episode.visited_steps
    n = len(steps)

    # 1. No-cycle score: unique nodes / total steps
    unique_nodes = len({s.node_id for s in steps})
    no_cycle_score = unique_nodes / n

    # 2. Relation diversity
    relation_types = {s.relation_type for s in steps}
    diversity = min(len(relation_types) / 3.0, 1.0)  # normalise by 3

    # 3. Aspect edge bonus
    aspect_steps = sum(
        1 for s in steps
        if "ASPECT" in s.relation_type.upper()
    )
    aspect_score = min(aspect_steps / max(n, 1), 1.0)

    path_score = 0.5 * no_cycle_score + 0.3 * diversity + 0.2 * aspect_score
    return float(path_score)


def compute_gold_answer_quality(episode: EpisodeState) -> Optional[float]:
    """
    Exact answer-quality signal computed against the QA dataset's gold answers.

    Returns the F1 between the Product nodes the agent actually visited and the
    gold parent_asins, in [0, 1].  Returns None when no gold answers are attached
    to the episode (free inference), so the caller can fall back to the LLM judge.

    Rationale
    ---------
    During training every query carries verified gold answers, so we can reward
    the policy precisely (did it reach the right products?) instead of paying for
    an LLM call at every terminal step.  The LLM judge remains the weak-supervision
    fallback for queries without gold.
    """
    gold = set(getattr(episode, "gold_answers", []) or [])
    if not gold:
        return None

    visited_products = {
        step.node_id
        for step in episode.visited_steps
        if step.node_type == "Product"
    }
    if (
        episode.current_node is not None
        and episode.current_node.node_type == "Product"
    ):
        visited_products.add(episode.current_node.node_id)

    if not visited_products:
        return 0.0

    hits = visited_products & gold
    if not hits:
        return 0.0
    precision = len(hits) / len(visited_products)
    recall = len(hits) / len(gold)
    return 2.0 * precision * recall / (precision + recall)


def compute_grounding_score(
    episode: EpisodeState,
    query_base_emb: torch.Tensor,
    device: torch.device,
) -> float:
    """
    Measures how well the collected evidence set covers the query.
    Uses the centroid of collected node BASE embeddings vs the query base
    embedding (both in the shared sentence-transformer space).
    Returns float in [0, 1].
    """
    if not episode.collected_base_reprs:
        return 0.0
    stack = torch.stack(episode.collected_base_reprs, dim=0).to(device)
    centroid = stack.mean(dim=0)
    cos = F.cosine_similarity(
        query_base_emb.to(device).unsqueeze(0),
        centroid.unsqueeze(0),
    ).item()
    return (cos + 1.0) / 2.0


def compute_efficiency_penalty(hop_count: int, max_hops: int) -> float:
    """
    Linear penalty per hop.  Encourages stopping earlier if evidence is sufficient.
    Returns float in [-1, 0].
    """
    return -(hop_count / max(max_hops, 1))


def compute_uncertainty_stop_bonus(
    uncertainty: float,
    threshold: float,
    is_terminal: bool,
) -> float:
    """
    Bonus for stopping when uncertainty is genuinely low (good stopping).
    Penalty for stopping when uncertainty is still high (premature stop).
    Only applies at terminal step.
    Returns float in [-1, 1].
    """
    if not is_terminal:
        return 0.0
    # Low uncertainty at stop → bonus; high uncertainty → penalty
    return float(threshold - uncertainty)   # positive when U < threshold


# ---------------------------------------------------------------------------
# Main RewardFunction
# ---------------------------------------------------------------------------

class RewardFunction:
    """
    Multi-objective reward function for the retrieval RL agent.

    Parameters
    ----------
    config : Config
    llm_judge : optional callable
        llm_judge(query, evidence_list) → float score in [0,1].
        Called only at terminal steps.  If None, answer quality is estimated
        from grounding score.
    """

    def __init__(
        self,
        config: Config = DEFAULT_CONFIG,
        llm_judge=None,
    ) -> None:
        self.config = config
        self.llm_judge = llm_judge
        self.device = torch.device(config.encoder.device)
        self.rc = config.reward
        self.sc = config.stopping
        logger.info("RewardFunction initialised (llm_judge=%s)", llm_judge is not None)

    def __call__(
        self,
        episode: EpisodeState,
        action: Action,
        is_terminal: bool,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute reward for a single step.

        Parameters
        ----------
        episode    : current EpisodeState AFTER the step has been applied
        action     : the Action that was taken
        is_terminal: True if this is the final step (STOP or max_hops)

        Returns
        -------
        total_reward : float
        breakdown    : dict of component rewards
        """
        bd = RewardBreakdown()
        # Base-space query embedding (shared with node base embeddings)
        query_base = episode.query_repr.base_embedding.to(self.device)

        # ------ Always-computed components ------

        # R2: retrieval relevance (how relevant is the current node to query)
        if not action.is_stop and episode.current_node is not None:
            node_base = episode.current_node.base_embedding.to(self.device)
            bd.retrieval_relevance = compute_retrieval_relevance(query_base, node_base)
        else:
            bd.retrieval_relevance = 0.0

        # R3: path quality
        bd.path_quality = compute_path_quality(episode)

        # R5: efficiency penalty (per step)
        bd.efficiency_penalty = compute_efficiency_penalty(
            episode.hop_count,
            self.config.environment.max_hops,
        )

        # ------ Terminal-only components ------
        if is_terminal:
            # R4: grounding
            bd.grounding = compute_grounding_score(episode, query_base, self.device)

            # R1: answer quality.
            # Prefer the exact gold-based score when the episode carries gold
            # answers (training); otherwise fall back to LLM judge (weak
            # supervision), then to the grounding proxy.
            gold_q = compute_gold_answer_quality(episode)
            if gold_q is not None:
                bd.answer_quality = gold_q
            elif self.llm_judge is not None:
                try:
                    bd.answer_quality = self.llm_judge(
                        query=episode.query_repr.raw_text,
                        evidence=episode.collected_evidence,
                    )
                except Exception as exc:
                    logger.warning("LLM judge failed: %s — using grounding proxy.", exc)
                    bd.answer_quality = bd.grounding
            else:
                bd.answer_quality = bd.grounding

            # R6: uncertainty stop bonus/penalty
            bd.uncertainty_stop = compute_uncertainty_stop_bonus(
                uncertainty=episode.uncertainty_score,
                threshold=self.sc.uncertainty_threshold,
                is_terminal=True,
            )
        else:
            # Non-terminal: partial grounding only
            bd.grounding = compute_grounding_score(episode, query_base, self.device) * 0.3
            bd.answer_quality = 0.0
            bd.uncertainty_stop = 0.0

        # ------ Weighted sum ------
        rc = self.rc
        bd.total = (
            rc.w_answer_quality    * bd.answer_quality
            + rc.w_retrieval_relevance * bd.retrieval_relevance
            + rc.w_path_quality    * bd.path_quality
            + rc.w_grounding       * bd.grounding
            + rc.w_efficiency_penalty * abs(bd.efficiency_penalty)
            + rc.w_uncertainty_stop   * bd.uncertainty_stop
        )

        return bd.total, bd.to_dict()