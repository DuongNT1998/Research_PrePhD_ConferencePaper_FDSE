"""
src/retrieval/adaptive_retriever.py

Adaptive Multi-hop Retriever — top-level orchestrator.

This is the main inference interface that combines:
  KGEnvironment + Actor (trained policy) + AdaptiveStoppingModule
→ produces a complete RetrievalResult with trajectory, evidence, and path.

Usage
-----
    retriever = AdaptiveRetriever(env, actor, stopping_module, config)
    result = retriever.retrieve("laptop under 500 with long battery")
    print(result.trajectory)
    print(result.evidence_chain)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from src.config.settings import Config, DEFAULT_CONFIG
from src.rl.kg_env import KGEnvironment
from src.rl.actor import Actor
from src.retrieval.stopping import AdaptiveStoppingModule

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """Complete output of one adaptive retrieval episode."""
    query: str
    # Ordered list of traversal steps
    trajectory: List[Dict[str, Any]] = field(default_factory=list)
    # Raw evidence strings collected
    evidence_chain: List[str] = field(default_factory=list)
    # Structured node/relation path for explainability
    reasoning_path: List[Dict[str, Any]] = field(default_factory=list)
    # Total hops taken
    n_hops: int = 0
    # Final uncertainty at stopping point
    final_uncertainty: float = 1.0
    # Total reward accumulated (for evaluation)
    total_reward: float = 0.0
    # Metadata
    anchor_node_id: str = ""
    elapsed_ms: float = 0.0
    stop_reason: str = ""


# ---------------------------------------------------------------------------
# Adaptive Retriever
# ---------------------------------------------------------------------------

class AdaptiveRetriever:
    """
    End-to-end adaptive multi-hop retriever.

    Runs a trained policy over the KG environment and applies adaptive stopping
    to determine when sufficient evidence has been collected.

    Parameters
    ----------
    env      : KGEnvironment
    actor    : Actor (trained)
    stopping : AdaptiveStoppingModule
    config   : Config
    """

    def __init__(
        self,
        env: KGEnvironment,
        actor: Actor,
        stopping: AdaptiveStoppingModule,
        config: Config = DEFAULT_CONFIG,
    ) -> None:
        self.env = env
        self.actor = actor
        self.stopping = stopping
        self.config = config
        self.device = torch.device(config.encoder.device)
        logger.info("AdaptiveRetriever initialised.")

    # ------------------------------------------------------------------
    # Main inference API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        anchor_node_id: Optional[str] = None,
        deterministic: bool = True,
    ) -> RetrievalResult:
        """
        Run adaptive multi-hop retrieval for a query.

        Parameters
        ----------
        query          : English user query
        anchor_node_id : optional anchor; resolved automatically if None
        deterministic  : greedy action selection at inference (True)

        Returns
        -------
        RetrievalResult
        """
        t0 = time.time()
        result = RetrievalResult(query=query)

        # --- Initialise episode ---
        obs, info = self.env.reset(query, anchor_node_id)
        result.anchor_node_id = info.get("anchor_node_id", "")
        query_emb = info["query_repr"].embedding
        episode = self.env.get_episode()

        done = False
        stop_reason = "max_hops"

        while not done:
            # Get dynamic action space
            actions = self.env.get_valid_actions()
            if not actions:
                stop_reason = "no_actions"
                break

            # Policy selects action
            action_idx, log_prob, policy_value, attn_weights = self.actor.act(
                state_vec=obs,
                actions=actions,
                query_emb=query_emb,
                deterministic=deterministic,
            )

            # Check if policy wants to stop
            selected_action = actions[action_idx]

            # --- Minimum-hop guard (inference) ---------------------------------
            # The PPO policy can collapse to "STOP immediately" (hop 0 → no
            # evidence). Until min_hops is reached, override an early STOP and
            # move to the neighbour most relevant to the query so that the agent
            # always collects at least some evidence to reason over.
            cur_hops = episode.hop_count if episode is not None else 0
            min_hops = max(getattr(self.config.stopping, "min_hops", 1), 2)
            if selected_action.is_stop and cur_hops < min_hops:
                non_stop = [a for a in actions if not a.is_stop]
                if non_stop:
                    qb = info["query_repr"].base_embedding
                    def _rel(a):
                        if a.target_base_embedding is None:
                            return -1.0
                        return torch.nn.functional.cosine_similarity(
                            qb.unsqueeze(0), a.target_base_embedding.unsqueeze(0)
                        ).item()
                    selected_action = max(non_stop, key=_rel)
                    action_idx = selected_action.index

            if selected_action.is_stop:
                stop_reason = "policy_stop"
                done = True
                break

            # Execute action in environment
            obs, reward, done, step_info = self.env.step(action_idx)
            result.total_reward += reward

            # Get updated episode state
            episode = self.env.get_episode()

            # Evaluate adaptive stopping
            if episode is not None:
                # Build action embs for MC Dropout (if needed)
                action_embs, _ = self.actor._stack_action_embeddings(actions)
                stop_signal = self.stopping.evaluate(
                    episode=episode,
                    new_node_emb=episode.current_node.embedding if episode.current_node else None,
                    policy_value=policy_value,
                    state_vec=obs,
                    action_embs=action_embs,
                    query_emb=query_emb,
                )
                if stop_signal.should_stop and episode.hop_count >= min_hops:
                    stop_reason = f"adaptive_stop: {stop_signal.reason}"
                    done = True

            # Log reasoning path step
            result.reasoning_path.append({
                "hop": step_info.hop,
                "from_node": episode.visited_steps[-2].node_id if len(episode.visited_steps) > 1 else result.anchor_node_id,
                "to_node": step_info.node_id,
                "node_type": step_info.node_type,
                "relation": step_info.relation_type,
                "edge_weight": actions[action_idx].edge_weight,
                "attn_weight": attn_weights[action_idx].item() if attn_weights is not None else 0.0,
                "uncertainty": step_info.uncertainty,
                "evidence": step_info.evidence_text,
            })

        # Collect final results
        result.trajectory = self.env.get_trajectory()
        result.evidence_chain = self.env.get_evidence()
        result.n_hops = episode.hop_count if episode else 0
        result.final_uncertainty = episode.uncertainty_score if episode else 1.0
        result.elapsed_ms = (time.time() - t0) * 1000
        result.stop_reason = stop_reason

        logger.info(
            "Retrieval done: query='%s...' hops=%d evidence=%d U=%.3f (%.1fms)",
            query[:50], result.n_hops, len(result.evidence_chain),
            result.final_uncertainty, result.elapsed_ms,
        )
        return result

    def retrieve_batch(
        self,
        queries: List[str],
        deterministic: bool = True,
    ) -> List[RetrievalResult]:
        """Retrieve for a list of queries sequentially."""
        return [self.retrieve(q, deterministic=deterministic) for q in queries]