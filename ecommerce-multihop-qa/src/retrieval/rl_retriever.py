"""
src/retrieval/rl_retriever.py

RL-Integrated Retriever.

This module is the bridge between the trained RL policy (Actor) and the
KGEnvironment. It handles:

1. Single-query inference: run policy on one query, return full RetrievalResult
2. Batch inference: efficient sequential retrieval over a query list
3. Episode data collection for PPO buffer: collect (s, a, r, s') tuples
4. Teacher-forced episode collection for imitation learning
5. Step-level diagnostics: per-hop attention weights, uncertainty trace

The distinction from AdaptiveRetriever is:
- AdaptiveRetriever is a clean high-level API (for production / evaluation)
- RLRetriever is the training-aware component with direct buffer integration

Design
------
    RLRetriever.run_episode()
        → calls Actor.act() at each step
        → records Transition into RolloutBuffer
        → calls AdaptiveStoppingModule.evaluate() for early stopping
        → returns EpisodeRecord (full trajectory + buffer transitions)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from src.config.settings import Config, DEFAULT_CONFIG
from src.rl.kg_env import KGEnvironment, Action
from src.rl.actor import Actor
from src.rl.replay_buffer import RolloutBuffer, Transition
from src.retrieval.stopping import AdaptiveStoppingModule, StoppingSignal
from src.retrieval.state_builder import EpisodeState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EpisodeRecord: full record of one retrieval episode
# ---------------------------------------------------------------------------

@dataclass
class EpisodeRecord:
    """Complete record of one retrieval episode for diagnostics and training."""
    query: str
    anchor_node_id: str
    # Per-step data
    transitions: List[Transition] = field(default_factory=list)
    stopping_signals: List[StoppingSignal] = field(default_factory=list)
    # Attention weights per step: (step_idx → (N_actions,) tensor)
    attention_traces: List[torch.Tensor] = field(default_factory=list)
    # Node sequence
    node_sequence: List[str] = field(default_factory=list)
    relation_sequence: List[str] = field(default_factory=list)
    node_type_sequence: List[str] = field(default_factory=list)
    # Aggregated metrics
    total_reward: float = 0.0
    n_hops: int = 0
    final_uncertainty: float = 1.0
    stop_reason: str = "unknown"
    elapsed_ms: float = 0.0
    # Evidence chain
    evidence_chain: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# RLRetriever
# ---------------------------------------------------------------------------

class RLRetriever:
    """
    Training-aware RL retriever that integrates policy, environment,
    stopping module, and replay buffer.

    Parameters
    ----------
    env             : KGEnvironment
    actor           : Actor
    stopping        : AdaptiveStoppingModule
    buffer          : RolloutBuffer | None  (None = inference only)
    config          : Config
    """

    def __init__(
        self,
        env: KGEnvironment,
        actor: Actor,
        stopping: AdaptiveStoppingModule,
        buffer: Optional[RolloutBuffer] = None,
        config: Config = DEFAULT_CONFIG,
    ) -> None:
        self.env = env
        self.actor = actor
        self.stopping = stopping
        self.buffer = buffer
        self.config = config
        self.device = torch.device(config.encoder.device)
        logger.info("RLRetriever initialised (buffer=%s).", buffer is not None)

    # ------------------------------------------------------------------
    # Core episode runner
    # ------------------------------------------------------------------

    def run_episode(
        self,
        query: str,
        anchor_node_id: Optional[str] = None,
        deterministic: bool = False,
        collect_transitions: bool = True,
        teacher_actions: Optional[List[int]] = None,
    ) -> EpisodeRecord:
        """
        Run one complete retrieval episode.

        Parameters
        ----------
        query              : English user query
        anchor_node_id     : optional anchor; auto-resolved if None
        deterministic      : greedy action selection (True for inference)
        collect_transitions: whether to push transitions into the buffer
        teacher_actions    : if provided, override policy actions at each step
                             (used for imitation learning data collection)

        Returns
        -------
        EpisodeRecord with full trajectory, transitions, and metrics.
        """
        t0 = time.time()
        record = EpisodeRecord(query=query, anchor_node_id="")

        # --- Init episode ---
        obs, info = self.env.reset(query, anchor_node_id)
        record.anchor_node_id = info.get("anchor_node_id", "")
        query_emb: torch.Tensor = info["query_repr"].embedding
        episode: Optional[EpisodeState] = self.env.get_episode()

        if collect_transitions and self.buffer is not None:
            self.buffer.start_episode()

        done = False
        step_idx = 0
        stop_reason = "max_hops_safety"

        while not done:
            actions: List[Action] = self.env.get_valid_actions()
            if not actions:
                stop_reason = "no_actions"
                break

            # --- Action selection ---
            if teacher_actions is not None and step_idx < len(teacher_actions):
                # Teacher-forced (imitation)
                action_idx = min(teacher_actions[step_idx], len(actions) - 1)
                action_embs, action_mask = self.actor._stack_action_embeddings(actions)
                with torch.no_grad():
                    _, _, policy_value, attn_w = self.actor.policy(
                        state_vec=obs.to(self.device),
                        action_embs=action_embs,
                        query_emb=query_emb.to(self.device),
                        action_mask=action_mask,
                    )
                log_prob = -1.0  # not used for imitation
                policy_value = policy_value.item()
            else:
                action_embs, action_mask = self.actor._stack_action_embeddings(actions)
                action_idx, log_prob, policy_value, attn_w = self.actor.act(
                    state_vec=obs,
                    actions=actions,
                    query_emb=query_emb,
                    deterministic=deterministic,
                )

            record.attention_traces.append(attn_w.detach().cpu())

            # --- Adaptive stopping check (before executing action) ---
            if episode is not None and not actions[action_idx].is_stop:
                stop_signal = self.stopping.evaluate(
                    episode=episode,
                    new_node_emb=None,   # not yet traversed; use current state
                    policy_value=policy_value,
                    state_vec=obs,
                    action_embs=action_embs,
                    query_emb=query_emb,
                )
                record.stopping_signals.append(stop_signal)
                if stop_signal.should_stop:
                    stop_reason = f"adaptive: {stop_signal.reason}"
                    done = True
                    break

            # --- Execute action in env ---
            obs_next, reward, env_done, step_info = self.env.step(action_idx)
            record.total_reward += reward

            episode = self.env.get_episode()
            done = env_done or actions[action_idx].is_stop

            if actions[action_idx].is_stop:
                stop_reason = "policy_stop"

            # --- Record trajectory ---
            record.node_sequence.append(step_info.node_id)
            record.relation_sequence.append(step_info.relation_type)
            record.node_type_sequence.append(step_info.node_type)
            if step_info.evidence_text:
                record.evidence_chain.append(step_info.evidence_text)

            # --- Store transition in buffer ---
            if collect_transitions and self.buffer is not None:
                teacher_label = (
                    teacher_actions[step_idx]
                    if teacher_actions is not None and step_idx < len(teacher_actions)
                    else None
                )
                transition = Transition(
                    state_vec=obs.detach().cpu(),
                    action_embs=action_embs.detach().cpu(),
                    query_emb=query_emb.detach().cpu(),
                    action_idx=action_idx,
                    log_prob=log_prob,
                    reward=reward,
                    value=policy_value,
                    done=done,
                    teacher_action_idx=teacher_label,
                    hop=step_info.hop,
                    node_id=step_info.node_id,
                    node_type=step_info.node_type,
                    relation=step_info.relation_type,
                    uncertainty=step_info.uncertainty,
                    reward_breakdown=step_info.reward_breakdown,
                )
                self.buffer.add(transition)
                record.transitions.append(transition)

            obs = obs_next
            step_idx += 1

        if collect_transitions and self.buffer is not None:
            self.buffer.end_episode()

        # --- Finalise record ---
        record.n_hops = episode.hop_count if episode else step_idx
        record.final_uncertainty = episode.uncertainty_score if episode else 1.0
        record.stop_reason = stop_reason
        record.elapsed_ms = (time.time() - t0) * 1000

        logger.debug(
            "Episode done: hops=%d reward=%.4f uncertainty=%.3f reason=%s (%.1fms)",
            record.n_hops, record.total_reward, record.final_uncertainty,
            stop_reason, record.elapsed_ms,
        )
        return record

    # ------------------------------------------------------------------
    # Batch inference
    # ------------------------------------------------------------------

    def run_batch(
        self,
        queries: List[str],
        deterministic: bool = True,
        collect_transitions: bool = False,
    ) -> List[EpisodeRecord]:
        """
        Run retrieval for a list of queries.

        Parameters
        ----------
        queries            : list of English queries
        deterministic      : greedy action selection
        collect_transitions: whether to store to buffer (usually False at eval)

        Returns
        -------
        list of EpisodeRecord
        """
        records: List[EpisodeRecord] = []
        for i, q in enumerate(queries):
            logger.info("Batch retrieval %d/%d: '%s'", i + 1, len(queries), q[:60])
            record = self.run_episode(
                query=q,
                deterministic=deterministic,
                collect_transitions=collect_transitions,
            )
            records.append(record)
        return records

    # ------------------------------------------------------------------
    # Collect rollouts for PPO training
    # ------------------------------------------------------------------

    def collect_rollouts(
        self,
        queries: List[str],
        n_steps: int,
        deterministic: bool = False,
    ) -> Dict[str, float]:
        """
        Collect at least n_steps transitions into the buffer.
        Called by PPOTrainer before each update.

        Parameters
        ----------
        queries      : pool of training queries
        n_steps      : minimum transitions to collect
        deterministic: stochastic collection (False) for exploration

        Returns
        -------
        stats dict: mean_reward, n_episodes, n_transitions
        """
        import random

        if self.buffer is None:
            raise RuntimeError("RLRetriever needs a buffer for rollout collection.")

        total_reward = 0.0
        n_episodes = 0

        while len(self.buffer) < n_steps:
            query = random.choice(queries)
            record = self.run_episode(
                query=query,
                deterministic=deterministic,
                collect_transitions=True,
            )
            total_reward += record.total_reward
            n_episodes += 1

        return {
            "mean_reward": total_reward / max(n_episodes, 1),
            "n_episodes": n_episodes,
            "n_transitions": len(self.buffer),
        }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_attention_summary(self, record: EpisodeRecord) -> List[Dict[str, Any]]:
        """
        Return per-hop attention weight summaries for interpretability.
        """
        summary = []
        for step_idx, (attn, node_id, rel) in enumerate(zip(
            record.attention_traces,
            record.node_sequence,
            record.relation_sequence,
        )):
            top_k = min(3, attn.shape[0])
            top_values, top_indices = torch.topk(attn, top_k)
            summary.append({
                "hop": step_idx + 1,
                "node_id": node_id,
                "relation": rel,
                "top_attention_values": top_values.tolist(),
                "top_attention_indices": top_indices.tolist(),
                "entropy": -(attn * (attn + 1e-8).log()).sum().item(),
            })
        return summary

    def episode_record_to_dict(self, record: EpisodeRecord) -> Dict[str, Any]:
        """Serialise EpisodeRecord to JSON-compatible dict."""
        return {
            "query": record.query,
            "anchor_node_id": record.anchor_node_id,
            "n_hops": record.n_hops,
            "total_reward": record.total_reward,
            "final_uncertainty": record.final_uncertainty,
            "stop_reason": record.stop_reason,
            "elapsed_ms": record.elapsed_ms,
            "node_sequence": record.node_sequence,
            "relation_sequence": record.relation_sequence,
            "node_type_sequence": record.node_type_sequence,
            "evidence_chain": record.evidence_chain,
            "attention_summary": self.get_attention_summary(record),
        }