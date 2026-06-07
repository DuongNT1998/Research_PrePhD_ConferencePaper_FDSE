"""
src/rl/replay_buffer.py

Rollout Buffer for on-policy PPO training.

Stores per-step transitions across multiple episodes:
  (state_vec, action_embs, query_emb, action_idx, log_prob, reward, value, done)

Supports:
- Variable-length action spaces per step (padded to max N in the buffer)
- GAE advantage computation via Critic
- Mini-batch sampling for PPO epochs
- Imitation trajectory storage (Stage 1)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch

from src.config.settings import Config, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single transition
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    state_vec: torch.Tensor          # (state_dim,)
    action_embs: torch.Tensor        # (N, action_dim) — current step's action space
    query_emb: torch.Tensor          # (query_dim,)
    action_idx: int
    log_prob: float
    reward: float
    value: float
    done: bool
    # For imitation learning: teacher label
    teacher_action_idx: Optional[int] = None
    # Metadata
    hop: int = 0
    node_id: str = ""
    node_type: str = ""
    relation: str = ""
    uncertainty: float = 1.0
    reward_breakdown: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# RolloutBuffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """
    On-policy rollout buffer for PPO.

    Collects transitions from multiple episodes, computes GAE, and yields
    mini-batches for the PPO update step.

    Parameters
    ----------
    config : Config
    """

    def __init__(self, config: Config = DEFAULT_CONFIG) -> None:
        self.config = config
        self.device = torch.device(config.encoder.device)
        self._transitions: List[Transition] = []
        self._episode_starts: List[int] = []   # indices where new episodes begin
        self._current_episode_start: int = 0
        logger.debug("RolloutBuffer initialised.")

    # ------------------------------------------------------------------
    # Collection API
    # ------------------------------------------------------------------

    def add(self, transition: Transition) -> None:
        """Add one transition to the buffer."""
        self._transitions.append(transition)

    def start_episode(self) -> None:
        """Mark the start of a new episode."""
        self._current_episode_start = len(self._transitions)
        self._episode_starts.append(self._current_episode_start)

    def end_episode(self) -> None:
        """Nothing required; episode boundary is inferred from done flags."""
        pass

    def __len__(self) -> int:
        return len(self._transitions)

    def is_ready(self) -> bool:
        """True when buffer has enough transitions for a PPO update."""
        return len(self._transitions) >= self.config.ppo.update_every

    def clear(self) -> None:
        self._transitions.clear()
        self._episode_starts.clear()
        self._current_episode_start = 0
        logger.debug("RolloutBuffer cleared.")

    # ------------------------------------------------------------------
    # GAE computation
    # ------------------------------------------------------------------

    def compute_advantages(
        self, gamma: float, gae_lambda: float, next_value: float = 0.0
    ) -> None:
        """
        Compute GAE advantages and returns in-place over all stored transitions.
        Must be called before get_mini_batches().
        """
        T = len(self._transitions)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(T)):
            trans = self._transitions[t]
            mask = 0.0 if trans.done else 1.0
            next_val = (
                next_value if t == T - 1
                else self._transitions[t + 1].value
            )
            delta = trans.reward + gamma * next_val * mask - trans.value
            gae = delta + gamma * gae_lambda * mask * gae
            advantages[t] = gae

        # Store as attributes on transitions
        for t, trans in enumerate(self._transitions):
            trans._advantage = float(advantages[t])
            trans._return = float(advantages[t] + trans.value)

        logger.debug(
            "GAE computed: mean_adv=%.4f std_adv=%.4f",
            advantages.mean(), advantages.std()
        )

    # ------------------------------------------------------------------
    # Mini-batch sampling
    # ------------------------------------------------------------------

    def get_mini_batches(
        self, mini_batch_size: int
    ) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Yield mini-batches from the buffer for PPO update epochs.

        Each batch contains padded tensors so variable-size action spaces
        are handled uniformly.

        Yields
        ------
        dict with keys:
            state_vecs, action_embs, query_embs, action_indices,
            old_log_probs, old_values, advantages, returns, action_masks
        """
        T = len(self._transitions)
        indices = np.random.permutation(T)
        max_actions = max(t.action_embs.shape[0] for t in self._transitions)
        action_dim = self._transitions[0].action_embs.shape[1]

        for start in range(0, T, mini_batch_size):
            batch_idx = indices[start: start + mini_batch_size]
            if len(batch_idx) == 0:
                continue
            batch = [self._transitions[i] for i in batch_idx]

            state_dim = batch[0].state_vec.shape[0]
            query_dim = batch[0].query_emb.shape[0]
            B = len(batch)

            state_vecs = torch.zeros(B, state_dim, device=self.device)
            query_embs = torch.zeros(B, query_dim, device=self.device)
            action_embs = torch.zeros(B, max_actions, action_dim, device=self.device)
            action_masks = torch.zeros(B, max_actions, dtype=torch.bool, device=self.device)
            action_indices = torch.zeros(B, dtype=torch.long, device=self.device)
            old_log_probs = torch.zeros(B, device=self.device)
            old_values = torch.zeros(B, device=self.device)
            advantages = torch.zeros(B, device=self.device)
            returns = torch.zeros(B, device=self.device)

            for i, t in enumerate(batch):
                state_vecs[i] = t.state_vec.to(self.device)
                query_embs[i] = t.query_emb.to(self.device)
                n = t.action_embs.shape[0]
                action_embs[i, :n, :] = t.action_embs.to(self.device)
                action_masks[i, :n] = True
                action_indices[i] = t.action_idx
                old_log_probs[i] = t.log_prob
                old_values[i] = t.value
                advantages[i] = getattr(t, "_advantage", 0.0)
                returns[i] = getattr(t, "_return", t.reward)

            # Normalise advantages within batch.
            # Use population std (unbiased=False) and skip when B==1, otherwise
            # std() of a single element is NaN and would poison the gradients.
            if advantages.numel() > 1:
                adv_mean = advantages.mean()
                adv_std = advantages.std(unbiased=False) + 1e-8
                advantages = (advantages - adv_mean) / adv_std

            yield {
                "state_vecs": state_vecs,
                "action_embs": action_embs,
                "query_embs": query_embs,
                "action_indices": action_indices,
                "old_log_probs": old_log_probs,
                "old_values": old_values,
                "advantages": advantages,
                "returns": returns,
                "action_masks": action_masks,
            }

    # ------------------------------------------------------------------
    # Imitation trajectory storage
    # ------------------------------------------------------------------

    def get_imitation_batch(
        self, batch_size: int
    ) -> Optional[List[Transition]]:
        """
        Return a random batch of transitions that have teacher labels.
        Used in Stage 1 imitation learning.
        """
        labeled = [t for t in self._transitions if t.teacher_action_idx is not None]
        if not labeled:
            return None
        indices = np.random.choice(len(labeled), min(batch_size, len(labeled)), replace=False)
        return [labeled[i] for i in indices]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, float]:
        if not self._transitions:
            return {}
        rewards = [t.reward for t in self._transitions]
        hops = [t.hop for t in self._transitions]
        return {
            "n_transitions": len(self._transitions),
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_hop": float(np.mean(hops)),
            "max_hop": float(np.max(hops)),
        }