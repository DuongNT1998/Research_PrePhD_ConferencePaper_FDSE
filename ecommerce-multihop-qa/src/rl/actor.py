"""
src/rl/actor.py

Actor wrapper for the retrieval policy.

Wraps PolicyNetwork and exposes a clean API for the PPO trainer and the
KGEnvironment.  Handles:
  - Action sampling (stochastic during training, deterministic at inference)
  - Log-probability computation for PPO ratio
  - Action embedding stacking from dynamic action lists
  - Imitation learning (supervised trajectory cloning, Stage 1)
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config.settings import Config, DEFAULT_CONFIG
from src.rl.policy_network import PolicyNetwork
from src.rl.kg_env import Action

logger = logging.getLogger(__name__)


class Actor(nn.Module):
    """
    Actor (policy) for the adaptive retrieval RL agent.

    Parameters
    ----------
    config : Config
    """

    def __init__(self, config: Config = DEFAULT_CONFIG) -> None:
        super().__init__()
        self.config = config
        self.device = torch.device(config.encoder.device)
        self.policy = PolicyNetwork(config).to(self.device)
        logger.info("Actor initialised.")

    # ------------------------------------------------------------------
    # Core action selection
    # ------------------------------------------------------------------

    def act(
        self,
        state_vec: torch.Tensor,
        actions: List[Action],
        query_emb: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[int, float, float, torch.Tensor]:
        """
        Select an action from the dynamic action list.

        Parameters
        ----------
        state_vec     : (state_dim,)
        actions       : list of Action (dynamic action space)
        query_emb     : (query_dim,)
        deterministic : greedy if True, sample if False

        Returns
        -------
        action_idx   : int (index into actions list)
        log_prob     : float
        value        : float (critic estimate)
        attn_weights : (N,) attention over actions (for logging)
        """
        action_embs, action_mask = self._stack_action_embeddings(actions)
        return self.policy.select_action(
            state_vec=state_vec.to(self.device),
            action_embs=action_embs,
            query_emb=query_emb.to(self.device),
            action_mask=action_mask,
            deterministic=deterministic,
        )

    def evaluate_actions(
        self,
        state_vecs: torch.Tensor,
        action_embs_batch: torch.Tensor,
        query_embs: torch.Tensor,
        action_indices: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Re-evaluate stored (s, a) pairs for PPO update.

        Parameters
        ----------
        state_vecs       : (B, state_dim)
        action_embs_batch: (B, N_max, action_dim)
        query_embs       : (B, query_dim)
        action_indices   : (B,)
        action_masks     : (B, N_max) bool | None

        Returns
        -------
        log_probs : (B,)
        entropy   : (B,)
        values    : (B,)
        """
        return self.policy.get_log_probs(
            state_vec=state_vecs.to(self.device),
            action_embs=action_embs_batch.to(self.device),
            query_emb=query_embs.to(self.device),
            action_indices=action_indices.to(self.device),
            action_mask=action_masks.to(self.device) if action_masks is not None else None,
        )

    # ------------------------------------------------------------------
    # Imitation learning (Stage 1)
    # ------------------------------------------------------------------

    def imitation_loss(
        self,
        state_vec: torch.Tensor,
        actions: List[Action],
        query_emb: torch.Tensor,
        teacher_action_idx: int,
    ) -> torch.Tensor:
        """
        Cross-entropy loss against a teacher trajectory action.

        Parameters
        ----------
        teacher_action_idx : int  index of the correct action in `actions`

        Returns
        -------
        loss : scalar tensor
        """
        action_embs, action_mask = self._stack_action_embeddings(actions)
        logits, probs, _, _ = self.policy(
            state_vec=state_vec.to(self.device),
            action_embs=action_embs,
            query_emb=query_emb.to(self.device),
            action_mask=action_mask,
        )
        target = torch.tensor(teacher_action_idx, dtype=torch.long, device=self.device)
        return F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _stack_action_embeddings(
        self, actions: List[Action]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stack action embeddings into a tensor and build a validity mask.

        Returns
        -------
        action_embs : (N, action_embed_dim)
        mask        : (N,) bool — all True (all actions valid by construction)
        """
        embs = []
        for a in actions:
            if a.action_embedding is not None:
                embs.append(a.action_embedding.to(self.device))
            else:
                # Fallback: zero vector if embedding not pre-computed
                action_dim = (
                    self.config.encoder.node_feature_dim
                    + self.config.state.edge_type_dim
                )
                embs.append(torch.zeros(action_dim, device=self.device))
        action_embs = torch.stack(embs, dim=0)          # (N, action_dim)
        mask = torch.ones(len(actions), dtype=torch.bool, device=self.device)
        return action_embs, mask

    def get_parameters(self):
        return self.policy.parameters()