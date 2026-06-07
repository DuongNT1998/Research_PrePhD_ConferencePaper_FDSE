"""
src/rl/critic.py

Critic (Value Network) with Generalised Advantage Estimation (GAE).

The critic shares the PolicyNetwork backbone; its value head produces V(s).
This module also provides:
- compute_gae(): GAE advantage estimation from a trajectory rollout
- compute_returns(): discounted return computation
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import torch
import torch.nn as nn

from src.config.settings import Config, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class Critic(nn.Module):
    """
    Thin wrapper around the value head of PolicyNetwork.
    In Actor-Critic with shared backbone, the Critic does NOT hold its own
    separate parameters — it references the policy's critic_head directly.
    This class provides GAE computation utilities.

    Parameters
    ----------
    config : Config
    """

    def __init__(self, config: Config = DEFAULT_CONFIG) -> None:
        super().__init__()
        self.config = config
        self.device = torch.device(config.encoder.device)
        self.gamma = config.ppo.gamma
        self.gae_lambda = config.ppo.gae_lambda
        logger.info("Critic (GAE utility) initialised.")

    # ------------------------------------------------------------------
    # GAE
    # ------------------------------------------------------------------

    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        next_value: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalised Advantage Estimates and discounted returns.

        Parameters
        ----------
        rewards    : list of per-step rewards (T,)
        values     : list of V(s_t) estimates   (T,)
        dones      : list of episode-end flags  (T,)
        next_value : V(s_{T+1}), 0 for terminal episodes

        Returns
        -------
        advantages : (T,) tensor
        returns    : (T,) tensor
        """
        T = len(rewards)
        advantages = torch.zeros(T, device=self.device)
        gae = 0.0

        for t in reversed(range(T)):
            next_val = next_value if t == T - 1 else values[t + 1]
            mask = 0.0 if dones[t] else 1.0
            delta = rewards[t] + self.gamma * next_val * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae

        returns = advantages + torch.tensor(values, dtype=torch.float32, device=self.device)
        return advantages, returns

    def compute_returns(
        self,
        rewards: List[float],
        dones: List[bool],
        next_value: float = 0.0,
    ) -> torch.Tensor:
        """
        Simple discounted return (no GAE).
        Returns (T,) tensor.
        """
        T = len(rewards)
        returns = torch.zeros(T, device=self.device)
        R = next_value
        for t in reversed(range(T)):
            mask = 0.0 if dones[t] else 1.0
            R = rewards[t] + self.gamma * R * mask
            returns[t] = R
        return returns

    def value_loss(
        self,
        predicted_values: torch.Tensor,
        returns: torch.Tensor,
        old_values: Optional[torch.Tensor] = None,
        clip_range: float = 0.2,
    ) -> torch.Tensor:
        """
        Clipped value loss (PPO-style).

        Parameters
        ----------
        predicted_values : (B,) new V(s) estimates
        returns          : (B,) target returns
        old_values       : (B,) V(s) at collection time (for clipping)
        clip_range       : PPO clip epsilon

        Returns
        -------
        loss : scalar tensor
        """
        vf_loss_unclipped = (predicted_values - returns).pow(2)
        if old_values is not None:
            v_clipped = old_values + (predicted_values - old_values).clamp(
                -clip_range, clip_range
            )
            vf_loss_clipped = (v_clipped - returns).pow(2)
            vf_loss = torch.max(vf_loss_unclipped, vf_loss_clipped).mean()
        else:
            vf_loss = vf_loss_unclipped.mean()
        return vf_loss

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Critic is a utility class. Value estimates come from PolicyNetwork."
        )


# Needed by value_loss signature
from typing import Optional