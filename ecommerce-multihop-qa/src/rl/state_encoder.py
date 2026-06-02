"""
src/rl/state_encoder.py

State Encoder for PPO training batch processing.

During PPO update we need to re-encode states efficiently.
This module provides batch-safe encoding utilities used by the PPO trainer.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import torch
import torch.nn as nn

from src.config.settings import Config, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class StateEncoder(nn.Module):
    """
    Lightweight MLP that re-projects stored state vectors.

    In our architecture state vectors are already assembled by StateBuilder,
    so this encoder primarily handles normalisation and optional
    learnable projection for the PPO batch pass.

    Parameters
    ----------
    config : Config
    """

    def __init__(self, config: Config = DEFAULT_CONFIG) -> None:
        super().__init__()
        self.config = config
        state_dim = config.state.total_dim

        # Layer norm for stability during batch training
        self.norm = nn.LayerNorm(state_dim)

        # Optional learnable re-projection (identity initially)
        self.proj = nn.Linear(state_dim, state_dim, bias=False)
        nn.init.eye_(self.proj.weight)

        logger.debug("StateEncoder: state_dim=%d", state_dim)

    def forward(self, state_vecs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state_vecs : (B, state_dim) or (state_dim,)

        Returns
        -------
        encoded : same shape
        """
        normed = self.norm(state_vecs)
        return self.proj(normed)

    def encode_batch(self, states: List[torch.Tensor]) -> torch.Tensor:
        """Stack and encode a list of state vectors."""
        batch = torch.stack(states, dim=0)
        return self.forward(batch)