"""
src/rl/checkpoint.py

Checkpoint management for model saving and loading.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from src.config.settings import Config, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Save and load model checkpoints.

    Directory layout
    ----------------
    outputs/checkpoints/
        best/
            policy.pt
            optimizer.pt
            meta.json
        latest/
            policy.pt
            ...
        imitation/
            policy.pt
            ...
    """

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        config: Config = DEFAULT_CONFIG,
    ) -> None:
        self.base_dir = base_dir or config.paths.checkpoints
        self.base_dir = Path(self.base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.info("CheckpointManager at %s", self.base_dir)

    def save(
        self,
        model: nn.Module,
        optimiser: torch.optim.Optimizer,
        step: int,
        metrics: Dict[str, Any],
        tag: str = "latest",
    ) -> Path:
        """Save model + optimiser + metadata."""
        save_dir = self.base_dir / tag
        save_dir.mkdir(parents=True, exist_ok=True)

        model_path = save_dir / "policy.pt"
        opt_path = save_dir / "optimizer.pt"
        meta_path = save_dir / "meta.json"

        torch.save(model.state_dict(), model_path)
        torch.save(optimiser.state_dict(), opt_path)

        meta = {"step": step, "tag": tag, **metrics}
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("Checkpoint saved: %s (step=%d)", tag, step)
        return save_dir

    def load(
        self,
        model: nn.Module,
        optimiser: Optional[torch.optim.Optimizer] = None,
        tag: str = "best",
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """Load model (and optionally optimiser) from checkpoint."""
        load_dir = self.base_dir / tag
        model_path = load_dir / "policy.pt"
        opt_path = load_dir / "optimizer.pt"
        meta_path = load_dir / "meta.json"

        if not model_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {load_dir}")

        map_loc = device or torch.device("cpu")
        model.load_state_dict(torch.load(model_path, map_location=map_loc))
        logger.info("Model loaded from %s", load_dir)

        if optimiser is not None and opt_path.exists():
            optimiser.load_state_dict(torch.load(opt_path, map_location=map_loc))
            logger.info("Optimiser loaded from %s", load_dir)

        meta: Dict[str, Any] = {}
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)

        return meta

    def latest_exists(self) -> bool:
        return (self.base_dir / "latest" / "policy.pt").exists()

    def best_exists(self) -> bool:
        return (self.base_dir / "best" / "policy.pt").exists()