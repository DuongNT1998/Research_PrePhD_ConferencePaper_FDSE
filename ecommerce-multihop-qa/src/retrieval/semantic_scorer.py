"""
src/retrieval/semantic_scorer.py

Semantic Relevance Scorer.

Computes relevance scores between a query representation and KG node
embeddings.  Used in:
  - Anchor node resolution (find best starting node)
  - Reward computation (retrieval relevance reward)
  - Action scoring within the policy (soft-attention over candidates)
  - Stopping criterion (evidence saturation)

Scoring modes
-------------
cosine     — normalised dot product (default)
bilinear   — learned W matrix: score = q^T W n
mlp        — MLP(concat(q, n)) → scalar

All modes return a float in [0, 1] for reward compatibility.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config.settings import Config, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class BilinearScorer(nn.Module):
    """Learned bilinear relevance: score = sigmoid(q^T W n + b)."""

    def __init__(self, query_dim: int, node_dim: int) -> None:
        super().__init__()
        self.W = nn.Linear(query_dim, node_dim, bias=True)

    def forward(self, query: torch.Tensor, node: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        query : (query_dim,) or (B, query_dim)
        node  : (node_dim,)  or (B, node_dim)

        Returns
        -------
        score : scalar tensor
        """
        q_proj = self.W(query)               # (..., node_dim)
        dot = (q_proj * node).sum(dim=-1)    # (...,)
        return torch.sigmoid(dot)


class MLPScorer(nn.Module):
    """MLP relevance scorer: score = MLP([q; n]) → [0,1]."""

    def __init__(self, query_dim: int, node_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(query_dim + node_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, query: torch.Tensor, node: torch.Tensor) -> torch.Tensor:
        x = torch.cat([query, node], dim=-1)
        return self.net(x).squeeze(-1)


class SemanticScorer:
    """
    Compute semantic relevance scores between query and node embeddings.

    Parameters
    ----------
    mode : str
        "cosine" (default) | "bilinear" | "mlp"
    config : Config
    """

    def __init__(
        self,
        mode: str = "cosine",
        config: Config = DEFAULT_CONFIG,
    ) -> None:
        self.mode = mode
        self.config = config
        self.device = torch.device(config.encoder.device)

        self._bilinear: Optional[BilinearScorer] = None
        self._mlp: Optional[MLPScorer] = None

        if mode == "bilinear":
            self._bilinear = BilinearScorer(
                config.state.query_dim,
                config.encoder.node_feature_dim,
            ).to(self.device)
            logger.info("SemanticScorer: bilinear mode")
        elif mode == "mlp":
            self._mlp = MLPScorer(
                config.state.query_dim,
                config.encoder.node_feature_dim,
            ).to(self.device)
            logger.info("SemanticScorer: MLP mode")
        else:
            logger.info("SemanticScorer: cosine mode")

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def score(
        self,
        query_emb: torch.Tensor,
        node_emb: torch.Tensor,
    ) -> float:
        """
        Compute scalar relevance score in [0,1].

        Parameters
        ----------
        query_emb : (query_dim,)
        node_emb  : (node_dim,)

        Returns
        -------
        float
        """
        q = query_emb.to(self.device)
        n = node_emb.to(self.device)

        if self.mode == "cosine":
            cos = F.cosine_similarity(q.unsqueeze(0), n.unsqueeze(0)).item()
            return (cos + 1.0) / 2.0   # map [-1,1] → [0,1]

        elif self.mode == "bilinear" and self._bilinear is not None:
            with torch.no_grad():
                return self._bilinear(q, n).item()

        elif self.mode == "mlp" and self._mlp is not None:
            with torch.no_grad():
                return self._mlp(q, n).item()

        # fallback
        cos = F.cosine_similarity(q.unsqueeze(0), n.unsqueeze(0)).item()
        return (cos + 1.0) / 2.0

    def score_batch(
        self,
        query_emb: torch.Tensor,
        node_embs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score multiple nodes against one query simultaneously.

        Parameters
        ----------
        query_emb : (query_dim,)
        node_embs : (N, node_dim)

        Returns
        -------
        scores : (N,) in [0,1]
        """
        q = query_emb.to(self.device).unsqueeze(0).expand(node_embs.shape[0], -1)
        n = node_embs.to(self.device)

        if self.mode == "cosine":
            cos = F.cosine_similarity(q, n, dim=-1)   # (N,)
            return (cos + 1.0) / 2.0

        elif self.mode == "bilinear" and self._bilinear is not None:
            with torch.no_grad():
                return self._bilinear(q, n)

        elif self.mode == "mlp" and self._mlp is not None:
            with torch.no_grad():
                return self._mlp(q, n)

        cos = F.cosine_similarity(q, n, dim=-1)
        return (cos + 1.0) / 2.0

    def compute_evidence_saturation(
        self,
        new_node_emb: torch.Tensor,
        existing_embs: List[torch.Tensor],
    ) -> float:
        """
        Measure how much new information a candidate node would add.
        Low saturation delta → evidence already covers this node → stop sooner.

        Returns
        -------
        delta ∈ [0,1]: 0 = fully redundant, 1 = completely novel
        """
        if not existing_embs:
            return 1.0
        stack = torch.stack(existing_embs, dim=0).to(self.device)
        centroid = stack.mean(dim=0)
        cos = F.cosine_similarity(
            new_node_emb.to(self.device).unsqueeze(0),
            centroid.unsqueeze(0),
        ).item()
        # High cosine → similar to existing evidence → low novelty
        novelty = 1.0 - (cos + 1.0) / 2.0
        return float(novelty)

    def get_learnable_parameters(self):
        params = []
        if self._bilinear:
            params += list(self._bilinear.parameters())
        if self._mlp:
            params += list(self._mlp.parameters())
        return params