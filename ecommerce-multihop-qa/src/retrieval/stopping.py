"""
src/retrieval/stopping.py

Adaptive Stopping Module with Uncertainty Estimation.

This is a key contribution of the research:
"The system learns WHEN to stop retrieval based on uncertainty,
not a fixed hop count."

Stopping criteria (all must pass to stop early):
-------------------------------------------------
1. Uncertainty score < uncertainty_threshold
2. Confidence score > confidence_threshold  (from policy value head)
3. Evidence saturation delta < saturation_threshold  (marginal gain low)
4. Min hops constraint satisfied

Uncertainty estimation methods
-------------------------------
A. MC Dropout  — run policy forward N times with dropout active → variance
B. Evidence centroid variance — spread of collected node embeddings
C. Policy value confidence — critic value mapped to [0,1]
D. Constraint coverage — what fraction of query constraints are covered

The final uncertainty score is a weighted combination of A+B+C+D.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config.settings import Config, DEFAULT_CONFIG
from src.retrieval.state_builder import EpisodeState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stopping signal dataclass
# ---------------------------------------------------------------------------

@dataclass
class StoppingSignal:
    should_stop: bool
    uncertainty: float
    confidence: float
    evidence_saturation: float
    constraint_coverage: float
    combined_score: float   # combined uncertainty ∈ [0,1]; lower → stop sooner
    reason: str             # human-readable reason for stop decision


# ---------------------------------------------------------------------------
# Individual estimators
# ---------------------------------------------------------------------------

def estimate_mc_dropout_uncertainty(
    policy_network: nn.Module,
    state_vec: torch.Tensor,
    action_embs: torch.Tensor,
    query_emb: torch.Tensor,
    n_samples: int = 10,
    device: torch.device = torch.device("cpu"),
) -> float:
    """
    Monte Carlo Dropout uncertainty:
    Run N forward passes with dropout active, compute variance of action probs.

    Returns
    -------
    uncertainty ∈ [0, 1]; higher = more uncertain
    """
    policy_network.train()   # activate dropout
    all_probs: List[torch.Tensor] = []

    with torch.no_grad():
        for _ in range(n_samples):
            _, probs, _, _ = policy_network(
                state_vec.to(device),
                action_embs.to(device),
                query_emb.to(device),
            )
            all_probs.append(probs)

    policy_network.eval()

    if not all_probs:
        return 1.0

    stacked = torch.stack(all_probs, dim=0)       # (N, num_actions)
    variance = stacked.var(dim=0).mean().item()    # mean variance over actions
    # Normalise: max theoretical variance for K actions ≈ 0.25
    uncertainty = min(variance / 0.25, 1.0)
    return float(uncertainty)


def estimate_evidence_spread(
    node_reprs: List[torch.Tensor],
    device: torch.device,
) -> float:
    """
    Variance-based evidence uncertainty:
    High spread → diverse evidence (uncertain) → continue.
    Low spread → evidence is clustering → stop.

    Returns
    -------
    spread ∈ [0, 1]; high = uncertain (diverse); low = confident (converged)
    """
    if len(node_reprs) < 2:
        return 1.0
    stack = torch.stack(node_reprs, dim=0).to(device)
    centroid = stack.mean(dim=0)
    distances = F.cosine_similarity(
        stack, centroid.unsqueeze(0).expand_as(stack), dim=-1
    )
    spread = 1.0 - distances.mean().item()   # low mean cosine → high spread
    return float(max(0.0, min(1.0, spread)))


def estimate_constraint_coverage(
    episode: EpisodeState,
) -> float:
    """
    Fraction of query constraints (aspects) that appear in collected evidence.

    Returns
    -------
    coverage ∈ [0, 1]; 1.0 = all constraints covered
    """
    query_aspects = set(episode.query_repr.detected_aspects)
    if not query_aspects:
        return 0.5   # neutral if no aspects to check

    evidence_text = " ".join(episode.collected_evidence).lower()
    covered = sum(1 for a in query_aspects if a in evidence_text)
    return float(covered / len(query_aspects))


def estimate_evidence_saturation(
    new_node_emb: torch.Tensor,
    existing_embs: List[torch.Tensor],
    device: torch.device,
) -> float:
    """
    Marginal novelty of a new node relative to existing evidence.
    Low novelty (high saturation) → evidence not growing → stop sooner.

    Returns
    -------
    novelty ∈ [0, 1]; 0 = fully saturated, 1 = completely novel
    """
    if not existing_embs:
        return 1.0
    stack = torch.stack(existing_embs, dim=0).to(device)
    centroid = stack.mean(dim=0)
    cos = F.cosine_similarity(
        new_node_emb.to(device).unsqueeze(0),
        centroid.unsqueeze(0),
    ).item()
    novelty = 1.0 - (cos + 1.0) / 2.0
    return float(max(0.0, min(1.0, novelty)))


# ---------------------------------------------------------------------------
# Main Stopping Module
# ---------------------------------------------------------------------------

class AdaptiveStoppingModule:
    """
    Decides whether to continue or stop retrieval at each step.

    Called by KGEnvironment.step() after each hop to update the uncertainty
    score and check stopping conditions.

    Parameters
    ----------
    config : Config
    policy_network : nn.Module | None
        If provided, MC Dropout uncertainty is used.
        If None, falls back to evidence-based estimates only.
    """

    def __init__(
        self,
        config: Config = DEFAULT_CONFIG,
        policy_network: Optional[nn.Module] = None,
    ) -> None:
        self.config = config
        self.policy_network = policy_network
        self.device = torch.device(config.encoder.device)
        sc = config.stopping
        self.uncertainty_threshold = sc.uncertainty_threshold
        self.confidence_threshold = sc.confidence_threshold
        self.min_hops = sc.min_hops
        self.saturation_threshold = sc.evidence_saturation_threshold
        self.mc_samples = sc.mc_dropout_samples
        logger.debug("AdaptiveStoppingModule initialised.")

    def evaluate(
        self,
        episode: EpisodeState,
        new_node_emb: Optional[torch.Tensor] = None,
        policy_value: float = 0.5,
        state_vec: Optional[torch.Tensor] = None,
        action_embs: Optional[torch.Tensor] = None,
        query_emb: Optional[torch.Tensor] = None,
    ) -> StoppingSignal:
        """
        Evaluate whether to stop retrieval.

        Parameters
        ----------
        episode       : current episode state
        new_node_emb  : embedding of the most recently retrieved node
        policy_value  : V(s) from critic head (maps to confidence)
        state_vec     : (state_dim,) — needed for MC Dropout
        action_embs   : (N, action_dim) — needed for MC Dropout
        query_emb     : (query_dim,) — needed for MC Dropout

        Returns
        -------
        StoppingSignal
        """
        hop = episode.hop_count
        existing_embs = episode.collected_node_reprs

        # --- Minimum hop guard ---
        if hop < self.min_hops:
            u = self._build_signal(
                uncertainty=1.0, confidence=0.0,
                saturation=1.0, coverage=0.0,
                should_stop=False, reason="min_hops_not_reached",
            )
            episode.uncertainty_score = u.combined_score
            return u

        # --- A: MC Dropout uncertainty ---
        mc_uncertainty = 0.5   # default if not available
        if (
            self.policy_network is not None
            and state_vec is not None
            and action_embs is not None
            and query_emb is not None
        ):
            mc_uncertainty = estimate_mc_dropout_uncertainty(
                policy_network=self.policy_network,
                state_vec=state_vec,
                action_embs=action_embs,
                query_emb=query_emb,
                n_samples=self.mc_samples,
                device=self.device,
            )

        # --- B: Evidence spread ---
        evidence_spread = estimate_evidence_spread(existing_embs, self.device)

        # --- C: Policy confidence (from critic value, normalised) ---
        confidence = float(torch.sigmoid(torch.tensor(policy_value)).item())

        # --- D: Constraint coverage ---
        coverage = estimate_constraint_coverage(episode)

        # --- E: Evidence saturation ---
        saturation_novelty = 1.0
        if new_node_emb is not None and existing_embs:
            saturation_novelty = estimate_evidence_saturation(
                new_node_emb, existing_embs, self.device
            )

        # --- Combined uncertainty score ---
        # Weighted average: lower combined → stop
        combined = (
            0.35 * mc_uncertainty
            + 0.25 * evidence_spread
            + 0.20 * (1.0 - confidence)
            + 0.10 * (1.0 - coverage)
            + 0.10 * saturation_novelty
        )
        combined = float(max(0.0, min(1.0, combined)))

        # --- Stopping decision ---
        should_stop = (
            combined < self.uncertainty_threshold
            and confidence > self.confidence_threshold
            and saturation_novelty < self.saturation_threshold
        )

        if should_stop:
            reason = (
                f"uncertainty={combined:.3f}<{self.uncertainty_threshold} "
                f"confidence={confidence:.3f}>{self.confidence_threshold} "
                f"saturation_novelty={saturation_novelty:.3f}"
            )
        else:
            reason = (
                f"continue: U={combined:.3f} conf={confidence:.3f} "
                f"coverage={coverage:.2f}"
            )

        signal = self._build_signal(
            uncertainty=combined,
            confidence=confidence,
            saturation=saturation_novelty,
            coverage=coverage,
            should_stop=should_stop,
            reason=reason,
        )
        episode.uncertainty_score = combined
        episode.done = should_stop

        logger.debug(
            "Stopping @ hop=%d: stop=%s U=%.3f conf=%.3f cov=%.2f sat=%.3f",
            hop, should_stop, combined, confidence, coverage, saturation_novelty,
        )
        return signal

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_signal(
        self,
        uncertainty: float,
        confidence: float,
        saturation: float,
        coverage: float,
        should_stop: bool,
        reason: str,
    ) -> StoppingSignal:
        return StoppingSignal(
            should_stop=should_stop,
            uncertainty=uncertainty,
            confidence=confidence,
            evidence_saturation=saturation,
            constraint_coverage=coverage,
            combined_score=uncertainty,
            reason=reason,
        )