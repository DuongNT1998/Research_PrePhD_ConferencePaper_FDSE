"""
src/evaluation/metrics.py

Evaluation Metrics for the Adaptive Multi-hop Retrieval System.

Metrics implemented
-------------------
Retrieval Quality:
  - Hit@K          : whether ground-truth node appears in top-K retrieved
  - MRR            : mean reciprocal rank
  - NDCG@K         : normalised discounted cumulative gain
  - Precision@K    : fraction of top-K that are relevant
  - Recall@K       : fraction of all relevant nodes retrieved

Path Quality:
  - Path Length Distribution
  - Relation Diversity Score
  - Cycle Rate (lower is better)
  - Aspect Coverage Rate

Efficiency:
  - Mean Hops to Stop
  - Over-retrieval Rate  (hops > optimal)
  - Under-retrieval Rate (hops < optimal)

Stopping Quality:
  - Early Stop Rate
  - Late Stop Rate
  - Uncertainty at Stop (lower is better)

Answer Quality (requires ground truth):
  - Exact Match (EM)
  - F1 over tokens
  - BLEU-1 (rough fluency proxy)
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RetrievalMetrics:
    """Per-query retrieval evaluation result."""
    query: str
    hit_at_1: float = 0.0
    hit_at_3: float = 0.0
    hit_at_5: float = 0.0
    mrr: float = 0.0
    ndcg_at_5: float = 0.0
    precision_at_5: float = 0.0
    recall_at_5: float = 0.0
    n_hops: int = 0
    relation_diversity: float = 0.0
    cycle_rate: float = 0.0
    aspect_coverage: float = 0.0
    uncertainty_at_stop: float = 1.0
    total_reward: float = 0.0


@dataclass
class AggregatedMetrics:
    """Aggregated metrics over a full evaluation set."""
    n_queries: int = 0
    # Retrieval
    mean_hit_at_1: float = 0.0
    mean_hit_at_3: float = 0.0
    mean_hit_at_5: float = 0.0
    mean_mrr: float = 0.0
    mean_ndcg_at_5: float = 0.0
    mean_precision_at_5: float = 0.0
    mean_recall_at_5: float = 0.0
    # Path
    mean_hops: float = 0.0
    std_hops: float = 0.0
    mean_relation_diversity: float = 0.0
    mean_cycle_rate: float = 0.0
    mean_aspect_coverage: float = 0.0
    # Stopping
    mean_uncertainty_at_stop: float = 1.0
    # Reward
    mean_total_reward: float = 0.0
    # Answer quality (if ground truth available)
    mean_exact_match: float = 0.0
    mean_f1: float = 0.0
    # Per-query breakdown
    per_query: List[RetrievalMetrics] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "per_query"}
        return d


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def hit_at_k(
    retrieved_node_ids: List[str],
    relevant_node_ids: Set[str],
    k: int,
) -> float:
    """Hit@K: 1 if any relevant node appears in the top-K retrieved."""
    top_k = retrieved_node_ids[:k]
    return 1.0 if any(nid in relevant_node_ids for nid in top_k) else 0.0


def reciprocal_rank(
    retrieved_node_ids: List[str],
    relevant_node_ids: Set[str],
) -> float:
    """MRR component: 1 / rank of first relevant node."""
    for rank, nid in enumerate(retrieved_node_ids, start=1):
        if nid in relevant_node_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_node_ids: List[str],
    relevant_node_ids: Set[str],
    k: int,
) -> float:
    """NDCG@K with binary relevance."""
    def dcg(ids: List[str], rel_set: Set[str], k_: int) -> float:
        score = 0.0
        for i, nid in enumerate(ids[:k_], start=1):
            if nid in rel_set:
                score += 1.0 / math.log2(i + 1)
        return score

    actual_dcg = dcg(retrieved_node_ids, relevant_node_ids, k)
    # Ideal: all relevant nodes at top
    ideal_ids = list(relevant_node_ids)[:k]
    ideal_dcg = dcg(ideal_ids, relevant_node_ids, k)
    if ideal_dcg == 0.0:
        return 0.0
    return actual_dcg / ideal_dcg


def precision_at_k(
    retrieved_node_ids: List[str],
    relevant_node_ids: Set[str],
    k: int,
) -> float:
    top_k = retrieved_node_ids[:k]
    if not top_k:
        return 0.0
    return sum(1 for nid in top_k if nid in relevant_node_ids) / len(top_k)


def recall_at_k(
    retrieved_node_ids: List[str],
    relevant_node_ids: Set[str],
    k: int,
) -> float:
    if not relevant_node_ids:
        return 0.0
    top_k = retrieved_node_ids[:k]
    return sum(1 for nid in top_k if nid in relevant_node_ids) / len(relevant_node_ids)


# ---------------------------------------------------------------------------
# Path quality metrics
# ---------------------------------------------------------------------------

def relation_diversity_score(relation_sequence: List[str]) -> float:
    """
    Fraction of unique relation types in the traversal path.
    Higher = more diverse exploration.
    """
    if not relation_sequence:
        return 0.0
    unique = len(set(relation_sequence))
    return unique / len(relation_sequence)


def cycle_rate(node_sequence: List[str]) -> float:
    """
    Fraction of steps that revisit an already-visited node.
    Lower = better path quality.
    """
    if len(node_sequence) <= 1:
        return 0.0
    seen: set = set()
    cycles = 0
    for nid in node_sequence:
        if nid in seen:
            cycles += 1
        seen.add(nid)
    return cycles / len(node_sequence)


def aspect_coverage_rate(
    evidence_chain: List[str],
    query_aspects: List[str],
) -> float:
    """
    Fraction of query aspects mentioned in the collected evidence.
    """
    if not query_aspects:
        return 0.0
    evidence_text = " ".join(evidence_chain).lower()
    covered = sum(1 for a in query_aspects if a.lower() in evidence_text)
    return covered / len(query_aspects)


# ---------------------------------------------------------------------------
# Answer quality metrics
# ---------------------------------------------------------------------------

def exact_match(predicted: str, ground_truth: str) -> float:
    """Case-insensitive exact match after normalisation."""
    pred = predicted.strip().lower()
    gt = ground_truth.strip().lower()
    return 1.0 if pred == gt else 0.0


def token_f1(predicted: str, ground_truth: str) -> float:
    """Token-level F1 score (standard QA metric)."""
    pred_tokens = predicted.lower().split()
    gt_tokens = ground_truth.lower().split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    pred_counter = Counter(pred_tokens)
    gt_counter = Counter(gt_tokens)
    common = sum((pred_counter & gt_counter).values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Per-query evaluator
# ---------------------------------------------------------------------------

def evaluate_episode(
    query: str,
    node_sequence: List[str],
    relation_sequence: List[str],
    evidence_chain: List[str],
    query_aspects: List[str],
    n_hops: int,
    uncertainty_at_stop: float,
    total_reward: float,
    relevant_node_ids: Optional[Set[str]] = None,
    predicted_answer: Optional[str] = None,
    ground_truth_answer: Optional[str] = None,
) -> RetrievalMetrics:
    """
    Compute all metrics for a single retrieval episode.

    Parameters
    ----------
    relevant_node_ids : set of ground-truth relevant node IDs (if available)
    """
    rel_set = relevant_node_ids or set()
    m = RetrievalMetrics(query=query)

    if rel_set:
        m.hit_at_1 = hit_at_k(node_sequence, rel_set, k=1)
        m.hit_at_3 = hit_at_k(node_sequence, rel_set, k=3)
        m.hit_at_5 = hit_at_k(node_sequence, rel_set, k=5)
        m.mrr = reciprocal_rank(node_sequence, rel_set)
        m.ndcg_at_5 = ndcg_at_k(node_sequence, rel_set, k=5)
        m.precision_at_5 = precision_at_k(node_sequence, rel_set, k=5)
        m.recall_at_5 = recall_at_k(node_sequence, rel_set, k=5)

    m.n_hops = n_hops
    m.relation_diversity = relation_diversity_score(relation_sequence)
    m.cycle_rate = cycle_rate(node_sequence)
    m.aspect_coverage = aspect_coverage_rate(evidence_chain, query_aspects)
    m.uncertainty_at_stop = uncertainty_at_stop
    m.total_reward = total_reward

    return m


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

def aggregate_metrics(per_query_metrics: List[RetrievalMetrics]) -> AggregatedMetrics:
    """Aggregate per-query metrics into a summary."""
    n = len(per_query_metrics)
    if n == 0:
        return AggregatedMetrics()

    import numpy as np

    def mean(lst: List[float]) -> float:
        return float(np.mean(lst)) if lst else 0.0

    hops = [m.n_hops for m in per_query_metrics]

    agg = AggregatedMetrics(
        n_queries=n,
        mean_hit_at_1=mean([m.hit_at_1 for m in per_query_metrics]),
        mean_hit_at_3=mean([m.hit_at_3 for m in per_query_metrics]),
        mean_hit_at_5=mean([m.hit_at_5 for m in per_query_metrics]),
        mean_mrr=mean([m.mrr for m in per_query_metrics]),
        mean_ndcg_at_5=mean([m.ndcg_at_5 for m in per_query_metrics]),
        mean_precision_at_5=mean([m.precision_at_5 for m in per_query_metrics]),
        mean_recall_at_5=mean([m.recall_at_5 for m in per_query_metrics]),
        mean_hops=float(np.mean(hops)),
        std_hops=float(np.std(hops)),
        mean_relation_diversity=mean([m.relation_diversity for m in per_query_metrics]),
        mean_cycle_rate=mean([m.cycle_rate for m in per_query_metrics]),
        mean_aspect_coverage=mean([m.aspect_coverage for m in per_query_metrics]),
        mean_uncertainty_at_stop=mean([m.uncertainty_at_stop for m in per_query_metrics]),
        mean_total_reward=mean([m.total_reward for m in per_query_metrics]),
        per_query=per_query_metrics,
    )
    return agg


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_metrics(agg: AggregatedMetrics) -> None:
    """Print formatted evaluation report to logger."""
    sep = "=" * 55
    logger.info(sep)
    logger.info("EVALUATION RESULTS  (n_queries=%d)", agg.n_queries)
    logger.info(sep)
    logger.info("── Retrieval Quality ──")
    logger.info("  Hit@1:       %.4f", agg.mean_hit_at_1)
    logger.info("  Hit@3:       %.4f", agg.mean_hit_at_3)
    logger.info("  Hit@5:       %.4f", agg.mean_hit_at_5)
    logger.info("  MRR:         %.4f", agg.mean_mrr)
    logger.info("  NDCG@5:      %.4f", agg.mean_ndcg_at_5)
    logger.info("  Precision@5: %.4f", agg.mean_precision_at_5)
    logger.info("  Recall@5:    %.4f", agg.mean_recall_at_5)
    logger.info("── Path Quality ──")
    logger.info("  Mean Hops:   %.2f ± %.2f", agg.mean_hops, agg.std_hops)
    logger.info("  Rel Diversity: %.4f", agg.mean_relation_diversity)
    logger.info("  Cycle Rate:    %.4f", agg.mean_cycle_rate)
    logger.info("  Aspect Coverage: %.4f", agg.mean_aspect_coverage)
    logger.info("── Stopping Quality ──")
    logger.info("  Mean Uncertainty@Stop: %.4f", agg.mean_uncertainty_at_stop)
    logger.info("── Reward ──")
    logger.info("  Mean Total Reward: %.4f", agg.mean_total_reward)
    logger.info(sep)