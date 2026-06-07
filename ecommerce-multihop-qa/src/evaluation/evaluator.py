"""
src/evaluation/evaluator.py

End-to-end Evaluator for the Adaptive Retrieval System.

Runs a trained ReasoningAgent or RLRetriever over a held-out query set,
computes all metrics, and saves JSON + log results.

Usage
-----
    evaluator = Evaluator(agent, config)
    results = evaluator.evaluate(query_list, ground_truth=gt_dict)
    evaluator.save_results(results, Path("outputs/results/eval_run_01.json"))
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.config.settings import Config, DEFAULT_CONFIG
from src.retrieval.rl_retriever import RLRetriever, EpisodeRecord
from src.reasoning.agent import ReasoningAgent, AgentAnswer
from src.evaluation.metrics import (
    RetrievalMetrics,
    AggregatedMetrics,
    evaluate_episode,
    aggregate_metrics,
    print_metrics,
)

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluation harness for the adaptive retrieval QA system.

    Parameters
    ----------
    retriever : RLRetriever  (low-level, returns EpisodeRecord)
    agent     : ReasoningAgent | None  (high-level, produces final answers)
    config    : Config
    """

    def __init__(
        self,
        retriever: RLRetriever,
        agent: Optional[ReasoningAgent] = None,
        config: Config = DEFAULT_CONFIG,
    ) -> None:
        self.retriever = retriever
        self.agent = agent
        self.config = config
        logger.info("Evaluator initialised.")

    # ------------------------------------------------------------------
    # Main evaluation entry point
    # ------------------------------------------------------------------

    def evaluate(
        self,
        queries: List[str],
        ground_truth: Optional[Dict[str, Any]] = None,
        deterministic: bool = True,
        run_answer_synthesis: bool = False,
    ) -> AggregatedMetrics:
        """
        Evaluate the system over a set of queries.

        Parameters
        ----------
        queries              : list of English query strings
        ground_truth         : optional dict mapping query → {
                                   "relevant_node_ids": list[str],
                                   "answer": str,
                                   "aspects": list[str],
                               }
        deterministic        : use greedy policy (recommended for eval)
        run_answer_synthesis : if True and agent is set, also run LLM synthesis

        Returns
        -------
        AggregatedMetrics
        """
        logger.info("Starting evaluation over %d queries.", len(queries))
        per_query_metrics: List[RetrievalMetrics] = []
        all_records: List[Dict[str, Any]] = []

        t_eval_start = time.time()

        for i, query in enumerate(queries):
            logger.info("  [%d/%d] %s", i + 1, len(queries), query[:70])

            gt = ground_truth.get(query, {}) if ground_truth else {}
            relevant_nodes: Set[str] = set(gt.get("relevant_node_ids", []))
            gt_answer: Optional[str] = gt.get("answer")
            aspects: List[str] = gt.get("aspects", [])

            # --- Run retrieval ---
            try:
                record: EpisodeRecord = self.retriever.run_episode(
                    query=query,
                    deterministic=deterministic,
                    collect_transitions=False,
                )
            except Exception as exc:
                logger.warning("Retrieval failed for '%s': %s", query[:50], exc)
                continue

            # --- Optionally run answer synthesis ---
            predicted_answer: Optional[str] = None
            if run_answer_synthesis and self.agent is not None:
                try:
                    ans: AgentAnswer = self.agent.answer(query, deterministic=True)
                    predicted_answer = ans.final_answer
                except Exception as exc:
                    logger.warning("Synthesis failed: %s", exc)

            # --- Compute metrics ---
            qm = evaluate_episode(
                query=query,
                node_sequence=record.node_sequence,
                relation_sequence=record.relation_sequence,
                evidence_chain=record.evidence_chain,
                query_aspects=aspects,
                n_hops=record.n_hops,
                uncertainty_at_stop=record.final_uncertainty,
                total_reward=record.total_reward,
                relevant_node_ids=relevant_nodes if relevant_nodes else None,
                predicted_answer=predicted_answer,
                ground_truth_answer=gt_answer,
            )
            per_query_metrics.append(qm)

            # Serialisable record
            all_records.append({
                "query": query,
                "n_hops": record.n_hops,
                "stop_reason": record.stop_reason,
                "total_reward": record.total_reward,
                "uncertainty": record.final_uncertainty,
                "node_sequence": record.node_sequence[:10],
                "relation_sequence": record.relation_sequence[:10],
                "evidence_count": len(record.evidence_chain),
                "metrics": {
                    "hit@1": qm.hit_at_1,
                    "hit@3": qm.hit_at_3,
                    "hit@5": qm.hit_at_5,
                    "mrr": qm.mrr,
                    "ndcg@5": qm.ndcg_at_5,
                    "aspect_coverage": qm.aspect_coverage,
                    "cycle_rate": qm.cycle_rate,
                },
                "predicted_answer": predicted_answer,
            })

        t_elapsed = time.time() - t_eval_start
        logger.info("Evaluation done: %d queries in %.1fs", len(per_query_metrics), t_elapsed)

        agg = aggregate_metrics(per_query_metrics)
        print_metrics(agg)
        return agg

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------

    def save_results(
        self,
        agg: AggregatedMetrics,
        output_path: Path,
    ) -> None:
        """Save aggregated metrics as JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = agg.to_dict()
        data["per_query"] = [
            {
                "query": m.query,
                "hit_at_1": m.hit_at_1,
                "hit_at_3": m.hit_at_3,
                "mrr": m.mrr,
                "n_hops": m.n_hops,
                "aspect_coverage": m.aspect_coverage,
                "uncertainty_at_stop": m.uncertainty_at_stop,
                "total_reward": m.total_reward,
            }
            for m in agg.per_query
        ]
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Results saved to %s", output_path)