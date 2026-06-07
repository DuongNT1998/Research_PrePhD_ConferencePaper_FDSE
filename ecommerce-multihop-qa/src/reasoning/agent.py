"""
src/reasoning/agent.py

ReasoningAgent — top-level QA agent for the full pipeline.

Combines:
  1. AdaptiveRetriever  → multi-hop KG traversal
  2. EvidenceAggregator → structured evidence chain
  3. ExplainablePathBuilder → human-readable reasoning path
  4. LLM answer synthesis  → final natural language answer

This is what the end-user / evaluation harness calls.

Usage
-----
    agent = ReasoningAgent(retriever, llm_synthesiser, config)
    answer = agent.answer("laptop under 500 with long battery, not too heavy")
    print(answer.final_answer)
    print(answer.reasoning_path_text)
    print(answer.evidence_chain)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.config.settings import Config, DEFAULT_CONFIG
from src.retrieval.adaptive_retriever import AdaptiveRetriever, RetrievalResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class AgentAnswer:
    """Complete output of the ReasoningAgent."""
    query: str
    final_answer: str
    # Human-readable reasoning path
    reasoning_path_text: str
    # Raw evidence strings
    evidence_chain: List[str] = field(default_factory=list)
    # Structured trajectory
    trajectory: List[Dict[str, Any]] = field(default_factory=list)
    # Number of hops taken
    n_hops: int = 0
    # Uncertainty at stopping
    uncertainty: float = 1.0
    # Stop reason
    stop_reason: str = ""
    # LLM raw response
    llm_raw: str = ""


# ---------------------------------------------------------------------------
# Evidence Aggregator
# ---------------------------------------------------------------------------

class EvidenceAggregator:
    """
    Aggregates and deduplicates evidence strings from the retrieval trajectory.
    Builds a structured evidence set for LLM consumption.
    """

    def __init__(self, max_items: int = 10) -> None:
        self.max_items = max_items

    def aggregate(self, evidence_chain: List[str]) -> List[str]:
        """
        Deduplicate and rank evidence by informativeness.
        Returns up to max_items evidence strings.
        """
        seen: set = set()
        unique: List[str] = []
        for ev in evidence_chain:
            ev = ev.strip()
            if ev and ev not in seen:
                seen.add(ev)
                unique.append(ev)
        # Simple heuristic: longer evidence = more informative
        unique.sort(key=len, reverse=True)
        return unique[: self.max_items]


# ---------------------------------------------------------------------------
# Explainable Path Builder
# ---------------------------------------------------------------------------

class ExplainablePathBuilder:
    """
    Converts a structured trajectory into a human-readable reasoning path string.

    Example output:
    ---------------
    Step 1 [Product → HAS_POSITIVE_ASPECT → Aspect]
      B07XYZ (Product: Sony WH-1000XM5 | Price: $279.99 | Rating: 4.7)
      → HAS_POSITIVE_ASPECT (weight=42) →
      battery (Aspect: battery, positive sentiment)

    Step 2 [Product → HAS_FEATURE → Feature]
      B07XYZ → HAS_FEATURE →
      "30-hour battery life with ANC" (Feature)
    """

    def build(self, trajectory: List[Dict[str, Any]], query: str) -> str:
        if not trajectory:
            return "No traversal steps recorded."

        lines: List[str] = [
            f"Reasoning Path for query: \"{query}\"",
            "=" * 60,
        ]
        for step in trajectory:
            hop = step.get("hop", "?")
            from_node = step.get("from_node", step.get("node_id", "?"))
            to_node = step.get("node_id", "?")
            node_type = step.get("node_type", "?")
            relation = step.get("relation", "?")
            edge_w = step.get("edge_weight", 1.0)
            evidence = step.get("evidence", "")
            uncertainty = step.get("uncertainty", 1.0)
            attn = step.get("attn_weight", 0.0)

            lines.append(
                f"\nStep {hop}: [{relation}] → {node_type}"
                f"  (attention={attn:.3f}, uncertainty={uncertainty:.3f})"
            )
            lines.append(f"  {from_node} ─[{relation} w={edge_w:.1f}]→ {to_node}")
            if evidence:
                lines.append(f"  Evidence: {evidence}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ReasoningAgent
# ---------------------------------------------------------------------------

class ReasoningAgent:
    """
    Full QA pipeline:  query → adaptive retrieval → answer synthesis.

    Parameters
    ----------
    retriever       : AdaptiveRetriever
    llm_synthesiser : callable  llm_synthesiser(query, evidence) → str
                      This is the ONLY place LLM is invoked (weak supervision
                      + final synthesis).  Not part of core retrieval.
    config          : Config
    """

    def __init__(
        self,
        retriever: AdaptiveRetriever,
        llm_synthesiser,
        config: Config = DEFAULT_CONFIG,
    ) -> None:
        self.retriever = retriever
        self.llm_synthesiser = llm_synthesiser
        self.config = config
        self.aggregator = EvidenceAggregator(
            max_items=config.llm.max_evidence_items
        )
        self.path_builder = ExplainablePathBuilder()
        logger.info("ReasoningAgent initialised.")

    def answer(
        self,
        query: str,
        anchor_node_id: Optional[str] = None,
        deterministic: bool = True,
    ) -> AgentAnswer:
        """
        Full pipeline: retrieve → aggregate → explain → synthesise answer.

        Parameters
        ----------
        query          : English user query
        anchor_node_id : optional KG anchor
        deterministic  : greedy policy (True for inference)

        Returns
        -------
        AgentAnswer
        """
        logger.info("ReasoningAgent.answer: '%s'", query[:80])

        # === Step 1: Adaptive multi-hop retrieval ===
        retrieval: RetrievalResult = self.retriever.retrieve(
            query=query,
            anchor_node_id=anchor_node_id,
            deterministic=deterministic,
        )

        # === Step 2: Aggregate evidence ===
        agg_evidence = self.aggregator.aggregate(retrieval.evidence_chain)

        # === Step 3: Build explainable path ===
        path_text = self.path_builder.build(
            trajectory=retrieval.reasoning_path,
            query=query,
        )

        # === Step 4: LLM answer synthesis (weak supervision role) ===
        try:
            final_answer = self.llm_synthesiser(
                query=query,
                evidence=agg_evidence,
                reasoning_path=path_text,
            )
            llm_raw = final_answer
        except Exception as exc:
            logger.warning("LLM synthesis failed: %s — using evidence fallback.", exc)
            final_answer = self._fallback_answer(query, agg_evidence)
            llm_raw = ""

        return AgentAnswer(
            query=query,
            final_answer=final_answer,
            reasoning_path_text=path_text,
            evidence_chain=agg_evidence,
            trajectory=retrieval.trajectory,
            n_hops=retrieval.n_hops,
            uncertainty=retrieval.final_uncertainty,
            stop_reason=retrieval.stop_reason,
            llm_raw=llm_raw,
        )

    def answer_batch(
        self,
        queries: List[str],
        deterministic: bool = True,
    ) -> List[AgentAnswer]:
        """Answer a list of queries."""
        return [self.answer(q, deterministic=deterministic) for q in queries]

    # ------------------------------------------------------------------
    # Fallback when LLM is unavailable
    # ------------------------------------------------------------------

    def _fallback_answer(self, query: str, evidence: List[str]) -> str:
        if not evidence:
            return f"No relevant evidence found for: {query}"
        lines = [f"Based on retrieved evidence for '{query}':"]
        for i, ev in enumerate(evidence[:5], 1):
            lines.append(f"  {i}. {ev}")
        return "\n".join(lines)