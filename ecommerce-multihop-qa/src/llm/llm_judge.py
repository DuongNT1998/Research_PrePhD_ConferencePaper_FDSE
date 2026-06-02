"""
src/llm/llm_judge.py

LLM Judge — Weak Supervision & Answer Synthesis.

The LLM plays TWO roles in this system (NOT the core retrieval):
-----------------------------------------------------------------
1. Reward shaper (weak supervision):
   Given (query, evidence), score answer quality → float [0,1].
   Used in RewardFunction to shape RL training signal.

2. Answer synthesiser:
   Given (query, evidence, reasoning_path), generate final natural
   language answer for the user.

Design constraints
------------------
- LLM is called ONLY at terminal steps during training (not every hop).
- LLM is never part of the retrieval loop.
- Uses Anthropic Claude API (key injected by proxy).
- Scores are cached to avoid redundant API calls.
- Gracefully degrades if API unavailable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Dict, List, Optional

import requests

from src.config.settings import Config, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for e-commerce product QA systems.
Your task is to evaluate how well the provided evidence answers the user query.

Respond ONLY with a JSON object in this exact format:
{"score": <float 0.0-1.0>, "reason": "<brief explanation>"}

Scoring guide:
- 1.0: Evidence directly and completely answers all aspects of the query.
- 0.7-0.9: Evidence mostly answers the query with minor gaps.
- 0.4-0.6: Evidence partially answers the query.
- 0.1-0.3: Evidence is tangentially related but doesn't answer the query.
- 0.0: Evidence is completely irrelevant."""

_SYNTHESIS_SYSTEM_PROMPT = """You are a helpful e-commerce product advisor.
Based on the retrieved evidence from a knowledge graph, provide a clear,
accurate, and concise answer to the user's question.

Rules:
- Base your answer ONLY on the provided evidence.
- If the evidence is insufficient, say so explicitly.
- Do not hallucinate product specifications not in the evidence.
- Be specific: mention product names, prices, ratings when available.
- Keep the answer under 200 words."""


# ---------------------------------------------------------------------------
# LLM client (Anthropic Claude)
# ---------------------------------------------------------------------------

class LLMClient:
    """Thin wrapper around the Anthropic messages API."""

    def __init__(self, config: Config = DEFAULT_CONFIG) -> None:
        self.config = config
        self.api_base = config.llm.api_base
        self.model = config.llm.model_name
        self.max_tokens = config.llm.max_tokens
        self.temperature = config.llm.temperature

    def call(
        self,
        system: str,
        user: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Call Claude API and return the text response.

        Returns
        -------
        str — model response text, or empty string on failure
        """
        payload = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "messages": [{"role": "user", "content": user}],
            "system": system,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        else:
            payload["temperature"] = self.temperature

        try:
            resp = requests.post(
                self.api_base,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data.get("content", [])
            texts = [block["text"] for block in content if block.get("type") == "text"]
            return " ".join(texts).strip()
        except Exception as exc:
            logger.warning("LLM API call failed: %s", exc)
            return ""


# ---------------------------------------------------------------------------
# LLM Judge
# ---------------------------------------------------------------------------

class LLMJudge:
    """
    LLM-based quality judge for reward shaping.

    Scores (query, evidence) pairs on [0, 1] using Claude as a semantic
    evaluator.  Results are cached by (query_hash, evidence_hash).

    Parameters
    ----------
    config : Config
    """

    def __init__(self, config: Config = DEFAULT_CONFIG) -> None:
        self.client = LLMClient(config)
        self._cache: Dict[str, float] = {}
        logger.info("LLMJudge initialised (model=%s).", config.llm.model_name)

    def __call__(
        self,
        query: str,
        evidence: List[str],
    ) -> float:
        """
        Score answer quality.

        Parameters
        ----------
        query    : user query string
        evidence : list of evidence strings from KG traversal

        Returns
        -------
        score ∈ [0, 1]
        """
        if not evidence:
            return 0.0

        cache_key = self._cache_key(query, evidence)
        if cache_key in self._cache:
            return self._cache[cache_key]

        evidence_text = "\n".join(f"- {e}" for e in evidence[:10])
        user_prompt = (
            f"User query: {query}\n\n"
            f"Retrieved evidence:\n{evidence_text}"
        )

        raw = self.client.call(
            system=_JUDGE_SYSTEM_PROMPT,
            user=user_prompt,
            max_tokens=128,
        )

        score = self._parse_score(raw)
        self._cache[cache_key] = score
        logger.debug("LLMJudge score=%.3f for query='%s...'", score, query[:40])
        return score

    def _parse_score(self, raw: str) -> float:
        """Parse JSON score response from LLM."""
        if not raw:
            return 0.3   # conservative fallback
        try:
            # Strip markdown fences if present
            cleaned = raw.strip().strip("```json").strip("```").strip()
            data = json.loads(cleaned)
            return float(max(0.0, min(1.0, data.get("score", 0.3))))
        except Exception:
            # Try regex fallback
            import re
            match = re.search(r'"score"\s*:\s*([0-9.]+)', raw)
            if match:
                return float(max(0.0, min(1.0, float(match.group(1)))))
            return 0.3

    def _cache_key(self, query: str, evidence: List[str]) -> str:
        content = query + "||" + "||".join(evidence[:10])
        return hashlib.md5(content.encode()).hexdigest()


# ---------------------------------------------------------------------------
# LLM Synthesiser
# ---------------------------------------------------------------------------

class LLMSynthesiser:
    """
    LLM-based answer synthesiser.

    Given retrieved evidence and reasoning path, generates a final
    natural language answer for the user.

    Parameters
    ----------
    config : Config
    """

    def __init__(self, config: Config = DEFAULT_CONFIG) -> None:
        self.client = LLMClient(config)
        logger.info("LLMSynthesiser initialised.")

    def __call__(
        self,
        query: str,
        evidence: List[str],
        reasoning_path: str = "",
    ) -> str:
        """
        Synthesise a final answer from query + evidence.

        Parameters
        ----------
        query          : user query
        evidence       : list of evidence strings
        reasoning_path : optional explainable path text

        Returns
        -------
        str — final answer
        """
        if not evidence:
            return f"I could not find sufficient evidence to answer: {query}"

        evidence_text = "\n".join(f"{i+1}. {e}" for i, e in enumerate(evidence))

        user_prompt = (
            f"User question: {query}\n\n"
            f"Evidence retrieved from knowledge graph:\n{evidence_text}"
        )
        if reasoning_path:
            user_prompt += f"\n\nRetrieval path summary:\n{reasoning_path[:500]}"

        answer = self.client.call(
            system=_SYNTHESIS_SYSTEM_PROMPT,
            user=user_prompt,
        )

        if not answer:
            # Fallback: structured summary from evidence
            lines = [f"Based on product knowledge graph data for '{query}':"]
            for ev in evidence[:5]:
                lines.append(f"  • {ev}")
            return "\n".join(lines)

        return answer