"""
src/data/qa_dataset.py

Loader for the synthetic QA dataset produced by scripts/build_qa_dataset.py.

Each JSONL line is parsed into a typed QAItem.  Used by:
  * imitation warm-up (teacher trajectories)
  * PPO rollouts (reward needs gold answers)
  * evaluation (Hit@K / MRR against gold answers)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class QAItem:
    qid: str
    query: str
    type: str
    gold_answers: List[str]
    constraints: Dict[str, Any] = field(default_factory=dict)
    reasoning_path: List[str] = field(default_factory=list)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "QAItem":
        return QAItem(
            qid=str(d.get("qid", "")),
            query=str(d["query"]),
            type=str(d.get("type", "unknown")),
            gold_answers=list(d.get("gold_answers", []) or []),
            constraints=dict(d.get("constraints", {}) or {}),
            reasoning_path=list(d.get("reasoning_path", []) or []),
        )


def load_qa_dataset(
    path: Union[str, Path],
    require_gold: bool = True,
) -> List[QAItem]:
    """
    Load a QA JSONL file into a list of QAItem.

    Parameters
    ----------
    path : path to qa_train.jsonl / qa_test.jsonl
    require_gold : skip (with a warning) any sample whose gold_answers is empty

    Raises
    ------
    FileNotFoundError if the path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"QA dataset not found: {path}. "
            f"Generate it first: python scripts/build_qa_dataset.py"
        )

    items: List[QAItem] = []
    skipped = 0
    with path.open("r", encoding="utf-8") as fh:
        for ln, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = QAItem.from_dict(json.loads(line))
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning("Skipping malformed line %d in %s: %s", ln, path, exc)
                skipped += 1
                continue
            if require_gold and not item.gold_answers:
                skipped += 1
                continue
            items.append(item)

    if not items:
        raise ValueError(f"No valid QA samples loaded from {path}.")

    from collections import Counter
    dist = Counter(it.type for it in items)
    logger.info(
        "Loaded %d QA samples from %s (skipped=%d). Distribution: %s",
        len(items), path, skipped, dict(dist),
    )
    return items


def build_gold_map(items: List[QAItem]) -> Dict[str, List[str]]:
    """Map query text → gold answers (used to thread gold into PPO rollouts)."""
    return {it.query: it.gold_answers for it in items}