"""
agent_service.py — the ONLY integration point with the research project.

It lazily builds the existing RL ReasoningAgent and calls:

    answer = agent.answer(query, deterministic=True)

mapping AgentAnswer -> {answer, hops, confidence, reasoning_path, evidence}.

If AGENT_ENABLED is false (or the import/build fails), it returns a clearly
labelled MOCK answer so the web demo is fully usable without Neo4j / the model.
"""

import logging
import os
import sys
import threading
from typing import Any, Dict, List

from .config import get_settings

logger = logging.getLogger("webapp.agent")

_agent = None
_agent_lock = threading.Lock()
_agent_failed = False


def _build_agent():
    """Build the ReasoningAgent from the research project (heavy; called once)."""
    settings = get_settings()
    root = settings.research_project_path.strip()
    if not root:
        raise RuntimeError("RESEARCH_PROJECT_PATH not set")

    # Propagate connection settings to os.environ so the research project's
    # os.getenv(...) calls (Neo4j, Anthropic) see them — one .env controls all.
    _env_map = {
        "NEO4J_URI": settings.neo4j_uri,
        "NEO4J_USER": settings.neo4j_user,
        "NEO4J_PASSWORD": settings.neo4j_password,
        "NEO4J_DATABASE": settings.neo4j_database,
        "ANTHROPIC_API_KEY": settings.anthropic_api_key,
    }
    for k, v in _env_map.items():
        if v:
            os.environ[k] = v

    if root not in sys.path:
        sys.path.insert(0, root)

    # Imports from the existing project
    from src.config.settings import DEFAULT_CONFIG          # noqa: E402
    from src.rl.checkpoint import CheckpointManager          # noqa: E402
    import torch                                             # noqa: E402

    # build_system lives in the training entrypoint
    import importlib.util
    entry = os.path.join(root, "training_rl_agent.py")
    if not os.path.exists(entry):
        entry = os.path.join(root, "train_rl_agent.py")
    spec = importlib.util.spec_from_file_location("rl_entry", entry)
    rl_entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rl_entry)

    components = rl_entry.build_system(DEFAULT_CONFIG)

    # Load trained weights into the policy the agent's retriever uses.
    ckpt: CheckpointManager = components["ckpt"]
    tag = (
        "best" if ckpt.exists("best")
        else "imitation_final" if ckpt.exists("imitation_final")
        else None
    )
    if tag:
        ckpt.load(
            model=components["actor"].policy,
            optimiser=None,
            tag=tag,
            device=torch.device(DEFAULT_CONFIG.encoder.device),
        )
        logger.info("Agent loaded checkpoint tag=%s", tag)
    else:
        logger.warning("No trained checkpoint found; agent uses untrained policy.")

    return components["agent"]


def _get_agent():
    global _agent, _agent_failed
    if _agent is not None or _agent_failed:
        return _agent
    with _agent_lock:
        if _agent is None and not _agent_failed:
            try:
                _agent = _build_agent()
                logger.info("ReasoningAgent ready.")
            except Exception as exc:
                _agent_failed = True
                logger.error("Agent build failed (%s) — using MOCK answers.", exc)
    return _agent


def _mock(query: str) -> Dict[str, Any]:
    return {
        "answer": (
            f"[DEMO MOCK] You asked: \"{query}\". The RL agent is not connected "
            f"(set AGENT_ENABLED=true and RESEARCH_PROJECT_PATH, and start Neo4j). "
            f"This placeholder lets you exercise the full chat UI and storage."
        ),
        "hops": 0,
        "confidence": 0.0,
        "reasoning_path": ["(mock) no reasoning path — agent disabled"],
        "evidence": [],
    }


def agent_status() -> Dict[str, Any]:
    """Lightweight status for diagnostics (does not trigger a build)."""
    settings = get_settings()
    return {
        "agent_enabled": settings.agent_enabled,
        "loaded": _agent is not None,
        "build_failed": _agent_failed,
        "research_project_path": settings.research_project_path or None,
        "mode": (
            "real" if (_agent is not None)
            else "mock(disabled)" if not settings.agent_enabled
            else "mock(build_failed)" if _agent_failed
            else "mock(not_built_yet)"
        ),
    }


def run_inference(query: str) -> Dict[str, Any]:
    """
    Blocking call — run from a threadpool in the async route.
    Always returns a dict with answer/hops/confidence/reasoning_path/evidence.
    """
    settings = get_settings()
    if not settings.agent_enabled:
        return _mock(query)

    agent = _get_agent()
    if agent is None:
        return _mock(query)

    try:
        ans = agent.answer(query, deterministic=True)
        path = ans.reasoning_path_text or ""
        reasoning_steps: List[str] = [
            ln.strip() for ln in path.splitlines() if ln.strip()
        ]
        confidence = max(0.0, min(1.0, 1.0 - float(ans.uncertainty)))
        return {
            "answer": ans.final_answer,
            "hops": int(ans.n_hops),
            "confidence": round(confidence, 4),
            "reasoning_path": reasoning_steps,
            "evidence": list(ans.evidence_chain),
        }
    except Exception as exc:
        logger.exception("Inference failed: %s", exc)
        return {
            "answer": f"The agent failed to answer this query ({exc}).",
            "hops": 0,
            "confidence": 0.0,
            "reasoning_path": [],
            "evidence": [],
        }