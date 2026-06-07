"""
src/retrieval/state_builder.py

State Construction module.

At every reasoning step the policy network receives a fixed-dimension state
vector that captures:

    s_t = f(query, current_node, visited_history, evidence, hop_count, uncertainty)

Components
----------
1. query_vec       — projected query embedding (query_dim)
2. node_vec        — current node embedding (node_dim)
3. type_vec        — current node type embedding (node_type_dim)
4. hop_vec         — sinusoidal hop-count encoding (hop_encoding_dim)
5. history_vec     — GRU over sequence of visited-node embeddings (history_dim)
6. evidence_vec    — GRU over sequence of collected evidence embeddings (evidence_dim)
7. uncertainty_vec — current uncertainty signal (uncertainty_dim)

Total: StateConfig.total_dim

The StateBuilder is stateful per episode: call reset() at the start of each
new query, then build() at every step.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.config.settings import Config, DEFAULT_CONFIG
from src.retrieval.query_encoder import QueryRepresentation
from src.retrieval.node_encoder import NodeRepresentation, EdgeRepresentation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RetrievalStep: records one step taken during an episode
# ---------------------------------------------------------------------------

@dataclass
class RetrievalStep:
    """One step in the multi-hop retrieval trajectory."""
    hop: int
    node_id: str
    node_type: str
    relation_type: str          # edge used to arrive here ("START" for hop-0)
    edge_weight: float
    node_embedding: torch.Tensor   # (node_feature_dim,)
    evidence_text: str             # textual evidence from this node


# ---------------------------------------------------------------------------
# EpisodeState: mutable state carried across one retrieval episode
# ---------------------------------------------------------------------------

@dataclass
class EpisodeState:
    """Complete mutable state of a retrieval episode at a given step."""
    # Fixed query context
    query_repr: QueryRepresentation
    # Current position on graph
    current_node: NodeRepresentation
    # Trajectory so far (not including current)
    visited_steps: List[RetrievalStep] = field(default_factory=list)
    # Set of visited node IDs (for deduplication)
    visited_node_ids: set = field(default_factory=set)
    # Accumulated evidence strings
    collected_evidence: List[str] = field(default_factory=list)
    # Gold answer parent_asins for this query (from the QA dataset). Empty when
    # running free inference; populated during training so the reward function
    # can score answer quality exactly instead of relying only on the LLM judge.
    gold_answers: List[str] = field(default_factory=list)
    # Collected node representations for evidence GRU (projected, node_feature_dim)
    collected_node_reprs: List[torch.Tensor] = field(default_factory=list)
    # Collected node BASE embeddings (embedding_dim) — same space as the query
    # base embedding; used only for grounding / relevance reward computation.
    collected_base_reprs: List[torch.Tensor] = field(default_factory=list)
    # Current hop count
    hop_count: int = 0
    # Current uncertainty score (updated by stopping module)
    uncertainty_score: float = 1.0
    # Whether retrieval has been stopped
    done: bool = False
    # Dense state vector (computed by StateBuilder.build())
    state_vector: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# Sinusoidal hop encoding
# ---------------------------------------------------------------------------

def sinusoidal_encoding(hop: int, dim: int, device: torch.device) -> torch.Tensor:
    """Return a sinusoidal positional encoding for the hop count."""
    encoding = torch.zeros(dim, device=device)
    for i in range(0, dim, 2):
        encoding[i] = math.sin(hop / (10000 ** (i / dim)))
        if i + 1 < dim:
            encoding[i + 1] = math.cos(hop / (10000 ** (i / dim)))
    return encoding


# ---------------------------------------------------------------------------
# History GRU: encodes the visited-node sequence
# ---------------------------------------------------------------------------

class HistoryGRU(nn.Module):
    """
    Single-layer GRU that reads the sequence of visited node embeddings
    and outputs a summary history vector.
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self._hidden: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self._hidden = None

    def step(self, node_emb: torch.Tensor) -> torch.Tensor:
        """
        Feed one new node embedding and return updated hidden state.

        Parameters
        ----------
        node_emb : (node_feature_dim,)

        Returns
        -------
        hidden : (hidden_dim,)
        """
        x = node_emb.unsqueeze(0).unsqueeze(0)  # (1,1,input_dim)
        if self._hidden is None:
            _, h = self.gru(x)
        else:
            _, h = self.gru(x, self._hidden)
        self._hidden = h
        return h.squeeze(0).squeeze(0)

    def get_hidden(self, device: torch.device, hidden_dim: int) -> torch.Tensor:
        if self._hidden is None:
            return torch.zeros(hidden_dim, device=device)
        return self._hidden.squeeze(0).squeeze(0)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Encode a full sequence at once.

        Parameters
        ----------
        sequence : (T, input_dim)

        Returns
        -------
        hidden : (hidden_dim,)
        """
        if sequence.shape[0] == 0:
            return torch.zeros(self.hidden_dim, device=sequence.device)
        x = sequence.unsqueeze(0)        # (1, T, input_dim)
        _, h = self.gru(x)
        return h.squeeze(0).squeeze(0)


# ---------------------------------------------------------------------------
# Uncertainty encoding MLP
# ---------------------------------------------------------------------------

class UncertaintyEncoder(nn.Module):
    """Maps scalar uncertainty score → uncertainty_dim vector."""

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, output_dim),
            nn.Tanh(),
        )

    def forward(self, uncertainty: float, device: torch.device) -> torch.Tensor:
        x = torch.tensor([[uncertainty]], dtype=torch.float32, device=device)
        return self.net(x).squeeze(0)


# ---------------------------------------------------------------------------
# Main StateBuilder
# ---------------------------------------------------------------------------

class StateBuilder(nn.Module):
    """
    Builds the dense state vector for the policy network at each hop.

    The builder maintains GRU hidden states across steps so the history
    and evidence are incrementally updated.

    Parameters
    ----------
    config : Config

    Usage
    -----
    >>> sb = StateBuilder(config)
    >>> sb.reset(query_repr, initial_node_repr)
    >>> state_vec = sb.build(current_node_repr, uncertainty=0.9)
    >>> state_vec.shape
    torch.Size([state_config.total_dim])
    """

    def __init__(self, config: Config = DEFAULT_CONFIG) -> None:
        super().__init__()
        self.config = config
        self.device = torch.device(config.encoder.device)
        sc = config.state

        self._history_gru = HistoryGRU(
            input_dim=config.encoder.node_feature_dim,
            hidden_dim=sc.history_dim,
        ).to(self.device)

        self._evidence_gru = HistoryGRU(
            input_dim=config.encoder.node_feature_dim,
            hidden_dim=sc.evidence_dim,
        ).to(self.device)

        self._uncertainty_enc = UncertaintyEncoder(
            output_dim=sc.uncertainty_dim,
        ).to(self.device)

        # Internal episode state
        self._episode: Optional[EpisodeState] = None
        logger.debug("StateBuilder initialised, total_state_dim=%d", sc.total_dim)

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset(
        self,
        query_repr: QueryRepresentation,
        initial_node: NodeRepresentation,
    ) -> EpisodeState:
        """
        Reset builder for a new query episode.

        Parameters
        ----------
        query_repr : QueryRepresentation
        initial_node : NodeRepresentation  — anchor node at hop 0

        Returns
        -------
        EpisodeState with initial state vector populated
        """
        self._history_gru.reset()
        self._evidence_gru.reset()

        # Seed history GRU with anchor node
        self._history_gru.step(initial_node.embedding.to(self.device))

        state = EpisodeState(
            query_repr=query_repr,
            current_node=initial_node,
            visited_node_ids={initial_node.node_id},
        )

        state.state_vector = self._compute_vector(
            query_repr=query_repr,
            node_repr=initial_node,
            hop=0,
            uncertainty=1.0,
        )
        self._episode = state
        return state

    def step(
        self,
        new_node: NodeRepresentation,
        edge: EdgeRepresentation,
        evidence_text: str,
        uncertainty: float,
    ) -> EpisodeState:
        """
        Advance the episode by one hop.

        Parameters
        ----------
        new_node : NodeRepresentation — node arrived at
        edge : EdgeRepresentation    — edge traversed
        evidence_text : str          — textual content to collect from new_node
        uncertainty : float          — updated uncertainty from StoppingModule

        Returns
        -------
        Updated EpisodeState
        """
        if self._episode is None:
            raise RuntimeError("StateBuilder.reset() must be called before step().")

        ep = self._episode
        ep.hop_count += 1

        # Record step
        step = RetrievalStep(
            hop=ep.hop_count,
            node_id=new_node.node_id,
            node_type=new_node.node_type,
            relation_type=edge.relation_type,
            edge_weight=edge.weight,
            node_embedding=new_node.embedding,
            evidence_text=evidence_text,
        )
        ep.visited_steps.append(step)
        ep.visited_node_ids.add(new_node.node_id)
        ep.collected_evidence.append(evidence_text)
        ep.collected_node_reprs.append(new_node.embedding.to(self.device))
        ep.collected_base_reprs.append(new_node.base_embedding.to(self.device))
        ep.current_node = new_node
        ep.uncertainty_score = uncertainty

        # Update GRUs
        self._history_gru.step(new_node.embedding.to(self.device))
        self._evidence_gru.step(new_node.embedding.to(self.device))

        ep.state_vector = self._compute_vector(
            query_repr=ep.query_repr,
            node_repr=new_node,
            hop=ep.hop_count,
            uncertainty=uncertainty,
        )
        return ep

    def get_current_episode(self) -> Optional[EpisodeState]:
        return self._episode

    # ------------------------------------------------------------------
    # State vector construction
    # ------------------------------------------------------------------

    def _compute_vector(
        self,
        query_repr: QueryRepresentation,
        node_repr: NodeRepresentation,
        hop: int,
        uncertainty: float,
    ) -> torch.Tensor:
        """
        Assemble and concatenate all state components into a single vector.

        Returns
        -------
        state_vec : (total_dim,)
        """
        sc = self.config.state
        device = self.device

        # 1. Query vector (query_dim,)
        q_vec = query_repr.embedding.to(device)

        # 2. Current node embedding (node_dim,)
        n_vec = node_repr.embedding.to(device)

        # 3. Node type embedding (node_type_dim,) — from index via lookup
        from src.retrieval.node_encoder import node_type_to_index
        type_idx = torch.tensor(
            node_type_to_index(node_repr.node_type),
            dtype=torch.long, device=device
        )

        # We use a fixed sinusoidal encoding for type instead of a learnable
        # table here to keep StateBuilder self-contained (the learnable table
        # lives in NodeEncoder).  The policy network can attend to both.
        type_vec = sinusoidal_encoding(
            type_idx.item(), sc.node_type_dim, device
        )

        # 4. Hop encoding (hop_encoding_dim,)
        hop_vec = sinusoidal_encoding(hop, sc.hop_encoding_dim, device)

        # 5. History GRU hidden (history_dim,)
        hist_vec = self._history_gru.get_hidden(device, sc.history_dim)

        # 6. Evidence GRU hidden (evidence_dim,)
        evid_vec = self._evidence_gru.get_hidden(device, sc.evidence_dim)

        # 7. Uncertainty encoding (uncertainty_dim,)
        unc_vec = self._uncertainty_enc(uncertainty, device)

        state_vec = torch.cat([
            q_vec,       # query_dim
            n_vec,       # node_dim
            type_vec,    # node_type_dim
            hop_vec,     # hop_encoding_dim
            hist_vec,    # history_dim
            evid_vec,    # evidence_dim
            unc_vec,     # uncertainty_dim
        ], dim=0)

        expected = sc.total_dim
        if state_vec.shape[0] != expected:
            logger.warning(
                "State vector dim mismatch: got %d expected %d",
                state_vec.shape[0], expected
            )
        return state_vec

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def extract_evidence_summary(self) -> List[str]:
        """Return collected evidence list from current episode."""
        if self._episode is None:
            return []
        return list(self._episode.collected_evidence)

    def extract_trajectory(self) -> List[Dict[str, Any]]:
        """Return structured traversal trajectory from current episode."""
        if self._episode is None:
            return []
        traj = []
        for step in self._episode.visited_steps:
            traj.append({
                "hop": step.hop,
                "node_id": step.node_id,
                "node_type": step.node_type,
                "relation": step.relation_type,
                "edge_weight": step.edge_weight,
                "evidence": step.evidence_text,
            })
        return traj

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """nn.Module forward — delegates to _compute_vector (for gradient flow)."""
        raise NotImplementedError(
            "Use StateBuilder.reset() / step() for episode management."
        )