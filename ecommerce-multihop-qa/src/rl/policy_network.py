"""
src/rl/policy_network.py

Adaptive Retrieval Policy Network.

Architecture
------------
The policy network maps (state_vector, action_embeddings) → action_logits.
It is query-aware: the query component of the state vector gates attention
over the candidate actions at each step.

Components
----------
1. StateEncoder MLP    — compresses state_vector → hidden_dim
2. ActionEncoder MLP   — projects each action_embedding → hidden_dim
3. Cross-attention     — state attends over action embeddings
4. Scoring head        — attention-weighted logits → softmax probabilities

The value head (critic) shares the StateEncoder backbone and outputs V(s).

This is the core contribution module:
"Policy that learns WHEN to traverse, WHERE to go, and WHEN to stop."
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config.settings import Config, DEFAULT_CONFIG
from src.retrieval.node_encoder import num_relation_types, num_node_types

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class StateEncoderMLP(nn.Module):
    """Encodes flat state vector → hidden representation."""

    def __init__(self, input_dim: int, hidden_dim: int,
                 num_layers: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_d = input_dim
        for i in range(num_layers - 1):
            layers += [
                nn.Linear(in_d, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_d = hidden_dim
        layers += [nn.Linear(in_d, hidden_dim), nn.LayerNorm(hidden_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActionEncoderMLP(nn.Module):
    """Projects action embedding (node_emb + edge_emb) → hidden_dim."""

    def __init__(self, input_dim: int, hidden_dim: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QueryGatedCrossAttention(nn.Module):
    """
    Multi-head cross-attention: state (query) attends over action candidates (keys/values).
    The query component of state is extracted and used as an additional gate.
    """

    def __init__(self, hidden_dim: int, num_heads: int,
                 query_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        # Gate: project raw query embedding to hidden_dim for gating
        self.query_gate = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        state_hidden: torch.Tensor,        # (1, hidden_dim)
        action_hiddens: torch.Tensor,      # (N_actions, hidden_dim)
        query_emb: torch.Tensor,           # (query_dim,)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        attended : (1, hidden_dim)
        attn_weights : (1, N_actions)
        """
        q = state_hidden.unsqueeze(0)            # (1, 1, hidden_dim)
        k = action_hiddens.unsqueeze(0)          # (1, N, hidden_dim)
        v = k

        attended, weights = self.attn(q, k, v)  # (1,1,H), (1,1,N)
        attended = attended.squeeze(0)           # (1, H)

        # Apply query gate
        gate = self.query_gate(query_emb.unsqueeze(0))  # (1, H)
        attended = attended * gate
        attended = self.norm(attended + state_hidden)
        return attended, weights.squeeze(0)      # (1,H), (1,N)


# ---------------------------------------------------------------------------
# Main Policy Network
# ---------------------------------------------------------------------------

class PolicyNetwork(nn.Module):
    """
    Adaptive Retrieval Policy Network (Actor + Critic shared backbone).

    Parameters
    ----------
    config : Config

    Inputs (forward)
    ----------------
    state_vec     : (state_dim,) or (B, state_dim)
    action_embs   : (N_actions, action_embed_dim) or (B, N, action_embed_dim)
    query_emb     : (query_dim,)  or (B, query_dim)
    action_mask   : (N_actions,) bool mask, True = valid

    Outputs
    -------
    action_logits : (N_actions,) unnormalised scores
    action_probs  : (N_actions,) softmax probabilities
    value         : scalar, V(s) estimate
    attn_weights  : (N_actions,) for interpretability
    """

    def __init__(self, config: Config = DEFAULT_CONFIG) -> None:
        super().__init__()
        self.config = config
        sc = config.state
        pc = config.policy

        state_dim = sc.total_dim
        action_dim = (
            config.encoder.node_feature_dim
            + config.state.edge_type_dim
        )  # node_emb + edge_emb concatenated

        logger.info(
            "PolicyNetwork: state_dim=%d action_dim=%d hidden=%d heads=%d",
            state_dim, action_dim, pc.hidden_dim, pc.attention_heads,
        )

        # --- Shared backbone ---
        self.state_encoder = StateEncoderMLP(
            input_dim=state_dim,
            hidden_dim=pc.hidden_dim,
            num_layers=pc.num_layers,
            dropout=pc.dropout,
        )

        self.action_encoder = ActionEncoderMLP(
            input_dim=action_dim,
            hidden_dim=pc.hidden_dim,
            dropout=pc.dropout,
        )

        self.cross_attn = QueryGatedCrossAttention(
            hidden_dim=pc.hidden_dim,
            num_heads=pc.attention_heads,
            query_dim=sc.query_dim,
            dropout=pc.dropout,
        )

        # --- Actor head ---
        self.actor_head = nn.Sequential(
            nn.Linear(pc.hidden_dim + pc.hidden_dim, pc.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(pc.hidden_dim // 2, 1),
        )

        # --- Critic head ---
        self.critic_head = nn.Sequential(
            nn.Linear(pc.hidden_dim, pc.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(pc.hidden_dim // 2, 1),
        )

        self._init_weights()
        logger.info(
            "PolicyNetwork parameters: %d",
            sum(p.numel() for p in self.parameters()),
        )

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        state_vec: torch.Tensor,
        action_embs: torch.Tensor,
        query_emb: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for a single step (no batch dim).

        Parameters
        ----------
        state_vec   : (state_dim,)
        action_embs : (N, action_embed_dim)
        query_emb   : (query_dim,)
        action_mask : (N,) bool, True = valid action; None = all valid

        Returns
        -------
        logits       : (N,)
        probs        : (N,)
        value        : scalar tensor
        attn_weights : (N,)
        """
        # Encode state
        s_hidden = self.state_encoder(state_vec)        # (hidden_dim,)

        # Encode all actions
        a_hiddens = self.action_encoder(action_embs)    # (N, hidden_dim)

        # Cross-attention: state over actions
        attended, attn_w = self.cross_attn(
            s_hidden.unsqueeze(0),    # (1, H)
            a_hiddens,                # (N, H)
            query_emb,                # (Q,)
        )
        attended = attended.squeeze(0)    # (H,)
        attn_w = attn_w.squeeze(0)        # (N,)

        # --- Compute per-action scores ---
        # Concatenate attended state with each action hidden
        N = a_hiddens.shape[0]
        attended_expanded = attended.unsqueeze(0).expand(N, -1)   # (N, H)
        actor_input = torch.cat([attended_expanded, a_hiddens], dim=-1)  # (N, 2H)
        logits = self.actor_head(actor_input).squeeze(-1)           # (N,)

        # Mask invalid actions (set to -inf so softmax → 0)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float("-inf"))

        probs = F.softmax(logits, dim=-1)

        # --- Value estimate ---
        value = self.critic_head(s_hidden).squeeze(-1)   # scalar

        return logits, probs, value, attn_w

    def get_log_probs(
        self,
        state_vec: torch.Tensor,
        action_embs: torch.Tensor,
        query_emb: torch.Tensor,
        action_indices: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute log-probabilities, entropy, and value for a batch of
        (state, selected_action) pairs — used in PPO update.

        Parameters
        ----------
        state_vec      : (B, state_dim)
        action_embs    : (B, N, action_embed_dim) — padded to max N
        query_emb      : (B, query_dim)
        action_indices : (B,) — selected action index at each step
        action_mask    : (B, N) bool

        Returns
        -------
        log_probs : (B,)
        entropy   : (B,)
        values    : (B,)
        """
        B = state_vec.shape[0]
        all_log_probs = []
        all_entropy = []
        all_values = []

        for i in range(B):
            mask_i = action_mask[i] if action_mask is not None else None
            logits, probs, value, _ = self.forward(
                state_vec[i],
                action_embs[i],
                query_emb[i],
                mask_i,
            )
            dist = torch.distributions.Categorical(probs=probs)
            log_p = dist.log_prob(action_indices[i])
            ent = dist.entropy()
            all_log_probs.append(log_p)
            all_entropy.append(ent)
            all_values.append(value)

        return (
            torch.stack(all_log_probs),
            torch.stack(all_entropy),
            torch.stack(all_values),
        )

    def select_action(
        self,
        state_vec: torch.Tensor,
        action_embs: torch.Tensor,
        query_emb: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[int, float, float, torch.Tensor]:
        """
        Sample or greedily select one action.

        Returns
        -------
        action_idx : int
        log_prob   : float
        value      : float
        attn_weights : (N,) for trajectory logging
        """
        with torch.no_grad():
            logits, probs, value, attn_w = self.forward(
                state_vec, action_embs, query_emb, action_mask
            )
        if deterministic:
            action_idx = probs.argmax().item()
        else:
            dist = torch.distributions.Categorical(probs=probs)
            action_idx = dist.sample().item()
        log_prob = torch.log(probs[action_idx] + 1e-8).item()
        return int(action_idx), log_prob, value.item(), attn_w