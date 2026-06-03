"""
src/config/settings.py
Global configuration for the Adaptive Multi-hop Retrieval system.
All hyper-parameters, paths, and Neo4j credentials are centralised here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# Project root (two levels up from this file)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Neo4j connection
# ---------------------------------------------------------------------------
@dataclass
class Neo4jConfig:
    uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user: str = os.getenv("NEO4J_USER", "neo4j")
    password: str = os.getenv("NEO4J_PASSWORD", "chauduong")
    database: str = os.getenv("NEO4J_DATABASE", "neo4j")
    max_connection_lifetime: int = 3600
    max_connection_pool_size: int = 50
    connection_acquisition_timeout: float = 60.0


# ---------------------------------------------------------------------------
# Encoder settings
# ---------------------------------------------------------------------------
@dataclass
class EncoderConfig:
    # Sentence-Transformers model used for query & node encoding
    sentence_model_name: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384          # must match sentence_model_name output
    node_feature_dim: int = 128       # projected node embedding dim
    query_feature_dim: int = 256      # projected query embedding dim
    max_query_length: int = 128
    device: str = "cpu"               # "cuda" if GPU available


# ---------------------------------------------------------------------------
# Graph traversal / environment
# ---------------------------------------------------------------------------
@dataclass
class EnvironmentConfig:
    # Hard safety cap — only as an emergency brake, NOT core stopping logic
    max_hops: int = 10
    # Maximum neighbours to consider at each step (performance guard)
    max_neighbours: int = 30
    # Node types present in the KG
    node_types: List[str] = field(default_factory=lambda: [
        "Product", "Brand", "Category", "Feature", "Aspect", "Detail"
    ])
    # Relation types present in the KG
    relation_types: List[str] = field(default_factory=lambda: [
        "HAS_BRAND",
        "BELONGS_TO_CATEGORY",
        "HAS_FEATURE",
        "HAS_DETAIL",
        "HAS_POSITIVE_ASPECT",
        "HAS_NEGATIVE_ASPECT",
    ])
    # Numeric weight on aspect edges used in reward shaping
    aspect_weight_scale: float = 1.0


# ---------------------------------------------------------------------------
# State representation
# ---------------------------------------------------------------------------
@dataclass
class StateConfig:
    # Dimensionality of the concatenated state vector fed to the policy
    # = query_feature_dim + node_feature_dim + hop_dim + uncertainty_dim + type_dim
    query_dim: int = 256
    node_dim: int = 128
    hop_encoding_dim: int = 16   # sinusoidal or learned
    uncertainty_dim: int = 16
    node_type_dim: int = 16
    edge_type_dim: int = 16
    history_dim: int = 64        # GRU hidden for visited-node history
    evidence_dim: int = 64       # GRU hidden for collected evidence
    # Total state dim computed as property
    @property
    def total_dim(self) -> int:
        return (
            self.query_dim
            + self.node_dim
            + self.hop_encoding_dim
            + self.uncertainty_dim
            + self.node_type_dim
            + self.history_dim
            + self.evidence_dim
        )


# ---------------------------------------------------------------------------
# Policy / RL
# ---------------------------------------------------------------------------
@dataclass
class PolicyConfig:
    hidden_dim: int = 512
    num_layers: int = 3
    dropout: float = 0.1
    attention_heads: int = 4
    # Action embedding dim (for each candidate neighbour)
    action_embed_dim: int = 256


@dataclass
class PPOConfig:
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    gamma: float = 0.99           # discount factor
    gae_lambda: float = 0.95      # GAE lambda
    clip_epsilon: float = 0.2     # PPO clip
    entropy_coef: float = 0.01    # entropy bonus
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    mini_batch_size: int = 64
    update_every: int = 2048      # steps between updates


@dataclass
class ImitationConfig:
    lr: float = 1e-3
    epochs: int = 20
    batch_size: int = 32
    teacher_forcing_ratio: float = 0.8


# ---------------------------------------------------------------------------
# Reward weights
# ---------------------------------------------------------------------------
@dataclass
class RewardConfig:
    # Multi-objective reward weights
    w_answer_quality: float = 1.0
    w_retrieval_relevance: float = 0.5
    w_path_quality: float = 0.3
    w_grounding: float = 0.4
    w_efficiency_penalty: float = -0.05   # per unnecessary hop
    w_uncertainty_stop: float = 0.3       # bonus for stopping at right time
    # Thresholds
    high_quality_threshold: float = 0.7
    low_quality_threshold: float = 0.3


# ---------------------------------------------------------------------------
# Stopping / uncertainty
# ---------------------------------------------------------------------------
@dataclass
class StoppingConfig:
    # Bayesian / ensemble uncertainty estimation
    mc_dropout_samples: int = 10
    uncertainty_threshold: float = 0.25   # stop when U < this
    confidence_threshold: float = 0.75    # stop when confidence > this
    min_hops: int = 1                     # never stop before 1 hop
    evidence_saturation_threshold: float = 0.05  # delta evidence < this → stop


# ---------------------------------------------------------------------------
# LLM (weak supervision only)
# ---------------------------------------------------------------------------
@dataclass
class LLMConfig:
    model_name: str = "claude-sonnet-4-20250514"
    max_tokens: int = 1024
    temperature: float = 0.0
    # Anthropic API base (proxy handles key injection)
    api_base: str = "https://api.anthropic.com/v1/messages"
    # How many retrieved evidences to pass to LLM for answer synthesis
    max_evidence_items: int = 10


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
@dataclass
class PathConfig:
    data_raw: Path = PROJECT_ROOT / "data" / "raw"
    data_interim: Path = PROJECT_ROOT / "data" / "interim"
    data_processed: Path = PROJECT_ROOT / "data" / "processed"
    outputs: Path = PROJECT_ROOT / "outputs"
    checkpoints: Path = PROJECT_ROOT / "outputs" / "checkpoints"
    logs: Path = PROJECT_ROOT / "outputs" / "logs"
    results: Path = PROJECT_ROOT / "outputs" / "results"
    figures: Path = PROJECT_ROOT / "outputs" / "figures"


# ---------------------------------------------------------------------------
# Master config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    state: StateConfig = field(default_factory=StateConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    imitation: ImitationConfig = field(default_factory=ImitationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    stopping: StoppingConfig = field(default_factory=StoppingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    seed: int = 42
    log_level: str = "INFO"


# Singleton-style default config accessible everywhere
DEFAULT_CONFIG = Config()