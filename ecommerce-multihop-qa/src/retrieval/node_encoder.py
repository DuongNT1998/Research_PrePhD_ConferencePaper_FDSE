"""
src/retrieval/node_encoder.py

KG Node Encoding module.

Responsibilities
----------------
1. Encode every KG node type (Product, Brand, Category, Feature, Aspect,
   Detail) into a unified dense vector of dimension node_feature_dim.
2. Encode edge (relation) types into relation embeddings used in action scoring.
3. Provide a cached, lazy-loading encoder so that Neo4j node properties are
   embedded on first access and reused afterwards.
4. Expose NodeRepresentation and EdgeRepresentation dataclasses consumed by
   StateBuilder and the dynamic action space generator.

Design notes
------------
- Each node type has a dedicated text template so heterogeneous properties are
  normalised before sentence-transformer encoding.
- A shared linear projection maps the base embedding → node_feature_dim.
- A lookup table encodes relation types (6 relation types → edge_type_dim).
- The module supports batch encoding via encode_nodes_batch().
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

from src.config.settings import Config, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class NodeRepresentation:
    """Dense representation of a single KG node."""
    node_id: str                      # Neo4j element id or unique property
    node_type: str                    # "Product" | "Brand" | ...
    # Projected embedding (node_feature_dim,) — used by policy / state vector
    embedding: torch.Tensor
    # Unprojected base embedding (embedding_dim,) from the sentence-transformer.
    # Lives in the SAME space as QueryRepresentation.base_embedding, so it is the
    # only embedding that may be compared against the query via cosine similarity.
    base_embedding: torch.Tensor
    # Raw text used to encode this node
    text_representation: str
    # Original properties dict from Neo4j
    properties: Dict[str, Any] = field(default_factory=dict)
    # Type index (used for type embedding lookup)
    type_index: int = 0


@dataclass
class EdgeRepresentation:
    """Representation of a directed edge (relation) in the KG."""
    relation_type: str
    # Embedding from relation type lookup table (edge_type_dim,)
    embedding: torch.Tensor
    # Numeric weight (e.g., aspect frequency)
    weight: float = 1.0
    # Edge direction metadata
    source_id: str = ""
    target_id: str = ""
    target_type: str = ""


# ---------------------------------------------------------------------------
# Node text template functions
# ---------------------------------------------------------------------------

def _product_text(props: Dict[str, Any]) -> str:
    parts: List[str] = []
    if props.get("title"):
        parts.append(props["title"])
    if props.get("main_category"):
        parts.append(f"category: {props['main_category']}")
    if props.get("price") is not None:
        parts.append(f"price: ${props['price']:.2f}")
    if props.get("average_rating") is not None:
        parts.append(f"rating: {props['average_rating']:.1f}")
    if props.get("rating_number") is not None:
        parts.append(f"reviews: {props['rating_number']}")
    return " | ".join(parts) if parts else "unknown product"


def _brand_text(props: Dict[str, Any]) -> str:
    return f"brand: {props.get('name', 'unknown')}"


def _category_text(props: Dict[str, Any]) -> str:
    return f"category: {props.get('name', 'unknown')}"


def _feature_text(props: Dict[str, Any]) -> str:
    return f"feature: {props.get('text', 'unknown')}"


def _aspect_text(props: Dict[str, Any]) -> str:
    return f"aspect: {props.get('name', 'unknown')}"


def _detail_text(props: Dict[str, Any]) -> str:
    key = props.get("key", "")
    value = props.get("value", "")
    return f"{key}: {value}"


# Map node type → text template function
_NODE_TEXT_FN = {
    "Product": _product_text,
    "Brand": _brand_text,
    "Category": _category_text,
    "Feature": _feature_text,
    "Aspect": _aspect_text,
    "Detail": _detail_text,
}

# Map node type → integer index
_NODE_TYPE_INDEX: Dict[str, int] = {
    nt: idx for idx, nt in enumerate(
        ["Product", "Brand", "Category", "Feature", "Aspect", "Detail"]
    )
}

# Map relation type → integer index
_RELATION_TYPE_INDEX: Dict[str, int] = {
    rt: idx for idx, rt in enumerate([
        "HAS_BRAND",
        "BELONGS_TO_CATEGORY",
        "HAS_FEATURE",
        "HAS_DETAIL",
        "HAS_POSITIVE_ASPECT",
        "HAS_NEGATIVE_ASPECT",
        "STOP",          # virtual "stop" action
    ])
}


# ---------------------------------------------------------------------------
# Projection MLP (shared across all node types)
# ---------------------------------------------------------------------------

class NodeProjectionMLP(nn.Module):
    """Projects sentence embedding → node_feature_dim."""

    def __init__(self, input_dim: int, output_dim: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        mid = (input_dim + output_dim) // 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, mid),
            nn.LayerNorm(mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Main encoder
# ---------------------------------------------------------------------------

class NodeEncoder:
    """
    Encodes KG nodes and edge relation types into dense vectors.

    Parameters
    ----------
    config : Config

    Usage
    -----
    >>> encoder = NodeEncoder(config)
    >>> node_repr = encoder.encode_node("Product", props, node_id="B07ABC")
    >>> edge_repr = encoder.encode_edge("HAS_POSITIVE_ASPECT", weight=5.0)
    """

    def __init__(self, config: Config = DEFAULT_CONFIG) -> None:
        self.config = config
        self.device = torch.device(config.encoder.device)

        logger.info(
            "NodeEncoder: loading sentence-transformer %s",
            config.encoder.sentence_model_name,
        )
        self._st_model = SentenceTransformer(
            config.encoder.sentence_model_name,
            device=config.encoder.device,
        )

        # Shared projection
        self._projection = NodeProjectionMLP(
            input_dim=config.encoder.embedding_dim,
            output_dim=config.encoder.node_feature_dim,
            dropout=config.policy.dropout,
        ).to(self.device)

        # Node type embedding lookup
        num_types = len(_NODE_TYPE_INDEX)
        self._type_embedding = nn.Embedding(
            num_types, config.state.node_type_dim
        ).to(self.device)

        # Relation type embedding lookup
        num_relations = len(_RELATION_TYPE_INDEX)
        self._relation_embedding = nn.Embedding(
            num_relations, config.state.edge_type_dim
        ).to(self.device)

        # Embedding cache: node_id → NodeRepresentation
        self._cache: Dict[str, NodeRepresentation] = {}

        logger.info(
            "NodeEncoder ready — node_feature_dim=%d edge_type_dim=%d",
            config.encoder.node_feature_dim,
            config.state.edge_type_dim,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode_node(
        self,
        node_type: str,
        properties: Dict[str, Any],
        node_id: str,
        use_cache: bool = True,
    ) -> NodeRepresentation:
        """
        Encode a single KG node.

        Parameters
        ----------
        node_type : str
            One of: Product, Brand, Category, Feature, Aspect, Detail.
        properties : dict
            Raw properties dict from Neo4j.
        node_id : str
            Unique identifier (parent_asin, brand_id, etc.).
        use_cache : bool
            Whether to return cached embedding on second call.

        Returns
        -------
        NodeRepresentation
        """
        cache_key = f"{node_type}:{node_id}"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        text_fn = _NODE_TEXT_FN.get(node_type, lambda p: str(p))
        text = text_fn(properties)
        type_idx = _NODE_TYPE_INDEX.get(node_type, 0)

        base_emb = self._embed_text(text)              # (embedding_dim,)
        proj_emb = self._project_node(base_emb)        # (node_feature_dim,)

        repr_ = NodeRepresentation(
            node_id=node_id,
            node_type=node_type,
            embedding=proj_emb,
            base_embedding=base_emb,
            text_representation=text,
            properties=properties,
            type_index=type_idx,
        )
        if use_cache:
            self._cache[cache_key] = repr_
        return repr_

    def encode_nodes_batch(
        self,
        nodes: List[Tuple[str, Dict[str, Any], str]],
    ) -> List[NodeRepresentation]:
        """
        Batch encode multiple nodes efficiently.

        Parameters
        ----------
        nodes : list of (node_type, properties, node_id)

        Returns
        -------
        list of NodeRepresentation
        """
        results: List[NodeRepresentation] = []
        texts: List[str] = []
        meta: List[Tuple[str, Dict[str, Any], str, int]] = []

        for node_type, props, node_id in nodes:
            cache_key = f"{node_type}:{node_id}"
            if cache_key in self._cache:
                results.append(self._cache[cache_key])
                texts.append("")          # placeholder
                meta.append(("__cached__", {}, "", 0))
            else:
                text_fn = _NODE_TEXT_FN.get(node_type, lambda p: str(p))
                text = text_fn(props)
                type_idx = _NODE_TYPE_INDEX.get(node_type, 0)
                texts.append(text)
                meta.append((node_type, props, node_id, type_idx))

        # Batch encode non-cached
        uncached_texts = [t for t, m in zip(texts, meta) if m[0] != "__cached__"]
        if uncached_texts:
            base_embs = self._embed_text_batch(uncached_texts)  # (N, emb_dim)
            proj_embs = self._project_node_batch(base_embs)      # (N, node_dim)

        emb_idx = 0
        final: List[NodeRepresentation] = []
        cached_iter = iter(results)
        for text, (node_type, props, node_id, type_idx) in zip(texts, meta):
            if node_type == "__cached__":
                final.append(next(cached_iter))
            else:
                repr_ = NodeRepresentation(
                    node_id=node_id,
                    node_type=node_type,
                    embedding=proj_embs[emb_idx],
                    base_embedding=base_embs[emb_idx],
                    text_representation=text,
                    properties=props,
                    type_index=type_idx,
                )
                self._cache[f"{node_type}:{node_id}"] = repr_
                final.append(repr_)
                emb_idx += 1
        return final

    def encode_edge(
        self,
        relation_type: str,
        weight: float = 1.0,
        source_id: str = "",
        target_id: str = "",
        target_type: str = "",
    ) -> EdgeRepresentation:
        """Encode a single KG edge/relation."""
        rel_idx = _RELATION_TYPE_INDEX.get(relation_type, 0)
        idx_tensor = torch.tensor(rel_idx, dtype=torch.long, device=self.device)
        with torch.no_grad():
            emb = self._relation_embedding(idx_tensor)   # (edge_type_dim,)
        return EdgeRepresentation(
            relation_type=relation_type,
            embedding=emb,
            weight=weight,
            source_id=source_id,
            target_id=target_id,
            target_type=target_type,
        )

    def get_type_embedding(self, node_type: str) -> torch.Tensor:
        """Return type embedding for a given node type string."""
        type_idx = _NODE_TYPE_INDEX.get(node_type, 0)
        idx_tensor = torch.tensor(type_idx, dtype=torch.long, device=self.device)
        with torch.no_grad():
            return self._type_embedding(idx_tensor)

    def clear_cache(self) -> None:
        """Flush the node embedding cache."""
        self._cache.clear()
        logger.debug("NodeEncoder cache cleared.")

    def get_projection_parameters(self):
        return list(self._projection.parameters()) + \
               list(self._type_embedding.parameters()) + \
               list(self._relation_embedding.parameters())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed_text(self, text: str) -> torch.Tensor:
        with torch.no_grad():
            emb = self._st_model.encode(
                text,
                convert_to_tensor=True,
                normalize_embeddings=True,
                device=self.config.encoder.device,
            )
        return emb.to(self.device)

    def _embed_text_batch(self, texts: List[str]) -> torch.Tensor:
        with torch.no_grad():
            embs = self._st_model.encode(
                texts,
                convert_to_tensor=True,
                normalize_embeddings=True,
                device=self.config.encoder.device,
                batch_size=64,
            )
        return embs.to(self.device)

    def _project_node(self, base_emb: torch.Tensor) -> torch.Tensor:
        self._projection.eval()
        with torch.no_grad():
            return self._projection(base_emb.unsqueeze(0)).squeeze(0)

    def _project_node_batch(self, base_embs: torch.Tensor) -> torch.Tensor:
        self._projection.eval()
        with torch.no_grad():
            return self._projection(base_embs)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def node_type_to_index(node_type: str) -> int:
    return _NODE_TYPE_INDEX.get(node_type, 0)


def relation_type_to_index(relation_type: str) -> int:
    return _RELATION_TYPE_INDEX.get(relation_type, 0)


def num_node_types() -> int:
    return len(_NODE_TYPE_INDEX)


def num_relation_types() -> int:
    return len(_RELATION_TYPE_INDEX)