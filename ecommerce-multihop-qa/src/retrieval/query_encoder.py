"""
src/retrieval/query_encoder.py

Query Understanding & Encoding module.

Responsibilities
----------------
1. Tokenise and encode the raw user query into a dense semantic vector.
2. Project the semantic vector into the policy-compatible query space.
3. Extract structured intent signals: constraint types (positive / negative),
   target node types, and key entity mentions.
4. Expose a QueryRepresentation dataclass consumed by StateBuilder and the
   KGEnvironment at every reasoning step.

Design notes
------------
- Uses sentence-transformers for base encoding (offline, no LLM calls here).
- A lightweight MLP projects the base embedding into query_feature_dim.
- Intent extraction is rule-based over Electronics domain keywords so that
  negative constraints such as "not too heavy" are captured explicitly.
- The module is stateless and thread-safe; encode() can be called in parallel.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

from src.config.settings import Config, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain knowledge for Electronics query understanding
# ---------------------------------------------------------------------------

# Positive constraint signal words
_POSITIVE_KEYWORDS: List[str] = [
    "good", "great", "best", "fast", "long", "high", "excellent",
    "powerful", "lightweight", "durable", "reliable", "responsive",
    "clear", "bright", "loud", "accurate", "efficient", "premium",
    "quality", "worth", "recommend", "love", "perfect", "amazing",
]

# Negative constraint signal words
_NEGATIVE_KEYWORDS: List[str] = [
    "not", "no", "avoid", "without", "never", "bad", "poor",
    "heavy", "slow", "buggy", "fragile", "overheating", "laggy",
    "crashing", "bulky", "expensive", "overpriced", "disappointing",
]

# Aspect → KG node hint mapping
_ASPECT_TO_NODE_TYPE: Dict[str, str] = {
    "battery": "Aspect",
    "screen": "Aspect",
    "performance": "Aspect",
    "sound": "Aspect",
    "camera": "Aspect",
    "bluetooth": "Aspect",
    "wifi": "Aspect",
    "design": "Aspect",
    "weight": "Aspect",
    "price": "Product",
    "warranty": "Feature",
    "brand": "Brand",
    "category": "Category",
}

# Target node type hints from query vocabulary
_NODE_TYPE_HINTS: Dict[str, str] = {
    "brand": "Brand",
    "category": "Category",
    "feature": "Feature",
    "detail": "Detail",
    "aspect": "Aspect",
    "product": "Product",
    "laptop": "Product",
    "phone": "Product",
    "headphone": "Product",
    "speaker": "Product",
    "tablet": "Product",
    "charger": "Product",
    "earphone": "Product",
    "keyboard": "Product",
    "mouse": "Product",
    "monitor": "Product",
}

# Numeric filter pattern: "under 500", "below 20 million", "less than 100"
_NUMERIC_PATTERN = re.compile(
    r"(under|below|less than|more than|above|over|at least|at most)\s*"
    r"(\d+(?:[.,]\d+)?)\s*"
    r"(dollar|usd|\$|k|million|gb|tb|mb|mah|inch|hz|watt|w)?",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class QueryConstraint:
    """A single extracted constraint from the query."""
    text: str                          # raw constraint phrase
    polarity: str                      # "positive" | "negative" | "neutral"
    aspect: Optional[str] = None       # matched aspect keyword if any
    numeric_op: Optional[str] = None   # "under" | "above" | None
    numeric_value: Optional[float] = None
    numeric_unit: Optional[str] = None
    target_node_type: Optional[str] = None


@dataclass
class QueryRepresentation:
    """Full structured representation of a user query."""
    raw_text: str
    # Dense semantic embedding (query_feature_dim,)
    embedding: torch.Tensor
    # Unprojected base embedding from sentence-transformer
    base_embedding: torch.Tensor
    # Structured signals
    constraints: List[QueryConstraint] = field(default_factory=list)
    positive_constraints: List[QueryConstraint] = field(default_factory=list)
    negative_constraints: List[QueryConstraint] = field(default_factory=list)
    # Target node type hints for anchor node selection
    target_node_types: List[str] = field(default_factory=list)
    # Aspect keywords detected in query
    detected_aspects: List[str] = field(default_factory=list)
    # Product type keywords detected (e.g., "laptop", "phone")
    product_type_hints: List[str] = field(default_factory=list)
    # Tokens for BM25-style sparse matching
    tokens: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Projection MLP
# ---------------------------------------------------------------------------

class QueryProjectionMLP(nn.Module):
    """Projects base sentence embedding → policy-compatible query vector."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 512,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Main encoder
# ---------------------------------------------------------------------------

class QueryEncoder:
    """
    Encodes a natural-language user query into a QueryRepresentation.

    Parameters
    ----------
    config : Config
        Master config object.

    Usage
    -----
    >>> encoder = QueryEncoder(config)
    >>> qr = encoder.encode("laptop under 500 dollars, long battery, not too heavy")
    >>> qr.embedding.shape
    torch.Size([256])
    """

    def __init__(self, config: Config = DEFAULT_CONFIG) -> None:
        self.config = config
        self.device = torch.device(config.encoder.device)

        logger.info(
            "Loading sentence-transformer: %s",
            config.encoder.sentence_model_name,
        )
        self._st_model = SentenceTransformer(
            config.encoder.sentence_model_name,
            device=config.encoder.device,
        )

        self._projection = QueryProjectionMLP(
            input_dim=config.encoder.embedding_dim,
            output_dim=config.state.query_dim,
            hidden_dim=512,
            dropout=config.policy.dropout,
        ).to(self.device)

        # Freeze projection initially; will be fine-tuned during RL
        # (caller can call encoder.unfreeze_projection() when ready)
        for p in self._projection.parameters():
            p.requires_grad = False

        logger.info(
            "QueryEncoder ready — base_dim=%d projected_dim=%d",
            config.encoder.embedding_dim,
            config.state.query_dim,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, query_text: str) -> QueryRepresentation:
        """
        Full pipeline: raw text → QueryRepresentation.

        Parameters
        ----------
        query_text : str
            Raw English user query.

        Returns
        -------
        QueryRepresentation
        """
        query_text = query_text.strip()
        tokens = self._tokenise(query_text)
        base_emb = self._embed_sentence(query_text)           # (base_dim,)
        projected_emb = self._project(base_emb)               # (query_dim,)
        constraints = self._extract_constraints(query_text, tokens)
        pos_constraints = [c for c in constraints if c.polarity == "positive"]
        neg_constraints = [c for c in constraints if c.polarity == "negative"]
        aspects = self._detect_aspects(tokens)
        node_types = self._infer_target_node_types(tokens, constraints)
        product_hints = self._detect_product_types(tokens)

        return QueryRepresentation(
            raw_text=query_text,
            embedding=projected_emb,
            base_embedding=base_emb,
            constraints=constraints,
            positive_constraints=pos_constraints,
            negative_constraints=neg_constraints,
            target_node_types=node_types,
            detected_aspects=aspects,
            product_type_hints=product_hints,
            tokens=tokens,
        )

    def encode_batch(self, queries: List[str]) -> List[QueryRepresentation]:
        """Encode a list of queries efficiently."""
        return [self.encode(q) for q in queries]

    def unfreeze_projection(self) -> None:
        """Allow projection MLP gradients (called before RL stage)."""
        for p in self._projection.parameters():
            p.requires_grad = True
        logger.info("QueryEncoder projection MLP unfrozen for training.")

    def get_projection_parameters(self):
        """Return parameters of projection MLP for optimiser."""
        return self._projection.parameters()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tokenise(self, text: str) -> List[str]:
        """Simple whitespace + punctuation tokeniser."""
        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        return tokens

    def _embed_sentence(self, text: str) -> torch.Tensor:
        """Embed via sentence-transformer; return (embedding_dim,) tensor."""
        with torch.no_grad():
            emb = self._st_model.encode(
                text,
                convert_to_tensor=True,
                normalize_embeddings=True,
                device=self.config.encoder.device,
            )
        return emb.to(self.device)

    def _project(self, base_emb: torch.Tensor) -> torch.Tensor:
        """Project base embedding through MLP."""
        self._projection.eval()
        with torch.no_grad():
            projected = self._projection(base_emb.unsqueeze(0)).squeeze(0)
        return projected

    def _extract_constraints(
        self, text: str, tokens: List[str]
    ) -> List[QueryConstraint]:
        """
        Extract structured constraints from the query.
        Handles:
        - Positive aspects: "good battery", "long battery life"
        - Negative constraints: "not too heavy", "no overheating"
        - Numeric filters: "under 500 dollars", "more than 8 hours"
        """
        constraints: List[QueryConstraint] = []
        text_lower = text.lower()

        # --- Numeric constraints ---
        for match in _NUMERIC_PATTERN.finditer(text):
            op = match.group(1).lower()
            val_str = match.group(2).replace(",", "")
            unit = match.group(3) or ""
            polarity = "positive" if op in ("more than", "above", "over", "at least") else "negative"
            constraints.append(QueryConstraint(
                text=match.group(0),
                polarity=polarity,
                numeric_op=op,
                numeric_value=float(val_str),
                numeric_unit=unit.lower(),
                target_node_type="Product",
            ))

        # --- Negative keyword window scan ---
        neg_indices: List[int] = []
        for i, tok in enumerate(tokens):
            if tok in _NEGATIVE_KEYWORDS:
                neg_indices.append(i)

        # --- Aspect + polarity constraints ---
        for aspect, node_type in _ASPECT_TO_NODE_TYPE.items():
            if aspect in text_lower:
                # Determine polarity: check window [-3, +1] around aspect for negation
                polarity = self._polarity_near_aspect(
                    tokens, aspect, neg_indices
                )
                constraints.append(QueryConstraint(
                    text=aspect,
                    polarity=polarity,
                    aspect=aspect,
                    target_node_type=node_type,
                ))

        # Deduplicate by text
        seen: set = set()
        unique: List[QueryConstraint] = []
        for c in constraints:
            if c.text not in seen:
                seen.add(c.text)
                unique.append(c)
        return unique

    def _polarity_near_aspect(
        self,
        tokens: List[str],
        aspect: str,
        neg_indices: List[int],
    ) -> str:
        """Check if any negative signal token is within a 4-token window of the aspect."""
        aspect_tokens = aspect.split()
        for i, tok in enumerate(tokens):
            if tok == aspect_tokens[0]:
                window = set(range(max(0, i - 4), min(len(tokens), i + len(aspect_tokens) + 3)))
                for ni in neg_indices:
                    if ni in window:
                        return "negative"
                return "positive"
        return "positive"

    def _detect_aspects(self, tokens: List[str]) -> List[str]:
        """Return list of aspect keywords found in tokens."""
        return [a for a in _ASPECT_TO_NODE_TYPE if a in tokens]

    def _infer_target_node_types(
        self,
        tokens: List[str],
        constraints: List[QueryConstraint],
    ) -> List[str]:
        """Infer which KG node types are likely relevant for this query."""
        types: set = set()
        for tok in tokens:
            if tok in _NODE_TYPE_HINTS:
                types.add(_NODE_TYPE_HINTS[tok])
        for c in constraints:
            if c.target_node_type:
                types.add(c.target_node_type)
        # Always include Product as default anchor
        types.add("Product")
        return list(types)

    def _detect_product_types(self, tokens: List[str]) -> List[str]:
        """Detect product category hints (laptop, phone, etc.)."""
        product_words = {
            "laptop", "phone", "smartphone", "headphone", "speaker",
            "tablet", "charger", "earphone", "keyboard", "mouse",
            "monitor", "camera", "smartwatch", "router", "printer",
        }
        return [t for t in tokens if t in product_words]


# ---------------------------------------------------------------------------
# Utility: cosine similarity between two query representations
# ---------------------------------------------------------------------------

def query_similarity(qr1: QueryRepresentation, qr2: QueryRepresentation) -> float:
    """Cosine similarity between two encoded queries."""
    v1 = qr1.embedding
    v2 = qr2.embedding
    cos = nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0))
    return cos.item()