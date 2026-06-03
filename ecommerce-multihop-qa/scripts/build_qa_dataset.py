"""
scripts/build_qa_dataset.py

Synthetic QA dataset generator for the adaptive multi-hop KG retrieval RL agent.

The generator is *seed-anchored* and *reverse-constructed*: it samples a real
Product from Neo4j, reads its true neighbourhood (category / brand / positive &
negative aspects / features / details / price), then builds a natural-language
English query from templates whose answer is GUARANTEED to be non-empty (the seed
itself always satisfies the constraints).  Gold answers are then recovered by
running an exact Cypher constraint query, so every sample has verified labels.

Output: JSONL, one sample per line:
    {
      "qid": "...",
      "query": "...",
      "type": "simple | multi_hop | negative | constraint",
      "gold_answers": ["B0XXXX", ...],
      "constraints": {
          "category": str|null, "brand": str|null, "price_max": float|null,
          "positive_aspects": [...], "negative_aspects": [...],
          # optional extra keys when used: "features": [...], "details": [...]
      },
      "reasoning_path": ["Product -[REL]-> Node(name)", ...]
    }

Usage
-----
    export NEO4J_URI=bolt://localhost:7687
    export NEO4J_USER=neo4j
    export NEO4J_PASSWORD=...
    python scripts/build_qa_dataset.py --total 3000 --out-dir data/processed
    python scripts/build_qa_dataset.py --self-test     # no DB needed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make `src` importable when run as `python scripts/build_qa_dataset.py`
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger("build_qa_dataset")


# ===========================================================================
# Natural-language phrasing tables (template-based, NOT LLM)
# ===========================================================================

from typing import Dict, List

# Cấu trúc từ điển Khía cạnh -> Danh sách các cụm từ khẳng định/tích cực (Bản đầy đủ nhất)
ASPECT_POSITIVE_PHRASES: Dict[str, List[str]] = {
    # --- PIN & NĂNG LƯỢNG ---
    "battery": [
        "good battery life", "long battery", "great battery", "excellent battery endurance", 
        "lasts all day", "amazing battery life", "stellar battery performance", "decent battery",
        "impressive battery", "holds a charge well", "fantastic battery"
    ],
    "charging": [
        "fast charging", "quick charge", "rapid charging", "convenient wireless charging",
        "charges super fast", "charges in no time", "efficient charging"
    ],

    # --- MÀN HÌNH & HIỂN THỊ ---
    "screen": [
        "a sharp screen", "good display", "a nice screen", "vibrant screen", 
        "beautiful panel", "stunning screen", "gorgeous screen", "bright screen", 
        "crystal clear screen", "flawless screen", "responsive touch screen"
    ],
    "display": [
        "a great display", "a crisp display", "stunning display", "vibrant display", 
        "excellent viewing angles", "bright display", "high refresh rate display", 
        "smooth display", "beautiful display quality", "color-accurate display"
    ],

    # --- HIỆU NĂNG & TỐC ĐỘ ---
    "performance": [
        "fast performance", "high performance", "snappy performance", "smooth performance", 
        "beast of a performer", "excellent multitasking", "lag-free performance", 
        "solid performance", "powerful performance", "flawless execution"
    ],
    "speed": [
        "fast speed", "quick response", "blazing fast", "lightning fast", 
        "very responsive", "super quick", "zero lag", "boots up instantly", "speedy operation"
    ],

    # --- ÂM THANH ---
    "sound": [
        "good sound quality", "great sound", "rich audio", "amazing sound", 
        "stunning sound quality", "crisp sound", "punchy bass", "impressive soundstage",
        "loud and clear sound", "immersive sound", "crystal-clear audio output"
    ],
    "audio": [
        "great audio", "clear audio", "excellent audio quality", "immersive audio", 
        "crisp audio", "high-fidelity audio", "clean audio", "phenomenal audio", "balanced audio"
    ],

    # --- PHẦN CỨNG BỔ TRỢ ---
    "camera": [
        "a great camera", "an excellent camera", "sharp camera quality", "stunning photos", 
        "amazing camera performance", "crisp video quality", "great low-light photos",
        "superb camera", "high-resolution camera"
    ],
    "microphone": [
        "a good microphone", "clear mic", "excellent mic quality", "crisp voice capture", 
        "good background noise cancellation", "studio-quality mic", "perfect microphone"
    ],
    "keyboard": [
        "a good keyboard", "tactile keys", "comfortable typing experience", "responsive keyboard", 
        "great key travel", "satisfying clicks", "well-spaced keyboard", "premium keyboard feel"
    ],
    "mouse_trackpad": [
        "smooth trackpad", "responsive touchpad", "accurate mouse tracking", "ergonomic mouse",
        "clicky mouse buttons", "precise trackpad"
    ],

    # --- KẾT NỐI ---
    "bluetooth": [
        "reliable bluetooth", "stable bluetooth connection", "seamless bluetooth pairing", 
        "quick bluetooth connection", "strong bluetooth signal", "effortless bluetooth sync"
    ],
    "wifi": [
        "strong wifi", "good wifi range", "stable wifi connection", "fast wifi speeds", 
        "excellent wifi reception", "reliable wifi", "no wifi drops"
    ],
    "connectivity": [
        "solid connectivity", "reliable connectivity", "versatile ports", "great port selection", 
        "seamless connectivity", "excellent wireless range", "has plenty of ports"
    ],

    # --- THIẾT KẾ & ĐỘ HOÀN THIỆN ---
    "design": [
        "a sleek design", "a premium build", "nice design", "beautiful aesthetics", 
        "modern look", "elegant design", "stylish appearance", "ergonomic design", 
        "attractive design", "minimalist design"
    ],
    "build": [
        "a sturdy build", "solid build quality", "premium build", "robust construction", 
        "durable build quality", "feels high-end", "well-constructed", "tough build", "well-made"
    ],
    "weight": [
        "a lightweight design", "light and portable", "feather-light", "very light", 
        "not heavy at all", "perfect weight distribution", "unbelievably light"
    ],
    "portability": [
        "good portability", "easy to carry", "highly portable", "travel-friendly", 
        "compact size", "perfect for travel", "easy to pack"
    ],
    "comfort": [
        "a comfortable fit", "great comfort", "very comfortable to wear", "ergonomic and comfy", 
        "fits perfectly", "easy on the ears", "easy on the hands", "pleasant to use for hours"
    ],

    # --- ĐỘ BỀN & BẢO HÀNH ---
    "durability": [
        "good durability", "a durable build", "built to last", "highly durable", 
        "withstands drops", "rugged durability", "long-lasting material", "very resilient"
    ],
    "warranty": [
        "a good warranty", "solid warranty coverage", "excellent warranty policy", 
        "peace of mind warranty", "great customer support warranty", "reliable warranty service"
    ],

    # --- GIÁ TRỊ & CHI PHÍ ---
    "value": [
        "good value", "great value for money", "excellent value", "bang for the buck", 
        "worth every penny", "highly cost-effective", "incredible value", "best value option"
    ],
    "price": [
        "good value", "an affordable price", "great price point", "budget-friendly price", 
        "reasonably priced", "excellent price", "fair price", "competitive pricing"
    ],
    "quality": [
        "good build quality", "high quality", "premium quality", "top-notch quality", 
        "excellent overall quality", "superb quality", "outstanding quality", "first-rate quality"
    ],

    # --- TẢN NHIỆT & PHẦN MỀM ---
    "cooling": [
        "good cooling", "effective cooling", "runs cool", "excellent thermal management", 
        "quiet fans", "no overheating issues", "stays cold under load"
    ],
    "dpi": [
        "high DPI", "precise tracking", "adjustable DPI", "accurate tracking", 
        "smooth sensor", "very precise", "perfect DPI switching"
    ],
    "storage": [
        "plenty of storage", "large storage", "generous storage capacity", "massive storage", 
        "fast SSD storage", "ample space", "more than enough storage"
    ],
    "software": [
        "clean software", "smooth OS", "user-friendly interface", "easy to set up", 
        "intuitive software", "bloatware-free", "great companion app"
    ],
    
    # --- KHÁC (TÍNH NĂNG ĐẶC TRƯNG CỦA AUDIO/ELECTRONICS) ---
    "noise_cancellation": [
        "excellent ANC", "great noise cancellation", "blocks out noise completely",
        "amazing active noise cancelling", "superb isolation"
    ]
}



# Cấu trúc từ điển Khía cạnh -> Danh sách cụm từ phủ định, loại trừ (Bản đầy đủ nhất)
ASPECT_NEGATIVE_PHRASES: Dict[str, List[str]] = {
    # --- PIN & NĂNG LƯỢNG ---
    "battery": [
        "without battery issues", "no battery drain", "but not draining too fast", 
        "without rapid battery loss", "no short battery life", "without constant recharging",
        "not dying quickly", "without poor battery endurance"
    ],
    "charging": [
        "without slow charging", "no sluggish charging", "but not taking too long to charge",
        "without overheating during charge", "no broken charging ports"
    ],

    # --- MÀN HÌNH & HIỂN THỊ ---
    "screen": [
        "without screen issues", "not blurry", "without screen glare", "no dead pixels",
        "not flickering", "without a dim screen", "not easily scratched screen"
    ],
    "display": [
        "without display problems", "not washed out colors", "no ghosting issues", 
        "without low brightness", "not reflective display", "without bad viewing angles",
        "no screen bleeding"
    ],

    # --- HIỆU NĂNG, TỐC ĐỘ & NHIỆT ĐỘ ---
    "performance": [
        "without lag", "no slowdowns", "without stuttering", "not laggy", 
        "without freezing", "no performance drops", "without crashing", "not sluggish"
    ],
    "speed": [
        "without being slow", "no lag", "not slow to respond", "without buffering", 
        "no delay", "without bottlenecking"
    ],
    "heat": [
        "without overheating", "that doesn't overheat", "without getting too hot", 
        "no thermal issues", "without burning up"
    ],
    "overheating": [
        "without overheating", "that stays cool", "no thermal throttling", 
        "without heating up under load", "does not overheat"
    ],
    "temperature": [
        "without overheating", "no high temperatures", "without running hot", 
        "keeps a safe temperature"
    ],
    "cooling": [
        "without cooling failure", "no loud fan noise", "without thermal throttling",
        "not running too hot"
    ],

    # --- ÂM THANH & TIẾNG ỒN ---
    "sound": [
        "without sound issues", "no audio problems", "not muffled", "without tinny sound", 
        "no static noise", "not distorted sound", "without low volume"
    ],
    "audio": [
        "without audio lag", "no static buzzing", "not low quality audio", 
        "without flat sound", "no crackling audio"
    ],
    "noise": [
        "without noise problems", "not noisy", "without high background noise", 
        "no annoying hissing", "without fan hums"
    ],
    "noise_cancellation": [
        "without poor ANC", "no ambient noise leaking", "without failing to block noise",
        "not letting background sounds in"
    ],

    # --- PHẦN CỨNG BỔ TRỢ ---
    "camera": [
        "without camera problems", "not grainy photos", "no blurry video", 
        "without laggy camera", "not poor low-light quality"
    ],
    "microphone": [
        "without mic issues", "no muffled voice", "without catching background noise", 
        "not static microphone", "no echo problems"
    ],
    "keyboard": [
        "without mushy keys", "not a stiff keyboard", "no sticky keys", 
        "without loud clicking", "not cheap feeling keys"
    ],
    "mouse_trackpad": [
        "without a laggy trackpad", "no jumpy cursor", "not unresponsive buttons",
        "without tracking issues"
    ],
    "storage": [
        "without running out of space", "not low storage", "without limited capacity", 
        "no slow storage speeds"
    ],

    # --- KẾT NỐI ---
    "connectivity": [
        "without connection drops", "no connectivity issues", "without signal drops", 
        "no disconnection problems", "without pairing issues"
    ],
    "bluetooth": [
        "without bluetooth dropouts", "no bluetooth lag", "without connection drops", 
        "no pairing failures"
    ],
    "wifi": [
        "without wifi drops", "no unstable wifi", "without losing internet connection", 
        "no weak wifi signal"
    ],

    # --- THIẾT KẾ, ĐỘ BỀN & ĐỘ HOÀN THIỆN ---
    "build": [
        "without feeling cheap", "not flimsy", "no cheap plastic feel", 
        "without rattling parts", "not poorly made"
    ],
    "durability": [
        "without durability issues", "not fragile", "not breaking easily", 
        "without wearing out quickly", "no scratchable body"
    ],
    "weight": [
        "not too heavy", "without being bulky", "not bulky", "without being cumbersome", 
        "not weighing a ton", "not brick-like"
    ],
    "size": [
        "not too big", "not oversized", "without taking up too much space", 
        "not too small", "not bulky size"
    ],
    "comfort": [
        "without causing ear pain", "not uncomfortable", "without straining my wrists", 
        "not hurting after hours", "without feeling tight"
    ],
    "design": [
        "not an ugly design", "without outdated looks", "not poorly designed",
        "without awkward button placement"
    ],

    # --- GIÁ CẢ & PHẦN MỀM ---
    "price": [
        "not too expensive", "without breaking the bank", "not overpriced", 
        "without costing a fortune", "not too pricey", "not paying an arm and a leg"
    ],
    "value": [
        "not a rip-off", "not a waste of money", "without being a bad investment",
        "not overpriced for what it offers"
    ],
    "quality": [
        "not poor quality", "without defects", "no cheap materials", 
        "not failing quality control"
    ],
    "software": [
        "without bloatware", "no system glitches", "without annoying software bugs", 
        "no laggy interface", "without crashing apps"
    ]
}

# ============================================================================
# BỘ TỪ NỐI VÀ TỪ DẪN ĐỂ NGẪU NHIÊN HÓA CÂU TRUY VẤN (QUERY GENERATION)
# ============================================================================

# Từ nối khẳng định: Liên kết sản phẩm với một khía cạnh tích cực (Positive Aspect).
# Đã bổ sung các dạng phân từ (participle) và mệnh đề quan hệ để tăng tính tự nhiên.
POS_CONNECTORS = [
    "with", 
    "that has", 
    "featuring", 
    "offering", 
    "which has",
    "equipped with", 
    "boasting", 
    "that comes with", 
    "comes with",
    "that provides", 
    "having", 
    "built with", 
    "packed with",
    "that delivers"
]

# Từ nối phủ định/loại trừ: Liên kết khía cạnh tích cực trước đó với một điều kiện né tránh (Negative Constraint).
# Rất quan trọng cho bài toán Multi-hop Reasoning khi User muốn lọc bỏ các lỗi phần cứng.
NEG_CONNECTORS = [
    "but", 
    "though", 
    "yet", 
    "but ideally", 
    "but without", 
    "while avoiding",
    "but not", 
    "provided it has no", 
    "but with no", 
    "as long as it has no",
    "however without", 
    "but avoiding", 
    "minus any", 
    "without having"
]

# Cụm từ mở đầu (Dẫn nhập): Cách người dùng bắt đầu một câu hỏi hoặc lệnh tìm kiếm.
# Bao gồm cả chuỗi rỗng "", văn phong trang trọng, và văn phong hội thoại (Chatbot style).
SIMPLE_LEADS = [
    "", 
    "looking for a ", 
    "I want a ", 
    "show me a ", 
    "need a ", 
    "find a ",
    "can you find a ", 
    "search for a ", 
    "I am looking for a ", 
    "please show me a ",
    "get me a ", 
    "recommend a ", 
    "I'm in need of a ", 
    "looking to buy a ",
    "is there a ", 
    "help me find a ", 
    "any recommendations for a "
]

# ============================================================================
# CẤU TRÚC DỰ PHÒNG CHUNG (FALLBACK DICTIONARIES)
# Sử dụng khi Aspect trích xuất từ đồ thị không nằm trong từ điển cụm từ tự nhiên.
# ============================================================================
_GENERIC_POS_FALLBACK = "good {name}"
_GENERIC_NEG_FALLBACK = "without {name} issues"


def positive_phrase(aspect_name: str, rng: random.Random) -> str:
    key = aspect_name.strip().lower()
    options = ASPECT_POSITIVE_PHRASES.get(key)
    if options:
        return rng.choice(options)
    return _GENERIC_POS_FALLBACK.format(name=key)


def negative_phrase(aspect_name: str, rng: random.Random) -> str:
    key = aspect_name.strip().lower()
    options = ASPECT_NEGATIVE_PHRASES.get(key)
    if options:
        return rng.choice(options)
    return _GENERIC_NEG_FALLBACK.format(name=key)


def clean_category(name: str) -> str:
    return (name or "product").strip().lower()


def clean_feature_snippet(text: str, max_words: int = 4) -> str:
    """Take a short, query-friendly snippet from a (possibly long) feature."""
    words = (text or "").strip().split()
    snippet = " ".join(words[:max_words]).lower().rstrip(".,;:")
    return snippet


# ===========================================================================
# Data structures
# ===========================================================================

@dataclass
class Expansion:
    """The true neighbourhood of a seed product, read from the KG."""
    asin: str
    title: str
    price: Optional[float]
    cat_id: Optional[str]
    cat_name: Optional[str]
    brand_id: Optional[str]
    brand_name: Optional[str]
    pos_aspects: List[Dict[str, Any]] = field(default_factory=list)   # {id,name,weight}
    neg_aspects: List[Dict[str, Any]] = field(default_factory=list)
    features: List[Dict[str, Any]] = field(default_factory=list)      # {id,text}
    details: List[Dict[str, Any]] = field(default_factory=list)       # {id,key,value}


@dataclass
class ConstraintSpec:
    """Internal (id-based) constraints used to recover exact gold answers."""
    category_id: Optional[str] = None
    brand_id: Optional[str] = None
    price_max: Optional[float] = None
    pos_aspect_ids: List[str] = field(default_factory=list)
    exclude_neg_aspect_ids: List[str] = field(default_factory=list)
    feature_ids: List[str] = field(default_factory=list)
    detail_ids: List[str] = field(default_factory=list)


@dataclass
class QASample:
    qid: str
    query: str
    type: str
    gold_answers: List[str]
    constraints: Dict[str, Any]
    reasoning_path: List[str]

    def to_json(self) -> str:
        return json.dumps(
            {
                "qid": self.qid,
                "query": self.query,
                "type": self.type,
                "gold_answers": self.gold_answers,
                "constraints": self.constraints,
                "reasoning_path": self.reasoning_path,
            },
            ensure_ascii=False,
        )


# ===========================================================================
# Generator
# ===========================================================================

TYPE_RATIOS = {
    "simple": 0.30,
    "multi_hop": 0.40,
    "negative": 0.20,
    "constraint": 0.10,
}


class QASyntheticGenerator:
    """
    Generates a balanced synthetic QA dataset from a Neo4j KG.

    Parameters
    ----------
    uri, user, password, database : Neo4j connection parameters.
    seed : RNG seed for reproducibility.
    max_gold : reject samples whose gold answer set exceeds this (keeps queries
               discriminative); gold lists are still capped by gold_limit on read.
    gold_limit : hard Cypher LIMIT when reading gold answers.
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
        seed: int = 42,
        max_gold: int = 100,
        gold_limit: int = 500,
    ) -> None:
        from neo4j import GraphDatabase  # local import keeps --self-test DB-free

        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.rng = random.Random(seed)
        self.max_gold = max_gold
        self.gold_limit = gold_limit
        self._neg_vocab: List[Dict[str, str]] = []   # global negative-aspect pool
        self._seen_sigs: set = set()
        logger.info("QASyntheticGenerator connected to %s (db=%s)", uri, database)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        self._driver.close()
        logger.info("Neo4j driver closed.")

    def _run(self, query: str, **params) -> List[Dict[str, Any]]:
        with self._driver.session(database=self.database) as session:
            return [dict(r) for r in session.run(query, **params)]

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def load_negative_vocab(self, limit: int = 300) -> None:
        rows = self._run(
            """
            MATCH (:Product)-[:HAS_NEGATIVE_ASPECT]->(a:Aspect)
            RETURN DISTINCT a.aspect_id AS id, a.name AS name
            LIMIT $limit
            """,
            limit=limit,
        )
        self._neg_vocab = [r for r in rows if r.get("id") and r.get("name")]
        logger.info("Loaded %d global negative aspects.", len(self._neg_vocab))

    def sample_seed_products(self, n: int) -> List[str]:
        """
        Random sample of products that have at least a category and one positive
        aspect (the minimum structure required to build a meaningful query).
        Uses rand() ordering with LIMIT — does not scan/return the full graph.
        """
        rows = self._run(
            """
            MATCH (p:Product)-[:BELONGS_TO_CATEGORY]->(:Category)
            WHERE p.price IS NOT NULL
            MATCH (p)-[:HAS_POSITIVE_ASPECT]->(:Aspect)
            WITH DISTINCT p, rand() AS r
            ORDER BY r
            LIMIT $n
            RETURN p.parent_asin AS asin
            """,
            n=n,
        )
        asins = [r["asin"] for r in rows if r.get("asin")]
        logger.info("Sampled %d seed products.", len(asins))
        return asins

    def expand_graph_paths(self, asin: str) -> Optional[Expansion]:
        """Read the true 1-hop neighbourhood of a seed product."""
        rows = self._run(
            """
            MATCH (p:Product {parent_asin: $asin})
            OPTIONAL MATCH (p)-[:BELONGS_TO_CATEGORY]->(c:Category)
            OPTIONAL MATCH (p)-[:HAS_BRAND]->(b:Brand)
            OPTIONAL MATCH (p)-[pa:HAS_POSITIVE_ASPECT]->(ap:Aspect)
            OPTIONAL MATCH (p)-[na:HAS_NEGATIVE_ASPECT]->(an:Aspect)
            OPTIONAL MATCH (p)-[:HAS_FEATURE]->(f:Feature)
            OPTIONAL MATCH (p)-[:HAS_DETAIL]->(d:Detail)
            RETURN p.parent_asin AS asin, p.title AS title, p.price AS price,
                   c.category_id AS cat_id, c.name AS cat_name,
                   b.brand_id AS brand_id, b.name AS brand_name,
                   collect(DISTINCT {id: ap.aspect_id, name: ap.name, weight: pa.weight}) AS pos,
                   collect(DISTINCT {id: an.aspect_id, name: an.name, weight: na.weight}) AS neg,
                   collect(DISTINCT {id: f.feature_id, text: f.text})[..8] AS feats,
                   collect(DISTINCT {id: d.detail_id, key: d.key, value: d.value})[..8] AS details
            """,
            asin=asin,
        )
        if not rows:
            return None
        r = rows[0]

        def _clean(lst, *keys):
            out = []
            for item in lst or []:
                if item and item.get(keys[0]):
                    out.append(item)
            return out

        return Expansion(
            asin=r["asin"],
            title=r.get("title") or "",
            price=r.get("price"),
            cat_id=r.get("cat_id"),
            cat_name=r.get("cat_name"),
            brand_id=r.get("brand_id"),
            brand_name=r.get("brand_name"),
            pos_aspects=_clean(r.get("pos"), "id"),
            neg_aspects=_clean(r.get("neg"), "id"),
            features=_clean(r.get("feats"), "id"),
            details=_clean(r.get("details"), "id"),
        )

    # ------------------------------------------------------------------
    # Query construction (template-based + randomisation)
    # ------------------------------------------------------------------
    def build_query_from_path(
        self, exp: Expansion, qa_type: str
    ) -> Optional[Tuple[str, ConstraintSpec, Dict[str, Any], List[str]]]:
        """
        Build (query_text, ConstraintSpec, public_constraints, reasoning_path).
        Returns None if the expansion lacks the structure for `qa_type`.
        """
        rng = self.rng
        if not exp.cat_name or not exp.cat_id:
            return None

        cat = clean_category(exp.cat_name)
        spec = ConstraintSpec(category_id=exp.cat_id)
        public: Dict[str, Any] = {
            "category": exp.cat_name,
            "brand": None,
            "price_max": None,
            "positive_aspects": [],
            "negative_aspects": [],
        }
        path: List[str] = [f"Product -[BELONGS_TO_CATEGORY]-> Category({exp.cat_name})"]

        pos = list(exp.pos_aspects)
        rng.shuffle(pos)

        # -- SIMPLE: category + 1 positive aspect, OR feature + category --------
        if qa_type == "simple":
            use_feature = bool(exp.features) and rng.random() < 0.4
            if use_feature:
                feat = rng.choice(exp.features)
                snippet = clean_feature_snippet(feat["text"])
                if not snippet:
                    return None
                spec.feature_ids = [feat["id"]]
                public["features"] = [snippet]
                path.append(f"Product -[HAS_FEATURE]-> Feature({snippet})")
                query = f"{rng.choice(SIMPLE_LEADS)}{cat} with {snippet}"
            else:
                if not pos:
                    return None
                a = pos[0]
                spec.pos_aspect_ids = [a["id"]]
                public["positive_aspects"] = [a["name"]]
                path.append(f"Product -[HAS_POSITIVE_ASPECT]-> Aspect({a['name']})")
                query = (
                    f"{rng.choice(SIMPLE_LEADS)}{cat} "
                    f"{rng.choice(POS_CONNECTORS)} {positive_phrase(a['name'], rng)}"
                )

        # -- MULTI_HOP: category + 2 pos aspects, or brand+category+pos ----------
        elif qa_type == "multi_hop":
            variant = rng.random()
            if variant < 0.5 and len(pos) >= 2:
                a1, a2 = pos[0], pos[1]
                spec.pos_aspect_ids = [a1["id"], a2["id"]]
                public["positive_aspects"] = [a1["name"], a2["name"]]
                path += [
                    f"Product -[HAS_POSITIVE_ASPECT]-> Aspect({a1['name']})",
                    f"Product -[HAS_POSITIVE_ASPECT]-> Aspect({a2['name']})",
                ]
                phrases = [positive_phrase(a1["name"], rng), positive_phrase(a2["name"], rng)]
                rng.shuffle(phrases)
                query = f"{cat} {rng.choice(POS_CONNECTORS)} {phrases[0]} and {phrases[1]}"
            elif exp.brand_name and exp.brand_id and pos:
                a = pos[0]
                spec.brand_id = exp.brand_id
                spec.pos_aspect_ids = [a["id"]]
                public["brand"] = exp.brand_name
                public["positive_aspects"] = [a["name"]]
                path += [
                    f"Product -[HAS_BRAND]-> Brand({exp.brand_name})",
                    f"Product -[HAS_POSITIVE_ASPECT]-> Aspect({a['name']})",
                ]
                query = (
                    f"{exp.brand_name} {cat} {rng.choice(POS_CONNECTORS)} "
                    f"{positive_phrase(a['name'], rng)}"
                )
            elif exp.details and pos:
                d = rng.choice(exp.details)
                a = pos[0]
                val = str(d.get("value", "")).strip().lower()
                if not val:
                    return None
                spec.detail_ids = [d["id"]]
                spec.pos_aspect_ids = [a["id"]]
                public["details"] = [{"key": d.get("key"), "value": d.get("value")}]
                public["positive_aspects"] = [a["name"]]
                path += [
                    f"Product -[HAS_DETAIL]-> Detail({d.get('key')}={d.get('value')})",
                    f"Product -[HAS_POSITIVE_ASPECT]-> Aspect({a['name']})",
                ]
                query = f"{val} {cat} with {positive_phrase(a['name'], rng)}"
            else:
                return None

        # -- NEGATIVE: positive aspect(s) + exclude a negative aspect -----------
        elif qa_type == "negative":
            if not pos:
                return None
            owned_neg_ids = {n["id"] for n in exp.neg_aspects}
            candidates = [n for n in self._neg_vocab if n["id"] not in owned_neg_ids]
            if not candidates:
                return None
            neg = rng.choice(candidates)
            a = pos[0]
            spec.pos_aspect_ids = [a["id"]]
            spec.exclude_neg_aspect_ids = [neg["id"]]
            public["positive_aspects"] = [a["name"]]
            public["negative_aspects"] = [neg["name"]]
            path += [
                f"Product -[HAS_POSITIVE_ASPECT]-> Aspect({a['name']})",
                f"Product -[NOT HAS_NEGATIVE_ASPECT]-> Aspect({neg['name']})",
            ]
            query = (
                f"{cat} {rng.choice(POS_CONNECTORS)} {positive_phrase(a['name'], rng)} "
                f"{rng.choice(NEG_CONNECTORS)} {negative_phrase(neg['name'], rng)}"
            )

        # -- CONSTRAINT COMPOSITE: price + category + pos (+brand) (+neg) -------
        elif qa_type == "constraint":
            if not pos or exp.price is None:
                return None
            a = pos[0]
            spec.pos_aspect_ids = [a["id"]]
            public["positive_aspects"] = [a["name"]]
            # price ceiling rounded up above the seed price so the seed qualifies
            ceiling = self._price_ceiling(float(exp.price))
            spec.price_max = ceiling
            public["price_max"] = ceiling
            path.append(f"Product.price <= {ceiling}")
            path.append(f"Product -[HAS_POSITIVE_ASPECT]-> Aspect({a['name']})")

            parts = [f"{cat} under {int(ceiling)} dollars",
                     f"{rng.choice(POS_CONNECTORS)} {positive_phrase(a['name'], rng)}"]
            if exp.brand_name and exp.brand_id and rng.random() < 0.5:
                spec.brand_id = exp.brand_id
                public["brand"] = exp.brand_name
                path.append(f"Product -[HAS_BRAND]-> Brand({exp.brand_name})")
                parts.insert(0, exp.brand_name)
            owned_neg_ids = {n["id"] for n in exp.neg_aspects}
            cand = [n for n in self._neg_vocab if n["id"] not in owned_neg_ids]
            if cand and rng.random() < 0.6:
                neg = rng.choice(cand)
                spec.exclude_neg_aspect_ids = [neg["id"]]
                public["negative_aspects"] = [neg["name"]]
                path.append(f"Product -[NOT HAS_NEGATIVE_ASPECT]-> Aspect({neg['name']})")
                parts.append(f"{rng.choice(NEG_CONNECTORS)} {negative_phrase(neg['name'], rng)}")
            query = " ".join(parts)
        else:
            raise ValueError(f"Unknown qa_type: {qa_type}")

        query = " ".join(query.split()).strip()
        if len(query) < 5:
            return None
        return query, spec, public, path

    @staticmethod
    def _price_ceiling(price: float) -> float:
        for step in (50, 100, 200, 300, 500, 1000, 2000):
            if price <= step:
                return float(step)
        return float(int(price / 500 + 1) * 500)

    # ------------------------------------------------------------------
    # Gold answer recovery (exact, id-based)
    # ------------------------------------------------------------------
    def extract_gold_answers(self, spec: ConstraintSpec) -> List[str]:
        """Recover ALL products satisfying the constraint spec (exact match)."""
        matches: List[str] = ["MATCH (p:Product)"]
        wheres: List[str] = []
        params: Dict[str, Any] = {"limit": self.gold_limit}

        if spec.category_id:
            matches.append(
                "MATCH (p)-[:BELONGS_TO_CATEGORY]->(:Category {category_id: $cat_id})"
            )
            params["cat_id"] = spec.category_id
        if spec.brand_id:
            matches.append("MATCH (p)-[:HAS_BRAND]->(:Brand {brand_id: $brand_id})")
            params["brand_id"] = spec.brand_id
        for i, aid in enumerate(spec.pos_aspect_ids):
            key = f"pos{i}"
            matches.append(
                f"MATCH (p)-[:HAS_POSITIVE_ASPECT]->(:Aspect {{aspect_id: ${key}}})"
            )
            params[key] = aid
        for i, fid in enumerate(spec.feature_ids):
            key = f"feat{i}"
            matches.append(f"MATCH (p)-[:HAS_FEATURE]->(:Feature {{feature_id: ${key}}})")
            params[key] = fid
        for i, did in enumerate(spec.detail_ids):
            key = f"det{i}"
            matches.append(f"MATCH (p)-[:HAS_DETAIL]->(:Detail {{detail_id: ${key}}})")
            params[key] = did
        if spec.price_max is not None:
            wheres.append("p.price <= $price_max")
            params["price_max"] = spec.price_max
        for i, nid in enumerate(spec.exclude_neg_aspect_ids):
            key = f"neg{i}"
            wheres.append(
                f"NOT (p)-[:HAS_NEGATIVE_ASPECT]->(:Aspect {{aspect_id: ${key}}})"
            )
            params[key] = nid

        query = "\n".join(matches)
        if wheres:
            query += "\nWHERE " + " AND ".join(wheres)
        query += "\nRETURN DISTINCT p.parent_asin AS asin LIMIT $limit"

        rows = self._run(query, **params)
        return [r["asin"] for r in rows if r.get("asin")]

    # ------------------------------------------------------------------
    # Dedup
    # ------------------------------------------------------------------
    def _signature(self, qa_type: str, public: Dict[str, Any]) -> str:
        payload = json.dumps(
            {
                "t": qa_type,
                "c": public.get("category"),
                "b": public.get("brand"),
                "p": public.get("price_max"),
                "pa": sorted(public.get("positive_aspects", [])),
                "na": sorted(public.get("negative_aspects", [])),
                "f": sorted(public.get("features", [])),
            },
            sort_keys=True,
        )
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def generate(self, total: int, max_retries_per_type: int = 40) -> List[QASample]:
        if not self._neg_vocab:
            self.load_negative_vocab()

        targets = {t: int(round(total * r)) for t, r in TYPE_RATIOS.items()}
        # fix rounding drift
        drift = total - sum(targets.values())
        targets["multi_hop"] += drift
        logger.info("Target distribution: %s", targets)

        # Oversample a seed pool (re-used across all types).
        pool_size = max(total * 3, 500)
        seed_pool = self.sample_seed_products(pool_size)
        if not seed_pool:
            raise RuntimeError("No seed products sampled — is the KG populated?")

        # Cache expansions so we don't re-query the same product.
        exp_cache: Dict[str, Optional[Expansion]] = {}

        def get_exp(asin: str) -> Optional[Expansion]:
            if asin not in exp_cache:
                exp_cache[asin] = self.expand_graph_paths(asin)
            return exp_cache[asin]

        samples: List[QASample] = []
        counter = 0

        for qa_type, need in targets.items():
            made = 0
            attempts = 0
            max_attempts = need * max_retries_per_type + 50
            while made < need and attempts < max_attempts:
                attempts += 1
                asin = self.rng.choice(seed_pool)
                exp = get_exp(asin)
                if exp is None:
                    continue
                built = self.build_query_from_path(exp, qa_type)
                if built is None:
                    continue
                query, spec, public, path = built

                sig = self._signature(qa_type, public)
                if sig in self._seen_sigs:
                    continue

                gold = self.extract_gold_answers(spec)
                if not gold:
                    continue
                if len(gold) > self.max_gold and qa_type != "simple":
                    # too broad → not discriminative for multi-hop/negative/constraint
                    continue

                self._seen_sigs.add(sig)
                counter += 1
                samples.append(
                    QASample(
                        qid=f"q_{counter:06d}",
                        query=query,
                        type=qa_type,
                        gold_answers=gold,
                        constraints=public,
                        reasoning_path=path,
                    )
                )
                made += 1
                if made % 50 == 0:
                    logger.info("[%s] %d/%d generated", qa_type, made, need)

            logger.info("[%s] DONE %d/%d (attempts=%d)", qa_type, made, need, attempts)
            if made < need:
                logger.warning(
                    "[%s] under target: %d/%d — KG may lack enough %s structure.",
                    qa_type, made, need, qa_type,
                )

        self.rng.shuffle(samples)
        logger.info("Total generated: %d samples.", len(samples))
        return samples


# ===========================================================================
# IO
# ===========================================================================

def save_jsonl(samples: List[QASample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(s.to_json() + "\n")
    logger.info("Wrote %d samples → %s", len(samples), path)


def split_train_test(
    samples: List[QASample], train_ratio: float
) -> Tuple[List[QASample], List[QASample]]:
    n_train = int(len(samples) * train_ratio)
    return samples[:n_train], samples[n_train:]


# ===========================================================================
# Self-test (no DB) — validates templating + gold-query construction
# ===========================================================================

def _self_test() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    gen = QASyntheticGenerator.__new__(QASyntheticGenerator)  # bypass __init__/DB
    gen.rng = random.Random(0)
    gen.max_gold = 100
    gen.gold_limit = 500
    gen._seen_sigs = set()
    gen._neg_vocab = [
        {"id": "asp_weight", "name": "weight"},
        {"id": "asp_heat", "name": "overheating"},
        {"id": "asp_price", "name": "price"},
    ]
    exp = Expansion(
        asin="B07TEST", title="Test Laptop", price=480.0,
        cat_id="cat_laptop", cat_name="Laptops",
        brand_id="brand_acme", brand_name="Acme",
        pos_aspects=[{"id": "a_bat", "name": "battery", "weight": 9},
                     {"id": "a_perf", "name": "performance", "weight": 7},
                     {"id": "a_cool", "name": "cooling", "weight": 5}],
        neg_aspects=[{"id": "a_noise", "name": "noise", "weight": 2}],
        features=[{"id": "f1", "text": "Backlit keyboard with numeric keypad and long life"}],
        details=[{"id": "d1", "key": "Color", "value": "Black"}],
    )
    for t in ("simple", "multi_hop", "negative", "constraint"):
        for _ in range(3):
            built = gen.build_query_from_path(exp, t)
            if built:
                q, spec, public, path = built
                print(f"[{t:10s}] {q}")
                print(f"             constraints={public}")
                print(f"             spec.pos={spec.pos_aspect_ids} "
                      f"neg={spec.exclude_neg_aspect_ids} price={spec.price_max} "
                      f"feat={spec.feature_ids} detail={spec.detail_ids}")
    print("\nSelf-test OK: templating and constraint-spec construction work.")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic QA dataset generator from Neo4j KG")
    parser.add_argument("--total", type=int, default=3000, help="Total samples (2000–5000)")
    parser.add_argument("--out-dir", type=str, default="data/processed")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-gold", type=int, default=100)
    parser.add_argument("--gold-limit", type=int, default=500)
    parser.add_argument("--uri", type=str, default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--user", type=str, default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--password", type=str, default=os.getenv("NEO4J_PASSWORD", "chauduong"))
    parser.add_argument("--database", type=str, default=os.getenv("NEO4J_DATABASE", "neo4j"))
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--self-test", action="store_true", help="Run DB-free templating test")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    gen = QASyntheticGenerator(
        uri=args.uri, user=args.user, password=args.password, database=args.database,
        seed=args.seed, max_gold=args.max_gold, gold_limit=args.gold_limit,
    )
    try:
        samples = gen.generate(total=args.total)
    finally:
        gen.close()

    train, test = split_train_test(samples, args.train_ratio)
    out_dir = Path(args.out_dir)
    save_jsonl(train, out_dir / "qa_train.jsonl")
    save_jsonl(test, out_dir / "qa_test.jsonl")

    # Distribution report
    from collections import Counter
    dist = Counter(s.type for s in samples)
    logger.info("Final distribution: %s", dict(dist))
    logger.info("Train=%d  Test=%d  Total=%d", len(train), len(test), len(samples))


if __name__ == "__main__":
    main()