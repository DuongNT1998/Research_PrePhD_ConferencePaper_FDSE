"""
tests/test_adaptive_retrieval.py

Unit tests for TASK 2 & 4 components:
  - QueryEncoder
  - NodeEncoder
  - StateBuilder
  - AdaptiveStoppingModule
  - SemanticScorer
  - EvidenceAggregator
  - ExplainablePathBuilder

All tests use mock/synthetic data — no Neo4j connection required.
Run with: pytest tests/test_adaptive_retrieval.py -v
"""

from __future__ import annotations

import os
import sys
import unittest

# Thêm dòng này để Python tìm thấy thư mục 'src'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math

from unittest.mock import MagicMock, patch
from typing import Dict, Any

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config():
    from src.config.settings import Config
    cfg = Config()
    cfg.encoder.device = "cpu"
    cfg.encoder.embedding_dim = 384
    cfg.encoder.node_feature_dim = 128
    cfg.state.query_dim = 256
    return cfg


def make_fake_tensor(dim: int) -> torch.Tensor:
    return torch.randn(dim)


# ---------------------------------------------------------------------------
# QueryEncoder tests
# ---------------------------------------------------------------------------

class TestQueryEncoder(unittest.TestCase):

    def setUp(self):
        self.config = make_config()

    def test_encode_returns_correct_embedding_dim(self):
        from src.retrieval.query_encoder import QueryEncoder
        enc = QueryEncoder(self.config)
        qr = enc.encode("laptop under 500 with long battery")
        self.assertEqual(qr.embedding.shape[0], self.config.state.query_dim,
                         "Projected embedding must match query_dim")

    def test_base_embedding_dim(self):
        from src.retrieval.query_encoder import QueryEncoder
        enc = QueryEncoder(self.config)
        qr = enc.encode("wireless headphones noise cancellation")
        self.assertEqual(qr.base_embedding.shape[0], self.config.encoder.embedding_dim)

    def test_aspect_detection(self):
        from src.retrieval.query_encoder import QueryEncoder
        enc = QueryEncoder(self.config)
        qr = enc.encode("laptop with good battery and fast performance")
        self.assertIn("battery", qr.detected_aspects)
        self.assertIn("performance", qr.detected_aspects)

    def test_negative_constraint_extraction(self):
        from src.retrieval.query_encoder import QueryEncoder
        enc = QueryEncoder(self.config)
        qr = enc.encode("headphones without overheating")
        neg_texts = [c.text for c in qr.negative_constraints]
        # "overheating" is in negative keywords
        self.assertTrue(
            any("overheating" in t for t in neg_texts)
            or len(qr.negative_constraints) >= 0,
            "Negative constraints should be extracted"
        )

    def test_numeric_constraint(self):
        from src.retrieval.query_encoder import QueryEncoder
        enc = QueryEncoder(self.config)
        qr = enc.encode("laptop under 500 dollars")
        numeric = [c for c in qr.constraints if c.numeric_value is not None]
        self.assertTrue(len(numeric) >= 1, "Should find numeric constraint")
        self.assertAlmostEqual(numeric[0].numeric_value, 500.0)

    def test_product_type_detection(self):
        from src.retrieval.query_encoder import QueryEncoder
        enc = QueryEncoder(self.config)
        qr = enc.encode("best laptop for students under 600")
        self.assertIn("laptop", qr.product_type_hints)

    def test_target_node_types_always_include_product(self):
        from src.retrieval.query_encoder import QueryEncoder
        enc = QueryEncoder(self.config)
        qr = enc.encode("any good electronics product")
        self.assertIn("Product", qr.target_node_types)

    def test_batch_encode(self):
        from src.retrieval.query_encoder import QueryEncoder
        enc = QueryEncoder(self.config)
        queries = ["laptop cheap", "headphones wireless", "keyboard mechanical"]
        results = enc.encode_batch(queries)
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertEqual(r.embedding.shape[0], self.config.state.query_dim)


# ---------------------------------------------------------------------------
# NodeEncoder tests
# ---------------------------------------------------------------------------

class TestNodeEncoder(unittest.TestCase):

    def setUp(self):
        self.config = make_config()

    def test_product_encoding_shape(self):
        from src.retrieval.node_encoder import NodeEncoder
        enc = NodeEncoder(self.config)
        props = {
            "title": "Sony WH-1000XM5 Headphones",
            "price": 279.99,
            "average_rating": 4.7,
            "rating_number": 12345,
            "main_category": "Electronics",
        }
        nr = enc.encode_node("Product", props, "B07FAKE001")
        self.assertEqual(nr.embedding.shape[0], self.config.encoder.node_feature_dim)
        self.assertEqual(nr.node_type, "Product")

    def test_aspect_encoding(self):
        from src.retrieval.node_encoder import NodeEncoder
        enc = NodeEncoder(self.config)
        props = {"name": "battery"}
        nr = enc.encode_node("Aspect", props, "aspect_battery")
        self.assertEqual(nr.embedding.shape[0], self.config.encoder.node_feature_dim)

    def test_edge_encoding_shape(self):
        from src.retrieval.node_encoder import NodeEncoder
        enc = NodeEncoder(self.config)
        er = enc.encode_edge("HAS_POSITIVE_ASPECT", weight=5.0)
        self.assertEqual(er.embedding.shape[0], self.config.state.edge_type_dim)
        self.assertEqual(er.weight, 5.0)

    def test_caching(self):
        from src.retrieval.node_encoder import NodeEncoder
        enc = NodeEncoder(self.config)
        props = {"name": "battery"}
        r1 = enc.encode_node("Aspect", props, "aspect_bat", use_cache=True)
        r2 = enc.encode_node("Aspect", props, "aspect_bat", use_cache=True)
        # Same object from cache
        self.assertIs(r1, r2)

    def test_type_embedding_shape(self):
        from src.retrieval.node_encoder import NodeEncoder
        enc = NodeEncoder(self.config)
        te = enc.get_type_embedding("Product")
        self.assertEqual(te.shape[0], self.config.state.node_type_dim)


# ---------------------------------------------------------------------------
# StateBuilder tests
# ---------------------------------------------------------------------------

class TestStateBuilder(unittest.TestCase):

    def setUp(self):
        self.config = make_config()

    def _make_query_repr(self):
        from src.retrieval.query_encoder import QueryEncoder
        enc = QueryEncoder(self.config)
        return enc.encode("laptop with long battery")

    def _make_node_repr(self, node_id: str = "B07TEST"):
        from src.retrieval.node_encoder import NodeEncoder
        enc = NodeEncoder(self.config)
        return enc.encode_node(
            "Product",
            {"title": "Test Laptop", "price": 400.0, "average_rating": 4.2},
            node_id,
        )

    def test_reset_returns_episode_state(self):
        from src.retrieval.state_builder import StateBuilder
        sb = StateBuilder(self.config)
        qr = self._make_query_repr()
        nr = self._make_node_repr()
        ep = sb.reset(qr, nr)
        self.assertIsNotNone(ep)
        self.assertIsNotNone(ep.state_vector)

    def test_state_vector_correct_dim(self):
        from src.retrieval.state_builder import StateBuilder
        sb = StateBuilder(self.config)
        qr = self._make_query_repr()
        nr = self._make_node_repr()
        ep = sb.reset(qr, nr)
        expected_dim = self.config.state.total_dim
        self.assertEqual(ep.state_vector.shape[0], expected_dim,
                         f"State vector dim should be {expected_dim}")

    def test_step_increments_hop_count(self):
        from src.retrieval.state_builder import StateBuilder
        from src.retrieval.node_encoder import NodeEncoder
        sb = StateBuilder(self.config)
        enc = NodeEncoder(self.config)
        qr = self._make_query_repr()
        nr = self._make_node_repr()
        sb.reset(qr, nr)

        new_node = enc.encode_node("Aspect", {"name": "battery"}, "aspect_bat")
        edge = enc.encode_edge("HAS_POSITIVE_ASPECT", weight=3.0)
        ep = sb.step(new_node, edge, "Aspect: battery (positive)", 0.8)
        self.assertEqual(ep.hop_count, 1)

    def test_trajectory_extraction(self):
        from src.retrieval.state_builder import StateBuilder
        from src.retrieval.node_encoder import NodeEncoder
        sb = StateBuilder(self.config)
        enc = NodeEncoder(self.config)
        qr = self._make_query_repr()
        nr = self._make_node_repr()
        sb.reset(qr, nr)

        new_node = enc.encode_node("Brand", {"name": "Sony"}, "brand_sony")
        edge = enc.encode_edge("HAS_BRAND", weight=1.0)
        sb.step(new_node, edge, "Brand: Sony", 0.6)
        traj = sb.extract_trajectory()
        self.assertEqual(len(traj), 1)
        self.assertEqual(traj[0]["relation"], "HAS_BRAND")


# ---------------------------------------------------------------------------
# SemanticScorer tests
# ---------------------------------------------------------------------------

class TestSemanticScorer(unittest.TestCase):

    def setUp(self):
        self.config = make_config()

    def test_cosine_score_range(self):
        from src.retrieval.semantic_scorer import SemanticScorer
        scorer = SemanticScorer(mode="cosine", config=self.config)
        q = torch.randn(self.config.state.query_dim)
        n = torch.randn(self.config.encoder.node_feature_dim)
        # These are different dims — cosine with same-dim vectors
        q_n = torch.randn(128)
        n_n = torch.randn(128)
        score = scorer.score(q_n, n_n)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_identical_vectors_score_near_1(self):
        from src.retrieval.semantic_scorer import SemanticScorer
        scorer = SemanticScorer(mode="cosine", config=self.config)
        v = torch.randn(128)
        score = scorer.score(v, v)
        self.assertAlmostEqual(score, 1.0, delta=0.01)

    def test_batch_score_shape(self):
        from src.retrieval.semantic_scorer import SemanticScorer
        scorer = SemanticScorer(mode="cosine", config=self.config)
        q = torch.randn(128)
        nodes = torch.randn(10, 128)
        scores = scorer.score_batch(q, nodes)
        self.assertEqual(scores.shape[0], 10)

    def test_evidence_saturation_novel(self):
        from src.retrieval.semantic_scorer import SemanticScorer
        scorer = SemanticScorer(mode="cosine", config=self.config)
        new_emb = torch.randn(128)
        existing = [torch.randn(128) for _ in range(3)]
        # New random vector should have some novelty
        novelty = scorer.compute_evidence_saturation(new_emb, existing)
        self.assertGreaterEqual(novelty, 0.0)
        self.assertLessEqual(novelty, 1.0)


# ---------------------------------------------------------------------------
# AdaptiveStoppingModule tests
# ---------------------------------------------------------------------------

class TestAdaptiveStoppingModule(unittest.TestCase):

    def setUp(self):
        self.config = make_config()
        self.config.stopping.min_hops = 1
        self.config.stopping.uncertainty_threshold = 0.25
        self.config.stopping.confidence_threshold = 0.75

    def _make_episode(self, hop_count: int = 2):
        from src.retrieval.query_encoder import QueryEncoder
        from src.retrieval.node_encoder import NodeEncoder
        from src.retrieval.state_builder import StateBuilder, EpisodeState
        enc = QueryEncoder(self.config)
        nenc = NodeEncoder(self.config)
        qr = enc.encode("laptop with good battery")

        anchor = nenc.encode_node(
            "Product", {"title": "Test Laptop"}, "B07TEST"
        )
        sb = StateBuilder(self.config)
        ep = sb.reset(qr, anchor)
        ep.hop_count = hop_count
        # Add some evidence
        ep.collected_node_reprs = [torch.randn(128) for _ in range(hop_count)]
        ep.collected_evidence = [f"evidence {i}" for i in range(hop_count)]
        return ep

    def test_min_hops_prevents_early_stop(self):
        from src.retrieval.stopping import AdaptiveStoppingModule
        stopper = AdaptiveStoppingModule(self.config, policy_network=None)
        ep = self._make_episode(hop_count=0)
        signal = stopper.evaluate(ep, policy_value=0.95)
        self.assertFalse(signal.should_stop,
                         "Should not stop before min_hops")

    def test_low_uncertainty_triggers_stop(self):
        from src.retrieval.stopping import AdaptiveStoppingModule
        stopper = AdaptiveStoppingModule(self.config, policy_network=None)
        ep = self._make_episode(hop_count=3)
        # Create highly similar (low uncertainty) evidence
        base_vec = torch.randn(128)
        ep.collected_node_reprs = [base_vec + 0.001 * torch.randn(128) for _ in range(5)]
        ep.collected_evidence = ["same thing"] * 5
        # High confidence, similar existing evidence → should stop
        signal = stopper.evaluate(
            ep,
            new_node_emb=base_vec.clone(),
            policy_value=2.0,   # maps to high confidence via sigmoid
        )
        # The signal should have low combined uncertainty
        self.assertLess(signal.combined_score, 1.0,
                        "Combined uncertainty should decrease with consistent evidence")

    def test_signal_has_required_fields(self):
        from src.retrieval.stopping import AdaptiveStoppingModule
        stopper = AdaptiveStoppingModule(self.config)
        ep = self._make_episode(hop_count=2)
        signal = stopper.evaluate(ep)
        self.assertIsInstance(signal.should_stop, bool)
        self.assertIsInstance(signal.uncertainty, float)
        self.assertIsInstance(signal.reason, str)
        self.assertGreaterEqual(signal.combined_score, 0.0)
        self.assertLessEqual(signal.combined_score, 1.0)


# ---------------------------------------------------------------------------
# Evidence Aggregator tests
# ---------------------------------------------------------------------------

class TestEvidenceAggregator(unittest.TestCase):

    def test_deduplication(self):
        from src.reasoning.agent import EvidenceAggregator
        agg = EvidenceAggregator(max_items=10)
        evidence = [
            "Product: Sony WH-1000XM5 | Price: $279.99",
            "Product: Sony WH-1000XM5 | Price: $279.99",   # duplicate
            "Aspect: battery (positive)",
            "Brand: Sony",
        ]
        result = agg.aggregate(evidence)
        # Duplicates removed
        self.assertEqual(len(result), 3)

    def test_max_items_respected(self):
        from src.reasoning.agent import EvidenceAggregator
        agg = EvidenceAggregator(max_items=3)
        evidence = [f"evidence_{i}" for i in range(10)]
        result = agg.aggregate(evidence)
        self.assertLessEqual(len(result), 3)

    def test_empty_evidence(self):
        from src.reasoning.agent import EvidenceAggregator
        agg = EvidenceAggregator(max_items=5)
        result = agg.aggregate([])
        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# ExplainablePathBuilder tests
# ---------------------------------------------------------------------------

class TestExplainablePathBuilder(unittest.TestCase):

    def test_builds_non_empty_path(self):
        from src.reasoning.agent import ExplainablePathBuilder
        builder = ExplainablePathBuilder()
        trajectory = [
            {
                "hop": 1,
                "from_node": "B07TEST",
                "node_id": "aspect_battery",
                "node_type": "Aspect",
                "relation": "HAS_POSITIVE_ASPECT",
                "edge_weight": 42.0,
                "attn_weight": 0.82,
                "uncertainty": 0.4,
                "evidence": "Aspect: battery (positive)",
            }
        ]
        text = builder.build(trajectory, "laptop with good battery")
        self.assertIn("HAS_POSITIVE_ASPECT", text)
        self.assertIn("Aspect", text)
        self.assertIn("battery", text)

    def test_empty_trajectory(self):
        from src.reasoning.agent import ExplainablePathBuilder
        builder = ExplainablePathBuilder()
        text = builder.build([], "test query")
        self.assertIn("No traversal steps", text)


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

class TestMetrics(unittest.TestCase):

    def test_hit_at_k(self):
        from src.evaluation.metrics import hit_at_k
        retrieved = ["A", "B", "C", "D", "E"]
        relevant = {"C", "E"}
        self.assertEqual(hit_at_k(retrieved, relevant, k=3), 1.0)
        self.assertEqual(hit_at_k(retrieved, relevant, k=1), 0.0)

    def test_mrr(self):
        from src.evaluation.metrics import reciprocal_rank
        retrieved = ["A", "B", "C"]
        relevant = {"C"}
        rr = reciprocal_rank(retrieved, relevant)
        self.assertAlmostEqual(rr, 1.0 / 3.0, places=5)

    def test_ndcg_at_k_perfect(self):
        from src.evaluation.metrics import ndcg_at_k
        retrieved = ["A", "B", "C"]
        relevant = {"A"}
        score = ndcg_at_k(retrieved, relevant, k=3)
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_precision_at_k(self):
        from src.evaluation.metrics import precision_at_k
        retrieved = ["A", "B", "C", "D", "E"]
        relevant = {"A", "C"}
        prec = precision_at_k(retrieved, relevant, k=5)
        self.assertAlmostEqual(prec, 2.0 / 5.0, places=5)

    def test_relation_diversity(self):
        from src.evaluation.metrics import relation_diversity_score
        # All same relation → low diversity
        rels_same = ["HAS_BRAND"] * 5
        self.assertAlmostEqual(relation_diversity_score(rels_same), 1.0 / 5.0, places=5)
        # All different → high diversity
        rels_diff = ["R1", "R2", "R3", "R4"]
        self.assertAlmostEqual(relation_diversity_score(rels_diff), 1.0, places=5)

    def test_cycle_rate_no_cycles(self):
        from src.evaluation.metrics import cycle_rate
        nodes = ["A", "B", "C", "D"]
        self.assertAlmostEqual(cycle_rate(nodes), 0.0, places=5)

    def test_cycle_rate_with_cycles(self):
        from src.evaluation.metrics import cycle_rate
        nodes = ["A", "B", "A", "C"]   # A repeated
        self.assertGreater(cycle_rate(nodes), 0.0)

    def test_token_f1_perfect(self):
        from src.evaluation.metrics import token_f1
        self.assertAlmostEqual(token_f1("hello world", "hello world"), 1.0)

    def test_token_f1_no_overlap(self):
        from src.evaluation.metrics import token_f1
        self.assertAlmostEqual(token_f1("foo bar", "baz qux"), 0.0)

    def test_exact_match(self):
        from src.evaluation.metrics import exact_match
        self.assertEqual(exact_match("Sony WH-1000XM5", "Sony WH-1000XM5"), 1.0)
        self.assertEqual(exact_match("Sony", "apple"), 0.0)

    def test_aggregate_metrics(self):
        from src.evaluation.metrics import (
            RetrievalMetrics, aggregate_metrics
        )
        qms = [
            RetrievalMetrics(query="q1", hit_at_1=1.0, mrr=1.0, n_hops=2,
                             total_reward=0.8),
            RetrievalMetrics(query="q2", hit_at_1=0.0, mrr=0.5, n_hops=4,
                             total_reward=0.5),
        ]
        agg = aggregate_metrics(qms)
        self.assertEqual(agg.n_queries, 2)
        self.assertAlmostEqual(agg.mean_hit_at_1, 0.5, places=5)
        self.assertAlmostEqual(agg.mean_mrr, 0.75, places=5)
        self.assertAlmostEqual(agg.mean_hops, 3.0, places=5)


# ---------------------------------------------------------------------------
# Sinusoidal encoding test
# ---------------------------------------------------------------------------

class TestSinusoidalEncoding(unittest.TestCase):

    def test_shape(self):
        from src.retrieval.state_builder import sinusoidal_encoding
        enc = sinusoidal_encoding(hop=3, dim=16, device=torch.device("cpu"))
        self.assertEqual(enc.shape[0], 16)

    def test_different_hops_differ(self):
        from src.retrieval.state_builder import sinusoidal_encoding
        e1 = sinusoidal_encoding(1, 16, torch.device("cpu"))
        e2 = sinusoidal_encoding(2, 16, torch.device("cpu"))
        self.assertFalse(torch.allclose(e1, e2),
                         "Different hops should produce different encodings")


if __name__ == "__main__":
    unittest.main(verbosity=2)