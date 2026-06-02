"""
tests/test_neo4j_connection.py

Integration tests for KGConnector.
These tests REQUIRE a live Neo4j instance with the AmazonReview23 KG loaded.

Skip with: pytest tests/test_neo4j_connection.py --ignore-glob="*neo4j*"
Or set env var: NEO4J_SKIP_TESTS=1
"""

from __future__ import annotations

import os
import sys
import unittest

# Thêm dòng này để Python tìm thấy thư mục 'src'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



SKIP_NEO4J = os.environ.get("NEO4J_SKIP_TESTS", "0") == "1"
SKIP_MSG = "NEO4J_SKIP_TESTS=1 — skipping Neo4j integration tests"


@unittest.skipIf(SKIP_NEO4J, SKIP_MSG)
class TestKGConnectorIntegration(unittest.TestCase):
    """Integration tests — requires live Neo4j."""

    @classmethod
    def setUpClass(cls):
        from src.config.settings import DEFAULT_CONFIG
        from src.kg.neo4j_connector import KGConnector
        cls.config = DEFAULT_CONFIG
        try:
            cls.connector = KGConnector(cls.config)
            if not cls.connector.ping():
                raise ConnectionError("Cannot ping Neo4j")
        except Exception as e:
            raise unittest.SkipTest(f"Neo4j not available: {e}")

    @classmethod
    def tearDownClass(cls):
        cls.connector.close()

    # ------------------------------------------------------------------

    def test_ping(self):
        self.assertTrue(self.connector.ping())

    def test_product_count_positive(self):
        count = self.connector.get_node_count("Product")
        self.assertGreater(count, 0, "Should have at least 1 Product node")

    def test_get_node_count_all(self):
        total = self.connector.get_node_count()
        self.assertGreater(total, 0)

    def test_search_products_returns_list(self):
        results = self.connector.search_products(keyword="", limit=5)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        # Each result should have parent_asin
        for r in results:
            self.assertIn("parent_asin", r)

    def test_search_products_keyword(self):
        results = self.connector.search_products(keyword="laptop", limit=10)
        self.assertIsInstance(results, list)
        # Should return products (may be 0 if no laptops in KG)

    def test_get_product_by_asin(self):
        # Get any product asin first
        products = self.connector.search_products(keyword="", limit=1)
        if not products:
            self.skipTest("No products in KG")
        asin = products[0]["parent_asin"]
        props, node_type = self.connector.get_node_by_id(asin)
        self.assertEqual(node_type, "Product")
        self.assertIn("parent_asin", props)
        self.assertEqual(props["parent_asin"], asin)

    def test_get_neighbours_returns_list(self):
        products = self.connector.search_products(keyword="", limit=1)
        if not products:
            self.skipTest("No products in KG")
        asin = products[0]["parent_asin"]
        neighbours = self.connector.get_neighbours(asin, "Product", max_results=10)
        self.assertIsInstance(neighbours, list)
        # Each neighbour has required keys
        for nb in neighbours:
            self.assertIn("target_id", nb)
            self.assertIn("target_type", nb)
            self.assertIn("relation_type", nb)

    def test_neighbour_relation_types_valid(self):
        from src.config.settings import DEFAULT_CONFIG
        valid_relations = set(DEFAULT_CONFIG.environment.relation_types)
        products = self.connector.search_products(keyword="", limit=1)
        if not products:
            self.skipTest("No products in KG")
        asin = products[0]["parent_asin"]
        neighbours = self.connector.get_neighbours(asin, "Product", max_results=30)
        for nb in neighbours:
            rel = nb["relation_type"]
            self.assertIn(rel, valid_relations,
                          f"Unexpected relation type: {rel}")

    def test_relation_count_positive(self):
        count = self.connector.get_relation_count()
        self.assertGreater(count, 0)

    def test_aspect_relations_exist(self):
        pos_count = self.connector.get_relation_count("HAS_POSITIVE_ASPECT")
        neg_count = self.connector.get_relation_count("HAS_NEGATIVE_ASPECT")
        # Both should exist given TASK 1 completed
        self.assertGreater(pos_count + neg_count, 0,
                           "Aspect relations should be present after TASK 1")


if __name__ == "__main__":
    unittest.main(verbosity=2)