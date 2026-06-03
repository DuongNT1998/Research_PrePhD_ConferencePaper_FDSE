"""
src/kg/neo4j_connector.py

Neo4j Knowledge Graph Connector.

Provides all graph queries needed by the RL environment:
- get_node_by_id()     — fetch node properties by unique ID
- get_neighbours()     — dynamic neighbourhood for action space
- search_products()    — keyword search for anchor resolution
- get_node_count()     — statistics

Optimised for Neo4j bolt protocol with connection pooling.
All queries use parameterised Cypher (no string injection).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from neo4j import GraphDatabase, Driver

from src.config.settings import Config, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class KGConnector:
    """
    Thread-safe Neo4j connector for KG traversal.

    Parameters
    ----------
    config : Config
    """

    def __init__(self, config: Config = DEFAULT_CONFIG) -> None:
        self.config = config
        nc = config.neo4j
        self._driver: Driver = GraphDatabase.driver(
            nc.uri,
            auth=(nc.user, nc.password),
            max_connection_lifetime=nc.max_connection_lifetime,
            max_connection_pool_size=nc.max_connection_pool_size,
            connection_acquisition_timeout=nc.connection_acquisition_timeout,
        )
        logger.info("KGConnector: connected to Neo4j at %s", nc.uri)

    def close(self) -> None:
        self._driver.close()
        logger.info("KGConnector: driver closed.")

    # ------------------------------------------------------------------
    # Node access
    # ------------------------------------------------------------------

    def get_node_by_id(
        self, node_id: str
    ) -> Tuple[Dict[str, Any], str]:
        """
        Fetch node properties and type by its unique identifier.

        Tries Product first (most common), then other types.
        Returns (properties_dict, node_type_string).
        """
        # Try Product
        product = self._get_product(node_id)
        if product:
            return product, "Product"

        # Try other node types via element ID or unique properties
        for label, id_field in [
            ("Brand", "brand_id"),
            ("Category", "category_id"),
            ("Feature", "feature_id"),
            ("Aspect", "aspect_id"),
            ("Detail", "detail_id"),
        ]:
            result = self._get_node_by_label_and_id(label, id_field, node_id)
            if result:
                return result, label

        logger.warning("Node not found: %s — returning empty.", node_id)
        return {}, "Product"

    def _get_product(self, parent_asin: str) -> Optional[Dict[str, Any]]:
        query = """
        MATCH (p:Product {parent_asin: $asin})
        RETURN p.parent_asin AS parent_asin,
               p.title AS title,
               p.average_rating AS average_rating,
               p.rating_number AS rating_number,
               p.price AS price,
               p.main_category AS main_category
        LIMIT 1
        """
        with self._driver.session(database=self.config.neo4j.database) as session:
            result = session.run(query, asin=parent_asin)
            record = result.single()
            if record:
                return dict(record)
        return None

    def _get_node_by_label_and_id(
        self, label: str, id_field: str, id_value: str
    ) -> Optional[Dict[str, Any]]:
        query = f"""
        MATCH (n:{label} {{{id_field}: $id_value}})
        RETURN properties(n) AS props
        LIMIT 1
        """
        with self._driver.session(database=self.config.neo4j.database) as session:
            result = session.run(query, id_value=id_value)
            record = result.single()
            if record:
                return dict(record["props"])
        return None

    # ------------------------------------------------------------------
    # Dynamic neighbourhood (core for action space)
    # ------------------------------------------------------------------

    def get_neighbours(
        self,
        node_id: str,
        node_type: str,
        max_results: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Get all neighbours of a node across all relation types.

        Returns list of dicts:
        {
            target_id: str,
            target_type: str,
            relation_type: str,
            weight: float,
            target_props: dict,
        }

        Products have outgoing relations; other types may also have
        incoming PRODUCT relations (traversal is bidirectional at
        the Product level only).
        """
        if node_type == "Product":
            return self._get_product_neighbours(node_id, max_results)
        else:
            return self._get_non_product_neighbours(node_id, node_type, max_results)

    def _get_product_neighbours(
        self, parent_asin: str, max_results: int
    ) -> List[Dict[str, Any]]:
        """All outgoing relations from a Product node."""
        query = """
        MATCH (p:Product {parent_asin: $asin})-[r]->(n)
        RETURN
            CASE
                WHEN n:Brand     THEN n.brand_id
                WHEN n:Category  THEN n.category_id
                WHEN n:Feature   THEN n.feature_id
                WHEN n:Aspect    THEN n.aspect_id
                WHEN n:Detail    THEN n.detail_id
                ELSE n.parent_asin
            END AS target_id,
            labels(n)[0]          AS target_type,
            type(r)               AS relation_type,
            coalesce(r.weight, 1) AS weight,
            properties(n)         AS target_props
        LIMIT $limit
        """
        with self._driver.session(database=self.config.neo4j.database) as session:
            result = session.run(query, asin=parent_asin, limit=max_results)
            return [dict(record) for record in result]

    def _get_non_product_neighbours(
        self, node_id: str, node_type: str, max_results: int
    ) -> List[Dict[str, Any]]:
        """
        For non-Product nodes: find co-connected Products (reverse traversal).
        Allows the policy to "jump back" to related products via shared attributes.
        """
        id_field = self._node_id_field(node_type)
        query = f"""
        MATCH (n:{node_type} {{{id_field}: $node_id}})<-[r]-(p:Product)
        RETURN
            p.parent_asin     AS target_id,
            'Product'         AS target_type,
            type(r)           AS relation_type,
            coalesce(r.weight, 1) AS weight,
            properties(p)     AS target_props
        LIMIT $limit
        """
        with self._driver.session(database=self.config.neo4j.database) as session:
            result = session.run(query, node_id=node_id, limit=max_results)
            return [dict(record) for record in result]

    # ------------------------------------------------------------------
    # Search (anchor resolution)
    # ------------------------------------------------------------------

    def search_products(
        self,
        keyword: str,
        limit: int = 20,
        min_rating: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Search products by title keyword (case-insensitive CONTAINS).

        Returns list of product property dicts.
        """
        query = """
        MATCH (p:Product)
        WHERE $keyword = ''
           OR toLower(p.title) CONTAINS toLower($keyword)
        RETURN p.parent_asin    AS parent_asin,
               p.title          AS title,
               p.average_rating AS average_rating,
               p.rating_number  AS rating_number,
               p.price          AS price,
               p.main_category  AS main_category
        ORDER BY coalesce(p.rating_number, 0) DESC
        LIMIT $limit
        """
        with self._driver.session(database=self.config.neo4j.database) as session:
            result = session.run(
                query,
                keyword=keyword,
                limit=limit,
            )
            return [dict(record) for record in result]

    def search_products_by_category(
        self, category_name: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        query = """
        MATCH (p:Product)-[:BELONGS_TO_CATEGORY]->(c:Category)
        WHERE toLower(c.name) CONTAINS toLower($cat)
        RETURN p.parent_asin    AS parent_asin,
               p.title          AS title,
               p.average_rating AS average_rating,
               p.price          AS price
        ORDER BY p.average_rating DESC
        LIMIT $limit
        """
        with self._driver.session(database=self.config.neo4j.database) as session:
            result = session.run(query, cat=category_name, limit=limit)
            return [dict(record) for record in result]

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_node_count(self, label: Optional[str] = None) -> int:
        if label:
            q = f"MATCH (n:{label}) RETURN count(n) AS cnt"
        else:
            q = "MATCH (n) RETURN count(n) AS cnt"
        with self._driver.session(database=self.config.neo4j.database) as session:
            result = session.run(q)
            record = result.single()
            return record["cnt"] if record else 0

    def get_relation_count(self, rel_type: Optional[str] = None) -> int:
        if rel_type:
            q = f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS cnt"
        else:
            q = "MATCH ()-[r]->() RETURN count(r) AS cnt"
        with self._driver.session(database=self.config.neo4j.database) as session:
            result = session.run(q)
            record = result.single()
            return record["cnt"] if record else 0

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _node_id_field(node_type: str) -> str:
        mapping = {
            "Product": "parent_asin",
            "Brand": "brand_id",
            "Category": "category_id",
            "Feature": "feature_id",
            "Aspect": "aspect_id",
            "Detail": "detail_id",
        }
        return mapping.get(node_type, "parent_asin")

    def ping(self) -> bool:
        """Check Neo4j connectivity."""
        try:
            with self._driver.session(database=self.config.neo4j.database) as session:
                session.run("RETURN 1")
            return True
        except Exception as exc:
            logger.error("Neo4j ping failed: %s", exc)
            return False