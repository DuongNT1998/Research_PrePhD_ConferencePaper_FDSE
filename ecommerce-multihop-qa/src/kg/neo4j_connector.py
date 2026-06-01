# Chức năng:
# connect database,
# execute cypher,
# return results.

import random # <-- ĐÃ SỬA DÒNG NÀY

from neo4j import GraphDatabase


class Neo4jConnector:

    def __init__(self, uri, username, password):
        self.driver = GraphDatabase.driver(
            uri,
            auth=(username, password)
        )

    def close(self):
        self.driver.close()

    def run_query(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.run(
                query,
                parameters or {}
            )
            return [record.data() for record in result]

    # === THÊM HÀM NÀY VÀO ĐỂ KHỚP VỚI kg_env.py ===
    def query(self, query, parameters=None):
        """Hàm wrapper gọi lại run_query để sửa lỗi AttributeError"""
        return self.run_query(query, parameters)

    def get_random_product(self):

        query = """
        MATCH (p:Product)
        RETURN p
        LIMIT 50
        """

        with self.driver.session() as session:

            result = session.run(query)

            products = []

            for record in result:

                node = dict(record["p"])

                node["_label"] = "Product"
                node["_id"] = node["parent_asin"]

                products.append(node)

            return random.choice(products)


    def get_neighbors(self, node_id):

        query = """
        MATCH (n)-[r]->(m)

        WHERE
            (
                (n:Product AND n.parent_asin = $node_id)
                OR
                (n:Brand AND n.brand_id = $node_id)
                OR
                (n:Category AND n.category_id = $node_id)
                OR
                (n:Feature AND n.feature_id = $node_id)
                OR
                (n:Aspect AND n.aspect_id = $node_id)
                OR
                (n:Detail AND n.detail_id = $node_id)
            )

        RETURN
            type(r) as rel,
            m,
            labels(m) as labels
        """

        with self.driver.session() as session:

            result = session.run(query, node_id=node_id)

            neighbors = []

            for record in result:

                node = dict(record["m"])

                label = record["labels"][0]

                node["_label"] = label

                # =========================
                # Dynamic node id mapping
                # =========================

                if label == "Product":
                    node["_id"] = node["parent_asin"]

                elif label == "Brand":
                    node["_id"] = node["brand_id"]

                elif label == "Category":
                    node["_id"] = node["category_id"]

                elif label == "Feature":
                    node["_id"] = node["feature_id"]

                elif label == "Aspect":
                    node["_id"] = node["aspect_id"]

                elif label == "Detail":
                    node["_id"] = node["detail_id"]

                neighbors.append({
                    "relation": record["rel"],
                    "node": node
                })

            return neighbors