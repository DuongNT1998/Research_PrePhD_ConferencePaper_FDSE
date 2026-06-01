import sys
import os

# Tự động tìm và thêm thư mục gốc của project vào hệ thống tìm kiếm của Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.kg.neo4j_connector import Neo4jConnector


from src.config.settings import *
from src.kg.neo4j_connector import Neo4jConnector


connector = Neo4jConnector(
    uri=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD
)

query = """
MATCH (p:Product)
RETURN p.title AS title
LIMIT 5
"""

results = connector.run_query(query)

for item in results:
    print(item)

connector.close()