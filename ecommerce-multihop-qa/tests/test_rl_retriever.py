from src.retrieval.rl_retriever import RLRetriever
from src.rl.kg_env import KGEnvironment

env = KGEnvironment(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password"
)

agent = RLRetriever()

query = """
wireless gaming headset
with good sound quality
and no heating issue
"""

env.reset()

env.set_start_node("Gaming Headset")

neighbors = env.get_neighbors()

print("\nNeighbors:\n")

for n in neighbors:
    print(n)