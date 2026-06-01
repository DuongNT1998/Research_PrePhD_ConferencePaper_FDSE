import os
import sys

# Lấy đường dẫn của thư mục gốc (ecommerce-multihop-qa) và thêm vào sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)


from src.rl.kg_env import KGEnvironment
from src.retrieval.adaptive_retriever import AdaptiveRetriever
from src.retrieval.stopping import AdaptiveStopping



query = "plannet stickers with good color but not heavy"

start_node = "B0BPLX8B2K"

env = KGEnvironment()

state = env.reset(
    query=query,
    start_node=start_node
)

retriever = AdaptiveRetriever(env)

stopper = AdaptiveStopping()

done = False

while not done:

    print("\nCURRENT NODE:")
    print(state.current_node)

    actions = retriever.get_possible_actions()

    print("\nPOSSIBLE ACTIONS:")

    for idx, action in enumerate(actions):

        print(idx, action)

    if len(actions) == 0:
        break

    selected_action = actions[0]

    state, reward, done = env.step(selected_action)

    if stopper.should_stop(state):
        done = True

print("\nFINAL STATE:")
print(state.to_dict())