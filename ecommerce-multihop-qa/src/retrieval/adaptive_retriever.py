from typing import List, Dict


class AdaptiveRetriever:

    def __init__(self, env):

        self.env = env

    def get_possible_actions(self) -> List[Dict]:

        neighbors = self.env.get_neighbors()

        actions = []

        for item in neighbors:

            action = {
                "relation": item["relation"],
                "target_type": item["node_type"],
                "target": item.get("asin") or item.get("name")
            }

            actions.append(action)

        return actions