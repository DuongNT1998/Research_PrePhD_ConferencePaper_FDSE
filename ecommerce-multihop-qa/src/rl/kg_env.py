class KGEnvironment:

    def __init__(self, kg_connector):

        self.kg = kg_connector

    def step(self, state, action):

        current_node = state.current_node

        current_node_id = current_node["_id"]

        neighbors = self.kg.get_neighbors(current_node_id)

        # =========================
        # STOP ACTION
        # =========================

        if action == 0:

            return state, 0.5, True

        # =========================
        # DEAD END
        # =========================

        if len(neighbors) == 0:

            return state, -1.0, True

        # =========================
        # SELECT NEXT NODE
        # =========================

        selected = neighbors[action % len(neighbors)]

        next_node = selected["node"]

        relation = selected["relation"]

        # =========================
        # UPDATE STATE
        # =========================

        state.history.append({
            "from": current_node["_id"],
            "relation": relation,
            "to": next_node["_id"]
        })

        state.current_node = next_node

        state.evidence.append({
            "node": next_node,
            "relation": relation
        })

        reward = self._compute_reward(
            relation,
            next_node
        )

        done = False

        return state, reward, done

    def _compute_reward(self, relation, node):

        reward = 0.2

        # =========================
        # Reward useful evidence
        # =========================

        useful_relations = [
            "HAS_FEATURE",
            "HAS_POSITIVE_ASPECT",
            "HAS_DETAIL"
        ]

        if relation in useful_relations:
            reward += 1.0

        return reward