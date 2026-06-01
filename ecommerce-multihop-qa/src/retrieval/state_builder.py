import numpy as np
from typing import List, Dict, Any
from src.kg.neo4j_connector import Neo4jConnector
from src.retrieval.query_encoder import QueryEncoder


class RetrievalState:

    def __init__(self, query, current_node):

        self.query = query

        self.current_node = current_node

        self.history = []

        self.evidence = []

        self.uncertainty = 1.0


class StateBuilder:
    def __init__(self, kg: Neo4jConnector):
        self.kg = kg
        self.encoder = QueryEncoder()

    def init_state(self, query: str):
        start_node = self.kg.get_random_product()
        return RetrievalState(query, start_node)

    def update_state(self, state: RetrievalState, action_result: Dict):
        state.history.append(action_result["action"])
        state.current_node = action_result["next_node"]

        if action_result.get("evidence"):
            state.evidence.append(action_result["evidence"])

        state.uncertainty = self._compute_uncertainty(state)

        return state

    def _compute_uncertainty(self, state: RetrievalState):
        if len(state.evidence) == 0:
            return 1.0
        return max(0.1, 1.0 - len(state.evidence) * 0.15)