from src.retrieval.state_builder import StateBuilder
from src.rl.actor import Actor
from src.retrieval.stopping import StoppingPolicy


class RetrievalAgent:
    def __init__(self, env, actor, state_builder):
        self.env = env
        self.actor = actor
        self.state_builder = state_builder
        self.stopping = StoppingPolicy()

    def run(self, query):
        state = self.state_builder.init_state(query)

        done = False

        while not done:
            state_embedding = self.state_builder.encoder.encode(state.query)

            action, _ = self.actor.select_action(state_embedding)

            state, reward, done = self.env.step(state, action)

            if self.stopping.should_stop(state):
                done = True

        return state.evidence