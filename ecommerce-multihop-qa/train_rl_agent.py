import torch
import numpy as np

from src.kg.neo4j_connector import Neo4jConnector
from src.retrieval.state_builder import StateBuilder
from src.rl.policy_network import PolicyNetwork
from src.rl.actor import Actor
from src.rl.critic import Critic
from src.rl.kg_env import KGEnvironment
from src.rl.ppo_trainer import PPOTrainer
from src.config.settings import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

class PPOTrainingPipeline:
    def __init__(self):

        # =========================
        # 1. KG + ENV
        # =========================
      # Thay thế bằng thông tin cấu hình thực tế của bạn
        self.kg = Neo4jConnector(uri=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
        self.env = KGEnvironment(self.kg)
        self.state_builder = StateBuilder(self.kg)

        # =========================
        # 2. MODELS
        # =========================
        self.policy = PolicyNetwork(input_dim=384, action_dim=10)
        self.actor = Actor(self.policy)
        self.critic = Critic(input_dim=384)

        self.trainer = PPOTrainer(self.actor, self.critic)

        # =========================
        # 3. MEMORY BUFFER
        # =========================
        self.trajectories = []

    def rollout(self, query, max_steps=5):

        state = self.state_builder.init_state(query)

        log_probs = []
        values = []
        rewards = []

        for step in range(max_steps):

            state_emb = torch.randn(384)  # placeholder encoder

            action, log_prob = self.actor.select_action(state_emb)

            value = self.critic(state_emb)

            next_state, reward, done = self.env.step(state, action)

            log_probs.append(torch.tensor(log_prob))
            values.append(value)
            rewards.append(reward)

            state = next_state

            if done:
                break

        return log_probs, values, rewards

    def update(self, log_probs, values, rewards):

        rewards = torch.tensor(rewards, dtype=torch.float)

        values = torch.cat(values).squeeze()

        loss = self.trainer.compute_loss(
            torch.stack(log_probs),
            values,
            rewards
        )

        loss.backward()

        print(f"Loss: {loss.item():.4f}")

    def train(self, epochs=10):

        queries = [
            "best laptop under 1000 dollars",
            "good headphones with noise cancellation",
            "cheap gaming mouse with good sensor"
        ]

        for epoch in range(epochs):

            total_reward = 0

            for query in queries:

                log_probs, values, rewards = self.rollout(query)

                self.update(log_probs, values, rewards)

                total_reward += sum(rewards)

            print(f"[Epoch {epoch}] Reward: {total_reward}")


if __name__ == "__main__":
    pipeline = PPOTrainingPipeline()
    pipeline.train()