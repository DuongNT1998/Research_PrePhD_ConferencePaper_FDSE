import torch

from torch.distributions import Categorical

from src.rl.state_encoder import StateEncoder
from src.rl.ppo_trainer import PPOTrainer
from src.rl.replay_buffer import ReplayBuffer


class RLRetriever:

    def __init__(self):

        self.encoder = StateEncoder()

        self.trainer = PPOTrainer()

        self.buffer = ReplayBuffer()

    def select_action(
        self,
        state_vector,
        num_actions
    ):

        probs = self.trainer.actor(
            state_vector
        )

        probs = probs[:num_actions]

        probs = probs / probs.sum()

        dist = Categorical(probs)

        action = dist.sample()

        log_prob = dist.log_prob(action)

        return (
            action.item(),
            log_prob
        )