import torch


class PPOTrainer:
    def __init__(self, actor, critic):
        self.actor = actor
        self.critic = critic

    def compute_loss(self, log_probs, values, rewards):
        advantage = rewards - values.detach()

        policy_loss = -(log_probs * advantage).mean()
        value_loss = (values - rewards).pow(2).mean()

        return policy_loss + value_loss