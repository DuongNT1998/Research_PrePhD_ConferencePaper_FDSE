import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, input_dim=384):
        super().__init__()

        self.value_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state):
        return self.value_net(state)