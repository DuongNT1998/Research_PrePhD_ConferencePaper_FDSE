import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=256, action_dim=10):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state_embedding):
        return self.net(state_embedding)