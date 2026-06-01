import torch
import torch.nn.functional as F


class Actor:
    def __init__(self, policy_net):
        self.policy_net = policy_net

    def select_action(self, state_embedding):
        logits = self.policy_net(state_embedding)
        probs = F.softmax(logits, dim=-1)

        action = torch.multinomial(probs, 1).item()
        return action, probs[action].item()