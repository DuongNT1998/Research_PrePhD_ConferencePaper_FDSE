import torch
import os


def save_checkpoint(
    actor,
    critic,
    path="outputs/checkpoints"
):

    os.makedirs(path, exist_ok=True)

    torch.save(
        actor.state_dict(),
        f"{path}/actor.pt"
    )

    torch.save(
        critic.state_dict(),
        f"{path}/critic.pt"
    )


def load_checkpoint(
    actor,
    critic,
    path="outputs/checkpoints"
):

    actor.load_state_dict(
        torch.load(f"{path}/actor.pt")
    )

    critic.load_state_dict(
        torch.load(f"{path}/critic.pt")
    )

    return actor, critic