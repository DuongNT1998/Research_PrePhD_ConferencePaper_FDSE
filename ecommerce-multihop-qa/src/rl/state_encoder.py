import torch
import torch.nn as nn

from sentence_transformers import SentenceTransformer


class StateEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.text_encoder = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        self.fc = nn.Sequential(

            nn.Linear(384 * 2 + 2, 256),

            nn.ReLU(),

            nn.Linear(256, 128),

            nn.ReLU()

        )

    def encode_text(self, text):

        emb = self.text_encoder.encode(text)

        return torch.tensor(
            emb,
            dtype=torch.float32
        )

    def forward(
        self,
        query,
        current_node,
        hop_count,
        evidence_count
    ):

        q_emb = self.encode_text(query)

        n_emb = self.encode_text(current_node)

        extra = torch.tensor([
            hop_count,
            evidence_count
        ], dtype=torch.float32)

        state = torch.cat([
            q_emb,
            n_emb,
            extra
        ])

        return self.fc(state)