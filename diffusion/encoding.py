import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    """
    Creates sinusoidal positional embeddings for the timestep t.
    """

    def __init__(self, embedding_dim: int, device="cpu"):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.half_embedding_dim = embedding_dim // 2
        self.div_term = torch.exp(
            torch.arange(0, self.half_embedding_dim, dtype=torch.float32)
            * -(torch.log(torch.tensor(10000.0)) / self.half_embedding_dim)
        ).to(device)

    def forward(self, t):
        if t.dim() > 1:
            t = t.view(-1)  # flatten if shape is [batch, 1]
        # Move t to the same device as div_term
        t = t.to(self.div_term.device)
        emb = t[:, None].float() * self.div_term[None, :]  # [batch, half_dim]
        emb = torch.cat(
            [torch.sin(emb), torch.cos(emb)], dim=1
        )  # [batch, embedding_dim] # Corrected dim
        return emb
