import torch 
import torch.nn as nn 

class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, device='cpu'):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.half_embedding_dim = embedding_dim // 2
        self.device = device
        self.div_term = torch.exp(torch.arange(0, self.half_embedding_dim, dtype=torch.float32) *
                                  - (torch.log(torch.tensor(10000.0)) / self.half_embedding_dim)).to(device)


    def forward(self, t):
        if t.dim() > 1:
              t = t.view(-1)  # flatten if shape is [batch, 1]
        t = t.to(self.device)
        emb = t[:, None].float() * self.div_term[None, :]  # [batch, half_dim]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1) # [batch, embedding_dim] 
        return emb.to(self.device)