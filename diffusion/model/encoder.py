import torch.nn as nn

class SimpleLabelEncoder(nn.Module):
    """
    Simple embedding-based text encoder for FashionMNIST class labels.
    Maps categorical labels (0-9) to dense embeddings.
    """
    def __init__(self, num_classes=10, embed_dim=128, device='cpu'):
        super().__init__()
        self.device = device
        self.embedding = nn.Embedding(num_classes + 1, embed_dim).to(device) # +1 for null token
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*2),
            nn.SiLU(),
            nn.Linear(embed_dim*2, embed_dim)
        ).to(device)
        self.num_classes = num_classes

    def forward(self, labels):
        """
        labels: tensor of class labels
        """
        if labels.max() > self.num_classes:
            raise ValueError(f"Label index {labels.max()} exceeds num_classes {self.num_classes}")

        return self.mlp(self.embedding(labels)) # [batch, embed_dim]