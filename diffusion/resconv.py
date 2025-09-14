import torch.nn as nn


class ResConvBlock(nn.Module):
    """
    Residual Convolutional Block with time embedding conditioning.
    """

    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(
            num_groups=min(8, out_channels), num_channels=out_channels
        )
        self.act1 = nn.SiLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(
            num_groups=min(8, out_channels), num_channels=out_channels
        )
        self.act2 = nn.SiLU()

        self.time_emb_proj = nn.Linear(time_emb_dim, out_channels)

        if in_channels != out_channels:
            self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.res_conv = nn.Identity()

    def forward(self, x, time_emb):
        # conv → norm → act
        h = self.act1(self.norm1(self.conv1(x)))

        # Add time embedding
        time_emb_h = self.time_emb_proj(time_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + time_emb_h

        # conv → norm → act
        h = self.act2(self.norm2(self.conv2(h)))

        # Residual connection
        return h + self.res_conv(x)
