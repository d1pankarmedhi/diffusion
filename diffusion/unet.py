import torch
import torch.nn as nn

from .encoding import PositionalEmbedding
from .resconv import ResConvBlock


class DownBlock(nn.Module):
    """
    Down-sampling block with a residual convolutional block.
    """

    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.res_conv_block = ResConvBlock(in_channels, out_channels, time_emb_dim)
        # Using Conv stride=2 instead of MaxPool for learnable downsampling
        self.downsample = nn.Conv2d(
            out_channels, out_channels, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x, time_emb):
        x = self.res_conv_block(x, time_emb)
        skip = x  # Save for skip connection
        x = self.downsample(x)
        return x, skip


class UpBlock(nn.Module):
    """
    Up-sampling block with an Upsample layer followed by a residual convolutional block.
    """

    def __init__(self, in_channels, out_channels, time_emb_dim, skip_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        # After concatenation: in_channels + skip_channels
        self.res_conv_block = ResConvBlock(
            in_channels + skip_channels, out_channels, time_emb_dim
        )

    def forward(self, x, skip_connection, time_emb):
        x = self.upsample(x)
        # Concatenate along channel dimension
        x = torch.cat([x, skip_connection], dim=1)
        x = self.res_conv_block(x, time_emb)
        return x


class UNet(nn.Module):
    """
    U-Net denoising diffusion model.
    Takes a noisy image x_t and a timestep t, predicts the noise ε.
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        time_emb_dim=128,
        base_channels=64,
        device="cpu",
    ):
        super().__init__()

        # Positional embedding for time
        self.time_mlp = nn.Sequential(
            PositionalEmbedding(time_emb_dim, device=device),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # Encoder
        self.initial_conv = nn.Conv2d(
            in_channels, base_channels, kernel_size=3, padding=1
        )
        self.down1 = DownBlock(base_channels, base_channels * 2, time_emb_dim)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, time_emb_dim)

        # Bottleneck
        self.bottleneck = ResConvBlock(
            base_channels * 4, base_channels * 4, time_emb_dim
        )

        # Decoder
        self.up1 = UpBlock(
            in_channels=base_channels * 4,
            out_channels=base_channels * 2,
            time_emb_dim=time_emb_dim,
            skip_channels=base_channels * 4,
        )
        self.up2 = UpBlock(
            in_channels=base_channels * 2,
            out_channels=base_channels,
            time_emb_dim=time_emb_dim,
            skip_channels=base_channels * 2,
        )

        # Final conv
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        self.device = device

    def forward(self, x, t):
        # Encode timestep into embedding
        t_emb = self.time_mlp(t)
        # print(f"[INFO] Timestep embedding shape: {t_emb.shape}")
        # Encoder path
        x = self.initial_conv(x)  # [B, base, H, W]
        x1, skip1 = self.down1(
            x, t_emb
        )  # [B, 2*base, H/2, W/2], skip1=[B,2*base,H/2,W/2]
        x2, skip2 = self.down2(
            x1, t_emb
        )  # [B, 4*base, H/4, W/4], skip2=[B,4*base,H/4,W/4]

        # print(f"[INFO] Original shape: {x.shape}, down1: {x1.shape}, down2: {x2.shape}")
        # Bottleneck
        x3 = self.bottleneck(x2, t_emb)

        # print(f"[INFO] Bottlenect shape: {x3.shape}")

        # Decoder path
        x = self.up1(x3, skip2, t_emb)  # [B, 2*base, H/2, W/2]
        # print(f"[INFO] up1: {x.shape}")
        x = self.up2(x, skip1, t_emb)  # [B, base, H, W]
        # print(f"[INFO] up2: {x.shape}")

        # Final output
        return self.final_conv(x)  # Predicts noise ε
