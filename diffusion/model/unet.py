import torch
import torch.nn as nn 
from .res_conv_block import ResConvBlockWithCrossAttn
from .pos_embd import PositionalEmbedding
from .encoder import SimpleLabelEncoder

class UNet(nn.Module):
    """
    U-Net architecture conditional diffusion. Inspired by stable diffusion
    """
    def __init__(self, in_channels=1, out_channels=1, time_emb_dim=128,
                 text_emb_dim=128, base_channels=64, num_classes=10, device='cpu'):
        super().__init__()
        self.device = device

        # Time embedding 
        self.time_mlp = nn.Sequential(
            PositionalEmbedding(time_emb_dim, device=device),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        ).to(device)

        # Text encoder 
        self.text_encoder = SimpleLabelEncoder(
            num_classes=num_classes,
            embed_dim=text_emb_dim,
            device=device
        ).to(device)

        # Initial convolution
        self.initial_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.down1 = DownBlockWithCrossAttn(
            base_channels, base_channels * 2,
            time_emb_dim, text_emb_dim,
            num_heads=4, use_self_attn=False  
        )
        self.down2 = DownBlockWithCrossAttn(
            base_channels * 2, base_channels * 4,
            time_emb_dim, text_emb_dim,
            num_heads=4, use_self_attn=True  
        )

        # Bottleneck with both self and cross-attention
        self.bottleneck = ResConvBlockWithCrossAttn(
            base_channels * 4, base_channels * 4,
            time_emb_dim, text_emb_dim,
            num_heads=4, use_self_attn=True
        )

        self.up1 = UpBlockWithCrossAttn(
            in_channels=base_channels * 4,
            out_channels=base_channels * 2,
            time_emb_dim=time_emb_dim,
            text_emb_dim=text_emb_dim,
            skip_channels=base_channels * 4,
            num_heads=4,
            use_self_attn=True
        )
        self.up2 = UpBlockWithCrossAttn(
            in_channels=base_channels * 2,
            out_channels=base_channels,
            time_emb_dim=time_emb_dim,
            text_emb_dim=text_emb_dim,
            skip_channels=base_channels * 2,
            num_heads=4,
            use_self_attn=False
        )

        self.final_conv = nn.Conv2d(base_channels, out_channels, 1)
        self.device = device

    def forward(self, x, t, labels):
        t = t.to(self.device)
        t_emb = self.time_mlp(t)

        text_emb = self.text_encoder(labels)  # [batch, text_emb_dim]
        text_emb = text_emb.unsqueeze(1).to(self.device)

        x = self.initial_conv(x)
        x1, skip1 = self.down1(x, t_emb, text_emb)
        x2, skip2 = self.down2(x1, t_emb, text_emb)

        x3 = self.bottleneck(x2, t_emb, text_emb)

        x = self.up1(x3, skip2, t_emb, text_emb)
        x = self.up2(x, skip1, t_emb, text_emb)

        return self.final_conv(x)


class DownBlockWithCrossAttn(nn.Module):
    """Downsampling block with cross-attention."""
    def __init__(self, in_channels, out_channels, time_emb_dim, text_emb_dim,
                 num_heads=4, use_self_attn=False):
        super().__init__()
        self.res_conv = ResConvBlockWithCrossAttn(
            in_channels, out_channels, time_emb_dim, text_emb_dim,
            num_heads, use_self_attn
        )
        self.downsample = nn.Conv2d(out_channels, out_channels, 4, 2, 1)

    def forward(self, x, t_emb, text_emb):
        x = self.res_conv(x, t_emb, text_emb)
        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlockWithCrossAttn(nn.Module):
    """Upsampling block with cross-attention."""
    def __init__(self, in_channels, out_channels, time_emb_dim, text_emb_dim,
                 skip_channels, num_heads=4, use_self_attn=False):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, 4, 2, 1)
        self.res_conv = ResConvBlockWithCrossAttn(
            in_channels + skip_channels, out_channels,
            time_emb_dim, text_emb_dim,
            num_heads, use_self_attn
        )

    def forward(self, x, skip, t_emb, text_emb):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res_conv(x, t_emb, text_emb)
        return x
