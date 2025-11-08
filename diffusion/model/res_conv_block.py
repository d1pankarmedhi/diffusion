import torch.nn as nn 
from .attention import CrossAttentionBlock, SelfAttentionBlock

class ResConvBlockWithCrossAttn(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, text_emb_dim,
                 num_heads=4, use_self_attn=True):
        super().__init__()

        self.use_self_attn = use_self_attn

        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)

        if use_self_attn:
            self.self_attn = SelfAttentionBlock(
                channels=out_channels,
                num_heads=num_heads,
                head_dim=out_channels // num_heads
            )
            self.norm_self_attn = nn.GroupNorm(8, out_channels)

        self.cross_attn = CrossAttentionBlock(
            query_dim=out_channels,
            context_dim=text_emb_dim,
            num_heads=num_heads,
            head_dim=out_channels // num_heads
        )
        self.norm_cross_attn = nn.GroupNorm(8, out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)

        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        self.act = nn.SiLU()

    def forward(self, x, t_emb, text_emb):
        """
        Args:
            x: Feature map [batch, channels, H, W]
            t_emb: Time embedding [batch, time_emb_dim]
            text_emb: Text embedding [batch, seq_len, text_emb_dim]

        Returns:
            Output features [batch, out_channels, H, W]
        """
        residual = self.residual_conv(x)

        h = self.conv1(x)
        h = self.norm1(h)
        h = h + self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.act(h)

        batch, channels, height, width = h.shape

        # Self Attention (if enabled)
        if self.use_self_attn:
            h_flat = h.view(batch, channels, -1).transpose(1, 2) # [B, C, H, W] -> [B, H*W, C]
            h_self_attn = self.self_attn(h_flat)
            h_self_attn = h_self_attn.transpose(1, 2).view(batch, channels, height, width)
            h = h + h_self_attn  
            h = self.norm_self_attn(h)

        # Cross-Attention 
        h_flat = h.view(batch, channels, -1).transpose(1, 2) 
        h_cross_attn = self.cross_attn(h_flat, text_emb)
        h_cross_attn = h_cross_attn.transpose(1, 2).view(batch, channels, height, width)
        h = h + h_cross_attn  
        h = self.norm_cross_attn(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)

        return h + residual
