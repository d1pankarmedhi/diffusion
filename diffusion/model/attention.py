import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossAttentionBlock(nn.Module):
    """
    Cross-Attention mechanism for conditioning in diffusion models.
    Image features (Q) attend text features (K, V)
    """
    def __init__(self, query_dim, context_dim, num_heads=8, head_dim=64, dropout=0.0):
        super().__init__()

        inner_dim = head_dim * num_heads
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5  

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)  
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False) 
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)  

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context):
        batch_size, spatial_dim, _ = x.shape

        q = self.to_q(x)  
        k = self.to_k(context)  
        v = self.to_v(context)  

        q = q.reshape(batch_size, spatial_dim, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)


        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)  

        attn_output = torch.matmul(attn_weights, v)  

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, spatial_dim, -1)
        output = self.to_out(attn_output)

        return output


class SelfAttentionBlock(nn.Module):
    """
    Self-Attention mechanism for spatial relationships in image features.
    """
    def __init__(self, channels, num_heads=8, head_dim=64):
        super().__init__()

        inner_dim = head_dim * num_heads
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.to_qkv = nn.Linear(channels, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, channels),
            nn.Dropout(0.0)
        )

    def forward(self, x):
        batch_size, spatial_dim, _ = x.shape

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(batch_size, spatial_dim, self.num_heads, self.head_dim).transpose(1, 2), qkv)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, spatial_dim, -1)
        output = self.to_out(attn_output)

        return output
