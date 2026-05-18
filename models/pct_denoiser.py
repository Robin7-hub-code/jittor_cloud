from __future__ import annotations

import math



def require_jittor():
    try:
        import jittor as jt
        from jittor import nn
    except ImportError as exc:
        raise ImportError(
            "Jittor is required. Install it first, e.g. pip install jittor"
        ) from exc
    return jt, nn


jt, nn = require_jittor()


class SelfAttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)

    def execute(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scale = math.sqrt(max(q.shape[-1], 1))
        attn = nn.softmax(jt.matmul(q, k.transpose((0, 2, 1))) / scale, dim=-1)
        out = jt.matmul(attn, v)
        out = self.out_proj(out)
        return self.norm(out + x)


class PCTDenoiser(nn.Module):
    def __init__(self, in_channels: int = 3, embed_dim: int = 128, depth: int = 4):
        super().__init__()
        self.in_proj = nn.Linear(in_channels, embed_dim)
        self.blocks = nn.ModuleList([SelfAttentionBlock(embed_dim) for _ in range(depth)])
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 3),
        )

    def execute(self, noisy_points):
        x = self.in_proj(noisy_points)
        for blk in self.blocks:
            x = blk(x)

        global_feat = jt.max(x, dim=1, keepdims=True)[0]
        global_feat = global_feat.broadcast([x.shape[0], x.shape[1], global_feat.shape[-1]])
        feat = jt.concat([x, global_feat], dim=-1)
        residual = self.mlp(feat)
        return noisy_points - residual
