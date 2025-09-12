from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


@dataclass
class MiniTransformerConfig:
    vocab_size: int
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 1024
    max_seq_len: int = 1024
    dropout: float = 0.0  # placeholder; dropout not applied for simplicity


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def __call__(self, x: mx.array, attn_mask: Optional[mx.array] = None) -> mx.array:
        B, T, D = x.shape
        H = self.n_heads
        q = self.q_proj(x).reshape(B, T, H, self.head_dim).transpose(0, 2, 1, 3)  # (B,H,T,hd)
        k = self.k_proj(x).reshape(B, T, H, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, H, self.head_dim).transpose(0, 2, 1, 3)

        scale = 1.0 / (self.head_dim ** 0.5)
        att = (q @ k.transpose(0, 1, 3, 2)) * scale  # (B,H,T,T)
        if attn_mask is not None:
            # attn_mask shape: (T,T) with 0 for allow, -inf for block
            att = att + attn_mask.reshape(1, 1, T, T)
        att = nn.softmax(att, axis=-1)
        out = att @ v  # (B,H,T,hd)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, D)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def __call__(self, x: mx.array, attn_mask: Optional[mx.array] = None) -> mx.array:
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask)
        x = x + self.ff(self.ln2(x))
        return x


class MiniTransformerLM(nn.Module):
    def __init__(self, cfg: MiniTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = [TransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff) for _ in range(cfg.n_layers)]
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def __call__(self, tokens: mx.array) -> mx.array:
        # tokens: [B, T] int32
        B, T = tokens.shape
        assert T <= self.cfg.max_seq_len
        pos_ids = mx.arange(T, dtype=mx.int32).reshape(1, T)
        pos_ids = mx.broadcast_to(pos_ids, (B, T))
        x = self.embed(tokens) + self.pos(pos_ids)
        attn_mask = make_causal_mask(T, x.dtype)
        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


def make_causal_mask(T: int, dtype) -> mx.array:
    # 0 on allowed, -inf on blocked entries
    mask = mx.full((T, T), float("-inf"), dtype=dtype)
    mask = mx.triu(mask, k=1)
    return mask


def xent_loss(logits: mx.array, labels: mx.array, ignore_index: int = -100) -> mx.array:
    # logits: [B, T, V], labels: [B, T]
    V = logits.shape[-1]
    logprobs = nn.log_softmax(logits, axis=-1)
    B, T = labels.shape
    labels = labels.astype(mx.int32)
    valid = labels != ignore_index
    labels_clamped = mx.maximum(labels, 0)
    gathered = mx.take_along_axis(logprobs, labels_clamped.reshape(B, T, 1), axis=-1).reshape(B, T)
    loss = -(gathered * valid).sum() / (valid.sum() + 1e-8)
    return loss

