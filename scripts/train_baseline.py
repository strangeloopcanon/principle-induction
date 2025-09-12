from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import AdamW

from models.mini_transformer import MiniTransformerConfig, MiniTransformerLM, xent_loss
from datasets.eca import make_eca_trajectories
from datasets.life import make_life_trajectories


def build_sequences_from_traj(traj: np.ndarray, sep_id: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Given trajectories (N, T, ...), build sequences for prediction-only baseline.

    For each consecutive pair (frame_t, frame_{t+1}), create input tokens:
      seq = flatten(frame_t) + [sep] + flatten(frame_{t+1})
    Labels ignore first len(frame_t)+1 tokens, supervise the last len(frame_{t+1}).

    Returns (tokens, labels) as np.int32 arrays shaped (N*(T-1), L).
    """
    N, T = traj.shape[0], traj.shape[1]
    frames = traj.reshape(N, T, -1)
    W = frames.shape[-1]
    L = W + 1 + W
    xs = []
    ys = []
    for n in range(N):
        for t in range(T - 1):
            a = frames[n, t, :]
            b = frames[n, t + 1, :]
            seq = np.concatenate([a, np.array([sep_id], dtype=np.int32), b], axis=0)
            lab = np.full(L, -100, dtype=np.int32)
            lab[W + 1 :] = b
            xs.append(seq.astype(np.int32))
            ys.append(lab)
    return np.stack(xs, axis=0), np.stack(ys, axis=0)


def batch_iter(tokens: np.ndarray, labels: np.ndarray, batch: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = np.arange(tokens.shape[0])
    rng.shuffle(idx)
    for i in range(0, len(idx), batch):
        sel = idx[i : i + batch]
        yield tokens[sel], labels[sel]


def train_epoch(model, opt, tok, lab):
    def step(x, y):
        def loss_fn():
            logits = model(x)
            return xent_loss(logits[:, 1:, :], y[:, 1:])  # drop first pos for stability

        val_and_grad = nn.value_and_grad(model, loss_fn)
        loss, grads = val_and_grad()
        from mlx.optimizers import clip_grad_norm

        clipped, _ = clip_grad_norm(grads, 1.0)
        opt.update(model, clipped)
        return float(loss.item())

    return step(tok, lab)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["eca", "life"], required=True)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--height", type=int, default=16)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--num-seqs", type=int, default=128)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # Data
    if args.task == "eca":
        traj = make_eca_trajectories(rule_number=110, width=args.width, horizon=args.horizon, num_seqs=args.num_seqs, seed=args.seed)
    else:
        traj = make_life_trajectories(born_set={3}, survive_set={2, 3}, height=args.height, width=args.width, horizon=args.horizon, num_seqs=args.num_seqs, seed=args.seed)
    tokens_np, labels_np = build_sequences_from_traj(traj, sep_id=2)

    # Model (vocab: 0,1 plus sep(2))
    vocab_size = 3
    max_seq_len = tokens_np.shape[1]
    cfg = MiniTransformerConfig(vocab_size=vocab_size, d_model=args.d_model, n_heads=args.heads, n_layers=args.layers, d_ff=args.d_model * 4, max_seq_len=max_seq_len)
    model = MiniTransformerLM(cfg)
    opt = AdamW(learning_rate=args.lr)

    # Training loop (simple)
    for step in range(args.steps):
        for btok, blab in batch_iter(tokens_np, labels_np, args.batch, seed=args.seed + step):
            x = mx.array(btok, dtype=mx.int32)
            y = mx.array(blab, dtype=mx.int32)
            loss = train_epoch(model, opt, x, y)
            break  # one batch per step to keep demo fast
        if step % 10 == 0:
            print(f"step={step} loss={loss:.4f}")


if __name__ == "__main__":
    main()

