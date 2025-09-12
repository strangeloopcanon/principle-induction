from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np

# Ensure repo root on path
ROOT = os.path.dirname(os.path.dirname(__file__))
import sys

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rl.envs import ECAParamEnv, LifeParamEnv
from rl.algo import grpo_step, gspo_group_step
from rl.prompts import (
    format_eca_obs,
    parse_eca_action,
    format_life_obs,
    parse_life_action,
)


def run(args: argparse.Namespace) -> None:
    # Load MLX model
    from mlx_lm import load
    from mlx_gen_parity import GenerationConfig, apply_lora, SoftPromptHook
    import mlx.core as mx  # noqa: F401
    from mlx.optimizers import AdamW

    model, tokenizer = load(args.model)
    # Reference (frozen) model for KL anchor (can override)
    ref_id = args.ref_model if args.ref_model else args.model
    ref_model, _ = load(ref_id)

    hooks = []
    if args.lora:
        apply_lora(model, rank=8, alpha=16)
    if args.soft_prompt:
        hooks.append(SoftPromptHook(n_virtual=10, init="rand", param_key="_soft_prompt"))

    opt = AdamW(learning_rate=args.lr)

    # Build env
    if args.env == "eca":
        env = ECAParamEnv(width=args.width, num_pairs=args.pairs, rule_number=None, max_attempts=args.attempts, reward_mode=args.reward_mode, seed=args.seed)
    else:
        env = LifeParamEnv(height=args.height, width=args.width, num_pairs=args.pairs, born_set={3}, survive_set={2, 3}, max_attempts=args.attempts, reward_mode=args.reward_mode, seed=args.seed)

    rng = np.random.default_rng(args.seed)

    # Prepare logging
    if args.log_csv:
        os.makedirs(os.path.dirname(args.log_csv) or ".", exist_ok=True)
        with open(args.log_csv, "w") as f:
            f.write("step,reward_mean,acc_mean,kl,beta,adv_mode,best_text\n")
    beta = args.beta
    for step in range(args.steps):
        obs = env.reset(seed=int(rng.integers(10_000_000)))
        if args.env == "eca":
            prompt = format_eca_obs(obs)
            prefix = tokenizer.encode("RULE=")
            max_tokens = 8
        else:
            prompt = format_life_obs(obs)
            prefix = tokenizer.encode("B")
            max_tokens = 12

        from mlx_gen_parity import generate

        actions = []
        rewards = []
        accs = []
        # Sample K actions per prompt
        for _ in range(args.samples):
            cfg = GenerationConfig(max_tokens=max_tokens, temperature=args.temp, top_p=args.top_p, force_words_ids=[prefix], seed=int(rng.integers(10_000_000)))
            out = generate(model, tokenizer, prompt, cfg)
            text = out.get("text") or out.get("texts", [""])[0]
            if args.env == "eca":
                action, action_text = parse_eca_action(text)
                res = env.step(action)
            else:
                bits, action_text = parse_life_action(text)
                res = env.step(bits)
            actions.append(action_text)
            rewards.append(float(res.reward))
            accs.append(float(res.info.get("acc", 0.0)))

        # GSPO group update
        loss, metrics = gspo_group_step(
            model,
            ref_model,
            tokenizer,
            prompt,
            actions,
            rewards,
            optimizer=opt,
            eta=args.eta,
            beta=beta,
            adv_norm=getattr(args, "adv_norm", "softmax"),
            hooks=hooks,
            dtype=None,
        )
        # KL schedule
        if getattr(args, "beta_schedule", "fixed") == "target":
            kl = metrics.get("kl_mean", 0.0)
            tgt = max(1e-4, getattr(args, "target_kl", 0.05))
            ratio = kl / tgt
            import math
            adj = math.exp(max(min(ratio - 1.0, 0.2), -0.2))
            beta = max(1e-6, min(1.0, beta * adj))

        # EMA reference update (prefer mlx-gen-parity helper if available)
        if getattr(args, "ema_ref_decay", 0.0) > 0.0:
            decay = args.ema_ref_decay
            try:
                from mlx_gen_parity.utils import ema_update

                ema_update(ref_model, model, decay)
            except Exception:
                import mlx.core as mx

                ref_params = ref_model.trainable_parameters()
                cur_params = model.trainable_parameters()
                ema = mx.tree_map(lambda r, c: decay * r + (1.0 - decay) * c, ref_params, cur_params)
                ref_model.update(ema)

        line = (
            f"step={step:04d} reward_mean={np.mean(rewards):.4f} acc_mean={np.mean(accs):.4f} "
            f"loss={loss:.4f} kl={metrics.get('kl_mean',0.0):.4f} beta={beta:.4f} best_text={actions[int(np.argmax(rewards))]}"
        )
        print(line)
        if args.log_csv:
            with open(args.log_csv, "a") as f:
                best_idx = int(np.argmax(rewards))
                txt = '"' + actions[best_idx].replace('"', '""') + '"'
                f.write(
                    f"{step},{np.mean(rewards):.6f},{np.mean(accs):.6f},{metrics.get('kl_mean',0.0):.6f},{beta:.6f},{getattr(args,'adv_norm','softmax')},{txt}\n"
                )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=["eca", "life"], required=True)
    ap.add_argument("--model", type=str, default="mlx-community/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--ref-model", type=str, default="", help="Frozen reference policy id/path for KL anchor (defaults to --model)")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--height", type=int, default=8)
    ap.add_argument("--attempts", type=int, default=1)
    ap.add_argument("--reward-mode", type=str, choices=["delta", "acc"], default="acc")
    ap.add_argument("--samples", type=int, default=4, help="Number of samples per prompt for GSPO group")
    ap.add_argument("--eta", type=float, default=5.0, help="Softmax/adv temperature (group weights)")
    ap.add_argument("--beta", type=float, default=0.05, help="KL penalty weight vs reference policy")
    ap.add_argument("--beta-schedule", type=str, choices=["fixed", "target"], default="fixed", help="KL schedule mode")
    ap.add_argument("--target-kl", type=float, default=0.05, help="Target KL if schedule=target")
    ap.add_argument("--temp", type=float, default=0.7, help="Sampling temperature")
    ap.add_argument("--top-p", dest="top_p", type=float, default=0.95, help="Top-p nucleus sampling")
    ap.add_argument("--adv-norm", type=str, choices=["softmax", "zscore", "rank", "baseline"], default="softmax", help="Group advantage normalization")
    ap.add_argument("--ema-ref-decay", type=float, default=0.0, help="EMA decay for reference policy (0 disables)")
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--soft-prompt", action="store_true")
    ap.add_argument("--log-csv", type=str, default="")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
