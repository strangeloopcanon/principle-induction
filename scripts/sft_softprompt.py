from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.eca import rule_number_to_table, eca_step
from rl.prompts import format_eca_obs
from scripts.text_utils import apply_chat_template, greedy_binary_completion, greedy_decimal_completion

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map

try:
    from third_party.mlx_gen_compat import SoftPromptHook, loss_forward
except Exception:
    SoftPromptHook = None
    loss_forward = None


def xent_loss(logits: mx.array, labels: mx.array, ignore_index: int = -100) -> mx.array:
    V = logits.shape[-1]
    logp = nn.log_softmax(logits, axis=-1)
    B, T = labels.shape
    labels = labels.astype(mx.int32)
    valid = labels != ignore_index
    labels_clamped = mx.maximum(labels, 0)
    picked = mx.take_along_axis(logp, labels_clamped.reshape(B, T, 1), axis=-1).reshape(B, T)
    return -(picked * valid).sum() / (valid.sum() + 1e-8)


def build_table_example(rule: int, tokenizer, *, chat: bool = True) -> Tuple[List[int], List[int]]:
    true_bits = ''.join(map(str, rule_number_to_table(rule).tolist()))
    prompt = (
        f"Describe the Elementary Cellular Automaton Rule {rule}.\n"
        "Provide the 8-bit truth table as TABLE=<bits> for neighborhoods (l,c,r) in order 000,001,010,011,100,101,110,111.\n"
        "Only output TABLE=... on the last line.\n\nTABLE="
    )
    prompt = apply_chat_template(tokenizer, prompt, chat=chat)
    target = "TABLE=" + true_bits
    p_ids = tokenizer.encode(prompt)
    t_ids = tokenizer.encode(target)
    tokens = p_ids + t_ids
    labels = [-100] * len(p_ids) + t_ids
    return tokens, labels


def build_rule_example(rule: int, tokenizer, *, width: int = 32, pairs: int = 8, chat: bool = True) -> Tuple[List[int], List[int]]:
    rng = np.random.default_rng(0)
    xs = rng.integers(0, 2, size=(pairs, width), dtype=np.uint8)
    ys = np.stack([eca_step(x, rule_number=rule) for x in xs], axis=0)
    pairs_arr = np.stack([xs, ys], axis=1)
    prompt = format_eca_obs(pairs_arr)
    prompt = apply_chat_template(tokenizer, prompt, chat=chat)
    target = f"RULE={rule}"
    p_ids = tokenizer.encode(prompt)
    t_ids = tokenizer.encode(target)
    tokens = p_ids + t_ids
    labels = [-100] * len(p_ids) + t_ids
    return tokens, labels


def build_rollout_example(rule: int, tokenizer, *, width: int = 32, steps: int = 5, chat: bool = True) -> Tuple[List[int], List[int]]:
    rng = np.random.default_rng(0)
    x0 = rng.integers(0, 2, size=(width,), dtype=np.uint8)
    # Build prompt
    xbits = ''.join(map(str, x0.tolist()))
    prompt = (
        "Under the Elementary Cellular Automaton with toroidal wrap,\n"
        f"apply Rule {int(rule)} for {int(steps)} steps to the starting row.\n"
        "Answer strictly as: Y=<W binary digits>\n\n"
        f"X={xbits}\n\nY="
    )
    prompt = apply_chat_template(tokenizer, prompt, chat=chat)
    y = x0.copy()
    for _ in range(steps):
        y = eca_step(y, rule_number=rule)
    ybits = ''.join(map(str, y.tolist()))
    target = "Y=" + ybits
    p_ids = tokenizer.encode(prompt)
    t_ids = tokenizer.encode(target)
    tokens = p_ids + t_ids
    labels = [-100] * len(p_ids) + t_ids
    return tokens, labels


def train_soft_prompt(args) -> None:
    from mlx_lm import load

    if SoftPromptHook is None or loss_forward is None:
        raise RuntimeError("SoftPromptHook/loss_forward not available; ensure mlx-genkit or mlx-gen-parity is installed.")

    model, tokenizer = load(args.model)
    hook = SoftPromptHook(n_virtual=args.n_virtual, init="rand", param_key=args.param_key)
    prompt_param = hook.soft_prompt
    m = mx.zeros_like(prompt_param)
    v = mx.zeros_like(prompt_param)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    rng = np.random.default_rng(args.seed)
    rules = [int(x) for x in args.rules.split(',') if x.strip()]

    def step_one(tokens: List[int], labels: List[int], step_idx: int) -> float:
        x = mx.array([tokens], dtype=mx.int32)
        y = mx.array([labels], dtype=mx.int32)

        def loss_fn():
            logits = loss_forward(model, x, hooks=[hook])
            return xent_loss(logits[:, 1:, :], y[:, 1:])

        val_and_grad = nn.value_and_grad(model, loss_fn)
        loss, grads = val_and_grad()

        # Soft prompt gradient
        soft_grad = hook.soft_prompt.grad
        if args.grad_clip > 0:
            norm = mx.sqrt(mx.sum(soft_grad * soft_grad))
            norm_val = float(norm.item())
            if not np.isnan(norm_val) and norm_val > args.grad_clip:
                soft_grad = soft_grad * (args.grad_clip / (norm_val + 1e-8))

        # Manual Adam update on soft prompt only
        m[:] = beta1 * m + (1 - beta1) * soft_grad
        v[:] = beta2 * v + (1 - beta2) * (soft_grad * soft_grad)
        m_hat = m / (1 - beta1 ** (step_idx + 1))
        v_hat = v / (1 - beta2 ** (step_idx + 1))
        update = m_hat / (mx.sqrt(v_hat) + eps)
        prompt_param[:] = prompt_param - args.lr * update

        return float(loss.item())

    for step in range(args.steps):
        rule = int(rng.choice(rules))
        which = step % max(1, len(args.tasks))
        task = args.tasks[which]
        if task == 'table':
            tok_ids, lab_ids = build_table_example(rule, tokenizer, chat=args.chat)
        elif task == 'rule':
            tok_ids, lab_ids = build_rule_example(rule, tokenizer, width=args.width, pairs=args.pairs, chat=args.chat)
        else:
            steps_h = int(rng.integers(args.rollout_min, args.rollout_max + 1))
            tok_ids, lab_ids = build_rollout_example(rule, tokenizer, width=args.width, steps=steps_h, chat=args.chat)
        loss = step_one(tok_ids, lab_ids, step)
        if step % 50 == 0:
            print(f"step={step:04d} task={task} rule={rule} loss={loss:.4f}")

    # Quick eval: table accuracy for each rule
    results = {}
    for r in rules:
        digits = greedy_binary_completion(model, tokenizer,
                                          f"Describe the Elementary Cellular Automaton Rule {r}.\nProvide the 8-bit truth table as TABLE=<bits> for neighborhoods (l,c,r) in order 000,001,010,011,100,101,110,111.\nOnly output TABLE=... on the last line.\n\nTABLE=",
                                          chat=args.chat, digits=8, hooks=[hook])
        true_bits = ''.join(map(str, rule_number_to_table(r).tolist()))
        acc = float(np.mean([int(a) == int(b) for a, b in zip(digits, true_bits)]))
        results[str(r)] = {"pred": digits, "true": true_bits, "acc": acc}
        print(f"EVAL RULE {r}: TABLE={digits} (true={true_bits}) acc={acc:.3f}")

    if args.save_npz:
        # Try to locate soft prompt weights by scanning model params matching n_virtual in dim 0
        nv = args.n_virtual
        sp = None
        params = model.trainable_parameters()
        # flatten tree
        flat: list[mx.array] = []
        def collect(x):
            if isinstance(x, mx.array):
                flat.append(x)
            return x
        tree_map(lambda t: collect(t), params)
        for arr in flat:
            if len(arr.shape) >= 2 and arr.shape[0] == nv:
                sp = arr
                break
        if sp is not None:
            os.makedirs(os.path.dirname(args.save_npz) or '.', exist_ok=True)
            np.savez_compressed(args.save_npz, soft_prompt=np.array(sp))
            print(f"Saved soft prompt to {args.save_npz}")
        else:
            print("Warning: could not locate soft prompt param to save.")

    if args.eval_out:
        os.makedirs(os.path.dirname(args.eval_out) or '.', exist_ok=True)
        with open(args.eval_out, 'w') as f:
            json.dump({"rules": rules, "table_eval": results}, f, indent=2)
            print("Saved eval results to", args.eval_out)


def parse_tasks(s: str) -> List[str]:
    out = []
    for t in s.split(','):
        t = t.strip().lower()
        if t in ("table", "rule", "rollout"):
            out.append(t)
    return out or ["table", "rule"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="mlx-community/Llama-3.2-3B-Instruct")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--rules", type=str, default="30,90,110")
    ap.add_argument("--tasks", type=str, default="table,rule")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--rollout-min", type=int, default=3)
    ap.add_argument("--rollout-max", type=int, default=10)
    ap.add_argument("--n-virtual", type=int, default=20)
    ap.add_argument("--param-key", type=str, default="_soft_prompt")
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--save-npz", dest="save_npz", type=str, default="")
    ap.add_argument("--eval-out", type=str, default="runs/sft_eval.json")
    args = ap.parse_args()

    args.tasks = parse_tasks(args.tasks)
    train_soft_prompt(args)


if __name__ == "__main__":
    main()
