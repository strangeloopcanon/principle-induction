from __future__ import annotations

import argparse
import os
import sys
from typing import List

import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.eca import eca_step, rule_number_to_table
from rl.prompts import format_eca_rollout_obs
from scripts.text_utils import apply_chat_template, clean_llm_output, greedy_binary_completion

try:
    from third_party.mlx_gen_compat import GenerationConfig, generate
except Exception:
    GenerationConfig = None
    generate = None


def gen_text(model, tokenizer, prompt: str, *, temp: float = 0.2, top_p: float = 0.95, max_tokens: int = 96, seed: int = 1234, chat: bool = False) -> str:
    prompt = apply_chat_template(tokenizer, prompt, chat=chat)
    if generate is None or GenerationConfig is None:
        raise RuntimeError("No generation backend available. Install mlx-genkit or mlx-gen-parity.")
    cfg = GenerationConfig(max_tokens=max_tokens, temperature=temp, top_p=top_p, seed=seed)
    out = generate(model, tokenizer, prompt, cfg)
    raw = out.get("text") or out.get("texts", [""])[0]
    return clean_llm_output(raw)


def probe_table(model, tokenizer, rule: int, *, seed: int, chat: bool) -> None:
    true_tbl = rule_number_to_table(rule)
    prompt = (
        f"Describe the Elementary Cellular Automaton Rule {rule}.\n"
        "Provide the 8-bit truth table as TABLE=<bits> for neighborhoods (l,c,r) in order 000,001,010,011,100,101,110,111.\n"
        "Only output TABLE=... on the last line.\n\nTABLE="
    )
    print("=== ECA Truth Table (Rule {}) ===".format(rule))
    print("PROMPT:\n" + prompt)
    digits = greedy_binary_completion(model, tokenizer, prompt, chat=chat, digits=8)
    print("RESPONSE:\nTABLE=" + digits)
    print("TRUE BITS:", ''.join(map(str, true_tbl.tolist())))
    print()


def probe_rollout(model, tokenizer, rule: int, width: int, steps: int, *, seed: int, chat: bool) -> None:
    rng = np.random.default_rng(seed)
    x0 = rng.integers(0, 2, size=(width,), dtype=np.uint8)
    prompt = format_eca_rollout_obs(x0, rule=rule, steps=steps)
    print("=== ECA Rollout (Rule {} for {} steps, width {}) ===".format(rule, steps, width))
    print("PROMPT:\n" + prompt)
    digits = greedy_binary_completion(model, tokenizer, prompt, chat=chat, digits=width)
    print("RESPONSE:\nY=" + digits)
    y = x0.copy()
    for _ in range(steps):
        y = eca_step(y, rule_number=rule)
    print("GROUND TRUTH Y:", ''.join(map(str, y.tolist())))
    print()


def probe_normal(model, tokenizer, *, seed: int, chat: bool) -> None:
    rng = np.random.default_rng(seed)
    qs: List[str] = [
        "Explain cellular automata to a high school student in 3 sentences.",
        "List 3 differences between breadth-first search and depth-first search.",
        "Write a short Python function to check if a string is a palindrome.",
    ]
    print("=== General Q/A Probe ===")
    for i, q in enumerate(qs, 1):
        print(f"Q{i}: {q}")
        txt = gen_text(model, tokenizer, q + "\n\nAnswer:", temp=0.4, top_p=0.95, max_tokens=120, seed=int(rng.integers(10_000_000)), chat=chat)
        print("A{}:\n{}\n".format(i, txt.strip()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="MLX model path or id")
    ap.add_argument("--what", choices=["all", "table", "rollout", "normal"], default="all")
    ap.add_argument("--rule", type=int, default=110)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--chat", action="store_true", help="Apply tokenizer chat template if available")
    args = ap.parse_args()

    from mlx_lm import load

    model, tokenizer = load(args.model)

    if args.what in ("all", "table"):
        probe_table(model, tokenizer, args.rule, seed=args.seed, chat=args.chat)
    if args.what in ("all", "rollout"):
        probe_rollout(model, tokenizer, args.rule, args.width, args.steps, seed=args.seed, chat=args.chat)
    if args.what in ("all", "normal"):
        probe_normal(model, tokenizer, seed=args.seed + 7, chat=args.chat)


if __name__ == "__main__":
    main()
