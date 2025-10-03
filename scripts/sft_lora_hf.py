from __future__ import annotations

"""
LoRA SFT (Hugging Face) for CA prompts.

Trains a small LoRA adapter on top of a HF CausalLM (defaults to Qwen2.5-0.5B-Instruct)
so the model learns to emit exact TABLE/RULE/Y strings for Elementary CA rules.

Why HF/LoRA here?
- Our MLX-side soft-prompt hooks don’t expose trainable weights cleanly. LoRA via
  Hugging Face + PEFT is stable and easy to control, and we can merge the adapter
  after SFT and convert the merged model to MLX for inference with `mlx_lm.convert`.

Usage (CPU is fine, just slower):

  python scripts/sft_lora_hf.py \
    --model-id Qwen/Qwen2.5-0.5B-Instruct \
    --steps 2000 --lr 2e-4 --batch 4 \
    --rules 30,90,110 --tasks table,rule \
    --width 32 --pairs 8 --rollout-min 3 --rollout-max 10 \
    --out-dir outputs/ca_lora

Then merge + convert to MLX for use with pi-probe:

  python scripts/sft_lora_hf.py --merge --out-dir outputs/ca_lora
  mlx_lm.convert --hf-path outputs/ca_lora_merged --mlx-path mlx_ca_lora

  pi-probe --model mlx_ca_lora --what table --rule 110 --chat
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from tools.eca import rule_number_to_table, eca_step
from rl.prompts import format_eca_obs


# ---------- Data synthesis ----------


def make_table_example(rule: int) -> Tuple[str, str]:
    true_bits = "".join(map(str, rule_number_to_table(rule).tolist()))
    prompt = (
        f"Describe the Elementary Cellular Automaton Rule {rule}.\n"
        "Provide the 8-bit truth table as TABLE=<bits> for neighborhoods (l,c,r) in order 000,001,010,011,100,101,110,111.\n"
        "Only output TABLE=... on the last line.\n\nTABLE="
    )
    target = f"TABLE={true_bits}"
    return prompt, target


def make_rule_example(rule: int, width: int, pairs: int, *, seed: int) -> Tuple[str, str]:
    rng = np.random.default_rng(seed)
    xs = rng.integers(0, 2, size=(pairs, width), dtype=np.uint8)
    ys = np.stack([eca_step(x, rule_number=rule) for x in xs], axis=0)
    pairs_arr = np.stack([xs, ys], axis=1)
    prompt = format_eca_obs(pairs_arr)
    target = f"RULE={rule}"
    return prompt, target


def make_rollout_example(rule: int, width: int, steps: int, *, seed: int) -> Tuple[str, str]:
    rng = np.random.default_rng(seed)
    x0 = rng.integers(0, 2, size=(width,), dtype=np.uint8)
    xbits = "".join(map(str, x0.tolist()))
    prompt = (
        "Under the Elementary Cellular Automaton with toroidal wrap,\n"
        f"apply Rule {int(rule)} for {int(steps)} steps to the starting row.\n"
        "Answer strictly as: Y=<W binary digits>\n\n"
        f"X={xbits}\n\nY="
    )
    y = x0.copy()
    for _ in range(steps):
        y = eca_step(y, rule_number=rule)
    ybits = "".join(map(str, y.tolist()))
    target = f"Y={ybits}"
    return prompt, target


@dataclass
class Example:
    prompt: str
    target: str


def build_dataset(rules: List[int], tasks: List[str], steps: int, width: int, pairs: int, rmin: int, rmax: int, seed: int) -> List[Example]:
    rng = np.random.default_rng(seed)
    out: List[Example] = []
    for i in range(steps):
        rule = int(rng.choice(rules))
        task = tasks[i % len(tasks)]
        if task == "table":
            p, t = make_table_example(rule)
        elif task == "rule":
            p, t = make_rule_example(rule, width, pairs, seed=int(rng.integers(10_000_000)))
        else:
            h = int(rng.integers(rmin, rmax + 1))
            p, t = make_rollout_example(rule, width, h, seed=int(rng.integers(10_000_000)))
        out.append(Example(p, t))
    return out


# ---------- Training ----------


def collate_batch(examples: List[Example], tokenizer) -> dict:
    # Label mask: ignore prompt tokens, supervise target tokens only
    inputs = tokenizer([ex.prompt for ex in examples], add_special_tokens=False)
    targets = tokenizer([ex.target for ex in examples], add_special_tokens=False)
    input_ids = []
    labels = []
    for inp, tgt in zip(inputs["input_ids"], targets["input_ids"]):
        ids = inp + tgt
        lab = [-100] * len(inp) + tgt
        input_ids.append(ids)
        labels.append(lab)
    # pad to max length
    max_len = max(len(x) for x in input_ids)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    for i in range(len(input_ids)):
        pad = max_len - len(input_ids[i])
        input_ids[i] = input_ids[i] + [pad_id] * pad
        labels[i] = labels[i] + [-100] * pad
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out-dir", type=str, default="outputs/ca_lora")
    ap.add_argument("--rules", type=str, default="30,90,110")
    ap.add_argument("--tasks", type=str, default="table,rule")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--rollout-min", type=int, default=3)
    ap.add_argument("--rollout-max", type=int, default=10)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--merge", action="store_true", help="Merge LoRA into base and save to <out-dir>_merged")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(args.model_id)
    model.train()

    # Apply LoRA via PEFT
    try:
        from peft import LoraConfig, get_peft_model, PeftModel
    except Exception as e:
        raise RuntimeError("Please install peft: pip install peft")

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=None,  # let PEFT pick common projection modules
    )
    model = get_peft_model(model, lora_cfg)

    # Data
    rules = [int(x) for x in args.rules.split(",") if x.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    data = build_dataset(rules, tasks, steps=args.steps, width=args.width, pairs=args.pairs,
                         rmin=args.rollout_min, rmax=args.rollout_max, seed=args.seed)
    loader = DataLoader(data, batch_size=args.batch, shuffle=True, collate_fn=lambda exs: collate_batch(exs, tokenizer))

    # Optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Train (one epoch over synthetic data)
    step = 0
    for batch in loader:
        inputs = {k: v for k, v in batch.items()}
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        optim.zero_grad(set_to_none=True)
        step += 1
        if step % 50 == 0:
            print(f"step={step} loss={loss.item():.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

    # Optional merge
    if args.merge:
        base = AutoModelForCausalLM.from_pretrained(args.model_id)
        merged = PeftModel.from_pretrained(base, args.out_dir)
        merged = merged.merge_and_unload()
        out_merged = args.out_dir.rstrip("/") + "_merged"
        os.makedirs(out_merged, exist_ok=True)
        merged.save_pretrained(out_merged)
        tokenizer.save_pretrained(out_merged)
        print("Merged model saved to", out_merged)

    # Quick eval (TABLE accuracy)
    model.eval()
    from transformers import TextStreamer
    results = {}
    with torch.no_grad():
        for r in rules:
            prompt, _ = make_table_example(r)
            ids = tokenizer([prompt], return_tensors="pt")
            out = model.generate(**ids, max_new_tokens=16, do_sample=False)
            text = tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
            # Extract 8 bits
            import re
            m = re.search(r"([01]{8})", text)
            pred = m.group(1) if m else "00000000"
            true_bits = "".join(map(str, rule_number_to_table(r).tolist()))
            acc = float(np.mean([int(a) == int(b) for a, b in zip(pred, true_bits)]))
            results[str(r)] = {"pred": pred, "true": true_bits, "acc": acc}
            print(f"EVAL RULE {r}: TABLE={pred} (true={true_bits}) acc={acc:.3f}")

    with open(os.path.join(args.out_dir, "eval_table.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

