from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# Repo helpers
from tools.eca import rule_number_to_table, eca_step
from rl.prompts import format_eca_obs


@dataclass
class Example:
    prompt: str
    target: str


def make_table_example(rule: int) -> Example:
    bits = "".join(map(str, rule_number_to_table(rule).tolist()))
    prompt = (
        f"Describe the Elementary Cellular Automaton Rule {rule}.\n"
        "Provide the 8-bit truth table as TABLE=<bits> for neighborhoods (l,c,r) in order 000,001,010,011,100,101,110,111.\n"
        "Only output TABLE=... on the last line.\n\nTABLE="
    )
    target = f"TABLE={bits}"
    return Example(prompt, target)


def make_rule_example(rule: int, width: int, pairs: int, *, seed: int) -> Example:
    rng = np.random.default_rng(seed)
    xs = rng.integers(0, 2, size=(pairs, width), dtype=np.uint8)
    ys = np.stack([eca_step(x, rule_number=rule) for x in xs], axis=0)
    pairs_arr = np.stack([xs, ys], axis=1)
    prompt = format_eca_obs(pairs_arr)
    target = f"RULE={rule}"
    return Example(prompt, target)


def make_rollout_example(rule: int, width: int, steps: int, *, seed: int) -> Example:
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
    return Example(prompt, target)


def build_dataset(rules: List[int], tasks: List[str], steps: int, width: int, pairs: int, rmin: int, rmax: int, seed: int) -> List[Example]:
    rng = np.random.default_rng(seed)
    out: List[Example] = []
    for i in range(steps):
        rule = int(rng.choice(rules))
        task = tasks[i % len(tasks)]
        if task == "table":
            out.append(make_table_example(rule))
        elif task == "rule":
            out.append(make_rule_example(rule, width, pairs, seed=int(rng.integers(10_000_000))))
        else:
            h = int(rng.integers(rmin, rmax + 1))
            out.append(make_rollout_example(rule, width, h, seed=int(rng.integers(10_000_000))))
    return out


def collate_batch(examples: List[Example], tokenizer, vprompt_ids: List[int], *, chat: bool = False) -> dict:
    # Tokenize without chat/template; we prepend learned virtual tokens explicitly
    prompts = []
    vprompt_str = None
    if vprompt_ids:
        # Build a space-separated string of virtual tokens, e.g., "<CA_SP_0> <CA_SP_1> ..."
        vprompt_tokens = tokenizer.convert_ids_to_tokens(vprompt_ids)
        vprompt_str = " ".join(vprompt_tokens)
    for ex in examples:
        ptxt = ex.prompt
        if vprompt_str:
            ptxt = vprompt_str + "\n" + ptxt
        # Optional chat template wrap if available
        if chat and hasattr(tokenizer, "apply_chat_template"):
            try:
                ptxt = tokenizer.apply_chat_template([
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": ptxt},
                ], tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
        prompts.append(ptxt)

    # Tokenize prompts (already contain the virtual tokens text) and targets
    inputs = tokenizer(prompts, add_special_tokens=False)
    targets = tokenizer([ex.target for ex in examples], add_special_tokens=False)
    ids: List[List[int]] = []
    labs: List[List[int]] = []
    for inp, tgt in zip(inputs["input_ids"], targets["input_ids"]):
        # Only prepend once: prompts already include the virtual tokens
        full = inp + tgt
        labels = [-100] * len(inp) + tgt
        ids.append(full)
        labs.append(labels)
    max_len = max(len(x) for x in ids)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    for i in range(len(ids)):
        k = max_len - len(ids[i])
        ids[i] = ids[i] + [pad_id] * k
        labs[i] = labs[i] + [-100] * k
    input_ids = torch.tensor(ids, dtype=torch.long)
    return {
        "input_ids": input_ids,
        "labels": torch.tensor(labs, dtype=torch.long),
        "attention_mask": (input_ids != pad_id).long(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out-dir", type=str, default="runs/ca_prompt_embed")
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
    ap.add_argument("--n-virtual", type=int, default=20)
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--save-virtual-only", action="store_true", default=True,
                    help="Save only virtual-token embeddings and tokenizer; skip full model weights.")
    ap.add_argument("--meta-file", type=str, default="meta.json",
                    help="Filename for saving small metadata about training.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Add N virtual tokens and resize embeddings
    vtokens = [f"<CA_SP_{i}>" for i in range(args.n_virtual)]
    tokenizer.add_special_tokens({"additional_special_tokens": vtokens})
    model = AutoModelForCausalLM.from_pretrained(args.model_id)
    model.resize_token_embeddings(len(tokenizer))

    # IDs of the virtual tokens
    vprompt_ids = tokenizer.convert_tokens_to_ids(vtokens)

    # Freeze all params
    for p in model.parameters():
        p.requires_grad_(False)

    # Enable grads only for the embedding rows of the virtual tokens
    emb = model.get_input_embeddings()  # nn.Embedding
    weight = emb.weight  # [vocab, hidden]
    weight.requires_grad_(True)

    # Mask grads so only virtual token rows update
    mask = torch.zeros_like(weight)
    mask[vprompt_ids] = 1.0
    def grad_mask(g):
        return g * mask
    weight.register_hook(grad_mask)

    optim = torch.optim.AdamW([weight], lr=args.lr)

    # Data
    rules = [int(x) for x in args.rules.split(",") if x.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    data = build_dataset(rules, tasks, steps=args.steps, width=args.width, pairs=args.pairs,
                         rmin=args.rollout_min, rmax=args.rollout_max, seed=args.seed)
    loader = DataLoader(
        data,
        batch_size=args.batch,
        shuffle=True,
        collate_fn=lambda exs: collate_batch(exs, tokenizer, vprompt_ids, chat=args.chat),
    )

    # Train (one epoch synthetic)
    model.train()
    step = 0
    for batch in loader:
        outputs = model(input_ids=batch["input_ids"], labels=batch["labels"], attention_mask=batch["attention_mask"])
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([weight], 1.0)
        optim.step(); optim.zero_grad(set_to_none=True)
        step += 1
        if step % 50 == 0:
            print(f"step={step} loss={loss.item():.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    tokenizer.save_pretrained(args.out_dir)

    if args.save_virtual_only:
        import json as _json
        # Extract only the virtual prompt rows and save compactly
        vp_weights = weight.detach().cpu().numpy()[vprompt_ids]
        np.savez_compressed(os.path.join(args.out_dir, "virtual_prompt.npz"),
                            virtual_prompt=vp_weights,
                            virtual_ids=np.array(vprompt_ids, dtype=np.int64))
        with open(os.path.join(args.out_dir, args.meta_file), "w") as f:
            _json.dump({
                "model_id": args.model_id,
                "n_virtual": args.n_virtual,
                "virtual_tokens": vtokens,
                "save_virtual_only": True,
            }, f, indent=2)
        print(f"Saved virtual prompt to {os.path.join(args.out_dir, 'virtual_prompt.npz')} (tokenizer saved; full model not saved)")
    else:
        model.save_pretrained(args.out_dir)
        print(f"Saved full model to {args.out_dir}")

    # Quick eval (TABLE accuracy); prepend vtokens to prompt
    model.eval(); results = {}
    with torch.no_grad():
        for r in rules:
            ex = make_table_example(r)
            ptxt = " ".join(vtokens) + "\n" + ex.prompt
            if args.chat and hasattr(tokenizer, "apply_chat_template"):
                try:
                    ptxt = tokenizer.apply_chat_template([
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": ptxt},
                    ], tokenize=False, add_generation_prompt=True)
                except Exception:
                    pass
            ids = tokenizer([ptxt], return_tensors="pt")
            out = model.generate(**ids, max_new_tokens=16, do_sample=False)
            text = tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
            m = __import__("re").search(r"([01]{8})", text)
            pred = m.group(1) if m else "00000000"
            true_bits = "".join(map(str, rule_number_to_table(r).tolist()))
            acc = float(np.mean([int(a) == int(b) for a, b in zip(pred, true_bits)]))
            results[str(r)] = {"pred": pred, "true": true_bits, "acc": acc}
            print(f"EVAL RULE {r}: TABLE={pred} (true={true_bits}) acc={acc:.3f}")

    with open(os.path.join(args.out_dir, "eval_table.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
