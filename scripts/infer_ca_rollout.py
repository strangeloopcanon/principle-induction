import argparse
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor


class Digits01Only(LogitsProcessor):
    """Constrain generation to tokens that decode to '0' or '1'."""

    def __init__(self, tokenizer):
        allowed = []
        # Build once; ok to scan vocab given small generation lengths
        for i in range(tokenizer.vocab_size):
            s = tokenizer.decode([i], skip_special_tokens=True)
            if s == "0" or s == "1":
                allowed.append(i)
        self.allowed = torch.tensor(allowed, dtype=torch.long)

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float("-inf"))
        mask[:, self.allowed] = scores[:, self.allowed]
        return mask


def predict_rollout(model_dir: str, rule: int, steps: int, xbits: str, *,
                    constrain01: bool = True, device: str = "cpu") -> str:
    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32, device_map=device)

    # Build prompt exactly like training (no chat template; virtual tokens first)
    n_virtual = 20
    vtokens = " ".join(f"<CA_SP_{i}>" for i in range(n_virtual))
    prompt = (
        f"{vtokens}\n"
        "Under the Elementary Cellular Automaton with toroidal wrap,\n"
        f"apply Rule {int(rule)} for {int(steps)} steps to the starting row.\n"
        "Answer strictly as: Y=<W binary digits>\n\n"
        f"X={xbits}\n\n"
        "Y="
    )

    inputs = tok([prompt], return_tensors="pt", add_special_tokens=False)
    logits_proc = [Digits01Only(tok)] if constrain01 else None
    out = mdl.generate(**inputs, max_new_tokens=len(xbits), do_sample=False, logits_processor=logits_proc)
    reply = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return reply.strip()


def eca_rollout(rule: int, steps: int, xbits: str) -> str:
    # Deterministic reference using repo's ECA tools
    import numpy as np
    from tools.eca import rule_number_to_table, eca_step

    table = rule_number_to_table(rule)
    row = np.array([int(c) for c in xbits], dtype=np.uint8)
    for _ in range(steps):
        row = eca_step(row, rule_table=table)
    return "".join(map(str, row.tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=str, required=True)
    ap.add_argument("--rule", type=int, default=110)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--x", type=str, required=True, help="binary string for X")
    ap.add_argument("--no-constrain", action="store_true", help="do not restrict outputs to 0/1")
    ap.add_argument("--device", type=str, default="cpu", help="device_map for HF (e.g., cpu, mps)")
    args = ap.parse_args()

    pred = predict_rollout(args.model_dir, args.rule, args.steps, args.x, constrain01=not args.no_constrain, device=args.device)
    truth = eca_rollout(args.rule, args.steps, args.x)
    # Try to extract first 0/1 span of correct length if model echoed extras
    import re
    m = re.search(fr"([01]{{{len(args.x)}}})", pred)
    clean = m.group(1) if m else pred

    print(f"X={args.x}")
    print(f"Y_pred={clean}")
    print(f"Y_true={truth}")
    print(f"match={clean == truth}")


if __name__ == "__main__":
    main()

