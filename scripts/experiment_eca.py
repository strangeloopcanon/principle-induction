from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Tuple, List

import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.eca import rule_number_to_table, eca_step
from rl.prompts import format_eca_rollout_obs
from scripts.text_utils import clean_llm_output, apply_chat_template, greedy_binary_completion, greedy_decimal_completion
try:
    from third_party.mlx_gen_compat import GenerationConfig, generate
except Exception:
    GenerationConfig = None
    generate = None


def generate_text(model, tokenizer, prompt: str, prefix: str, max_tokens: int, temp: float, top_p: float, seed: int, *, chat: bool = False) -> str:
    prompt = apply_chat_template(tokenizer, prompt, chat=chat)
    if generate is None or GenerationConfig is None:
        raise RuntimeError("No generation backend available. Please install mlx-genkit or mlx-gen-parity.")
    cfg = GenerationConfig(max_tokens=max_tokens, temperature=temp, top_p=top_p, seed=seed)
    out = generate(model, tokenizer, prompt, cfg)
    raw = out.get("text") or out.get("texts", [""])[0]
    return clean_llm_output(raw)


def eval_rule_table(model, tokenizer, rule: int, seed: int, *, chat: bool = False) -> Dict[str, Any]:
    """Ask for ECA truth table bits and compare to ground truth.

    We expect TABLE=<8 bits> in index order i=(l<<2)|(c<<1)|r, i=0..7 (000..111).
    """
    true_tbl = rule_number_to_table(rule)
    lines = [
        f"Describe the Elementary Cellular Automaton Rule {rule}.",
        "Provide the 8-bit truth table as TABLE=<bits> for neighborhoods (l,c,r) in order 000,001,010,011,100,101,110,111.",
        "Only output TABLE=... on the last line.",
        "",
        "TABLE=",
    ]
    prompt = "\n".join(lines)
    digits = greedy_decimal_completion(model, tokenizer, prompt, chat=chat, max_digits=3)
    try:
        rule_pred = int(digits)
    except ValueError:
        rule_pred = 0
    rule_pred = max(0, min(255, rule_pred))
    pred_tbl = rule_number_to_table(rule_pred)
    pred_bits = pred_tbl
    acc = float(np.mean(pred_bits == true_tbl))
    return {
        "text": f"RULE={rule_pred}",
        "acc": acc,
        "pred_bits": ''.join(map(str, pred_bits.tolist())),
        "true_bits": ''.join(map(str, true_tbl.tolist())),
        "rule_pred": rule_pred,
    }


def eval_rollout(model, tokenizer, rule: int, width: int, steps: int, seed: int, *, chat: bool = False) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    x0 = rng.integers(0, 2, size=(width,), dtype=np.uint8)
    prompt = format_eca_rollout_obs(x0, rule=rule, steps=steps)
    digits = greedy_binary_completion(model, tokenizer, prompt, chat=chat, digits=width)
    txt = "Y=" + digits
    bits = np.array([int(c) for c in digits], dtype=np.uint8)
    canon = txt

    # Ground truth rollout
    y = x0.copy()
    for _ in range(steps):
        y = eca_step(y, rule_number=rule)
    acc = float(np.mean(bits == y))
    return {"text": txt, "canonical": canon, "acc": acc, "x0": ''.join(map(str, x0.tolist())), "pred": ''.join(map(str, bits.tolist())), "true": ''.join(map(str, y.tolist()))}


def eval_rollouts(model, tokenizer, rule: int, width: int, horizons: List[int], seed: int, *, chat: bool = False) -> Dict[str, Any]:
    out: Dict[str, Any] = {"by_h": {}, "mean_acc": 0.0}
    accs = []
    for i, h in enumerate(horizons):
        res = eval_rollout(model, tokenizer, rule=rule, width=width, steps=h, seed=seed + i, chat=chat)
        out["by_h"][str(h)] = {"acc": res["acc"], "text": res["text"], "canonical": res["canonical"], "x0": res["x0"], "pred": res["pred"], "true": res["true"]}
        accs.append(res["acc"])
    out["mean_acc"] = float(np.mean(accs)) if accs else 0.0
    return out


def eval_rollouts_random(model, tokenizer, rule: int, width: int, h_min: int, h_max: int, samples: int, seed: int, *, chat: bool = False) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    horizons: List[int] = []
    accs: List[float] = []
    details: List[Dict[str, Any]] = []
    for i in range(max(0, int(samples))):
        h = int(rng.integers(int(h_min), int(h_max) + 1))
        res = eval_rollout(model, tokenizer, rule=rule, width=width, steps=h, seed=seed + 1000 + i, chat=chat)
        horizons.append(h)
        accs.append(res["acc"])
        details.append({"h": h, "acc": res["acc"], "canonical": res["canonical"], "x0": res["x0"], "pred": res["pred"], "true": res["true"]})
    return {"min": int(h_min), "max": int(h_max), "samples": int(samples), "mean_acc": float(np.mean(accs)) if accs else 0.0, "horizons": horizons, "details": details}


def eval_normal_questions(model, tokenizer, seed: int, temp: float = 0.2, top_p: float = 0.95, max_tokens: int = 128, *, chat: bool = False) -> Dict[str, Any]:
    """Lightweight general-ability probe: sample answers to generic questions pre/post.

    Returns a dict with a few Q/A pairs. This is qualitative; it helps spot drift.
    """
    rng = np.random.default_rng(seed)
    questions = [
        "Explain cellular automata to a high school student in 3 sentences.",
        "List 3 differences between breadth-first search and depth-first search.",
        "Write a short Python function to check if a string is a palindrome.",
        "Summarize the causes of the seasons on Earth in 2 sentences.",
        "What is overfitting in machine learning? Provide a one-line definition and one mitigation.",
        "Convert 72°F to Celsius and show the formula.",
    ]
    samples = []
    for i, q in enumerate(questions):
        prompt = q + "\n\nAnswer:"
        txt = generate_text(
            model,
            tokenizer,
            prompt,
            prefix="",
            max_tokens=max_tokens,
            temp=float(temp),
            top_p=float(top_p),
            seed=int(rng.integers(10_000_000)),
            chat=chat,
        )
        samples.append({"q": q, "a": txt.strip()})
    return {"count": len(samples), "samples": samples}


def gspo_train_inplace(
    model,
    ref_model,
    tokenizer,
    *,
    env: str,
    width: int,
    height: int,
    pairs: int,
    steps: int,
    eta: float,
    beta: float,
    temp: float,
    top_p: float,
    samples: int,
    seed: int,
    rule_for_rollout: int | None = None,
    rollout_horizon: int = 10,
    train_horizons: List[int] | None = None,
    train_h_min: int | None = None,
    train_h_max: int | None = None,
    reward_mode: str = "acc",
    chat: bool = False,
) -> None:
    """Run a short GSPO training loop in-place, updating `model` weights."""
    from mlx.optimizers import AdamW
    from rl.envs import ECAParamEnv, LifeParamEnv, ECARolloutEnv
    from rl.prompts import (
        format_eca_obs,
        parse_eca_action,
        format_life_obs,
        parse_life_action,
        format_eca_rollout_obs,
        parse_eca_rollout_action,
    )
    from rl.algo import gspo_group_step

    opt = AdamW(learning_rate=5e-5)
    rng = np.random.default_rng(seed)

    if env == "eca":
        ev = ECAParamEnv(width=width, num_pairs=pairs, rule_number=None, max_attempts=1, reward_mode=reward_mode, seed=seed)
    elif env == "eca_rollout":
        ev = ECARolloutEnv(width=width, steps=rollout_horizon, rule_number=rule_for_rollout, max_attempts=1, reward_mode=reward_mode, seed=seed)
    else:
        ev = LifeParamEnv(height=height, width=width, num_pairs=pairs, born_set={3}, survive_set={2, 3}, max_attempts=1, reward_mode=reward_mode, seed=seed)

    for step_idx in range(steps):
        if env == "eca_rollout":
            # Sample a training horizon and set it on the env
            if (train_h_min is not None) and (train_h_max is not None):
                ev.steps = int(rng.integers(int(train_h_min), int(train_h_max) + 1))
            elif train_horizons:
                ev.steps = int(train_horizons[int(rng.integers(len(train_horizons)))])
        obs = ev.reset(seed=int(rng.integers(10_000_000)))
        if env == "eca":
            prompt_text = format_eca_obs(obs)
        elif env == "eca_rollout":
            prompt_text = format_eca_rollout_obs(obs["x0"], obs["rule"], obs["steps"])
        else:
            prompt_text = format_life_obs(obs)

        prompt = apply_chat_template(tokenizer, prompt_text, chat=chat)

        # Sample group
        from third_party.mlx_gen_compat import GenerationConfig, generate

        actions: list[str] = []
        rewards: list[float] = []
        for _ in range(samples):
            cfg = GenerationConfig(max_tokens=64, temperature=temp, top_p=top_p, seed=int(rng.integers(10_000_000)))
            out = generate(model, tokenizer, prompt, cfg)
            text_raw = out.get("text") or out.get("texts", [""])[0]
            text = clean_llm_output(text_raw)
            if env == "eca":
                digits = greedy_decimal_completion(model, tokenizer, prompt_text, chat=chat, max_digits=3)
                try:
                    action_val = int(digits)
                except ValueError:
                    action_val = 0
                action_val = max(0, min(255, action_val))
                a_txt = f"RULE={action_val}"
                res = ev.step(action_val)
                actions.append(a_txt)
            elif env == "eca_rollout":
                digits = greedy_binary_completion(model, tokenizer, prompt_text, chat=chat, digits=width)
                bits = np.array([int(c) for c in digits], dtype=np.uint8)
                a_txt = f"Y={digits}"
                res = ev.step(bits)
                actions.append(a_txt)
            else:
                b_bits, a_txt = parse_life_action(text)
                res = ev.step(b_bits)
            rewards.append(float(res.reward))

        # Group update
        loss, _ = gspo_group_step(
            model,
            ref_model,
            tokenizer,
            prompt if not chat else prompt_text,
            actions,
            rewards,
            optimizer=opt,
            eta=eta,
            beta=beta,
            adv_norm="softmax",
            hooks=[],
        )
        if step_idx % 10 == 0:
            print(f"gspo step={step_idx} loss={loss:.4f} rmean={np.mean(rewards):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--rule", type=int, default=110)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--rl-steps", type=int, default=100)
    ap.add_argument("--rl-task", choices=["eca", "eca_rollout"], default="eca", help="Which GSPO env to train on")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", type=str, default="runs/eca_experiment.json")
    # Evaluation horizons
    ap.add_argument("--eval-horizons", type=str, default="3,5,10,20", help="Comma-separated list of rollout horizons to evaluate pre/post, e.g. 3,5,10,20")
    ap.add_argument("--eval-h-min", type=int, default=None, help="If set with --eval-h-max and --eval-h-samples>0, sample random horizons in [min,max]")
    ap.add_argument("--eval-h-max", type=int, default=None)
    ap.add_argument("--eval-h-samples", type=int, default=0)
    # Training horizons
    ap.add_argument("--train-horizons", type=str, default="3,5,10", help="Comma-separated list of rollout horizons to sample during RL if rl-task=eca_rollout")
    ap.add_argument("--train-h-min", type=int, default=None, help="If set with --train-h-max, sample random horizons in [min,max] during RL for eca_rollout")
    ap.add_argument("--train-h-max", type=int, default=None)
    # Normal Q/A probe
    ap.add_argument("--eval-normal", action="store_true", help="Also run a small general-ability Q/A probe pre/post")
    ap.add_argument("--chat", action="store_true", help="Apply tokenizer chat template if available for all prompts")
    args = ap.parse_args()

    # Load model/tokenizer and a frozen reference for KL
    from mlx_lm import load

    model, tokenizer = load(args.model)
    ref_model, _ = load(args.model)

    # Parse horizons lists
    eval_h_list = [int(x) for x in str(args.eval_horizons).split(',') if x.strip()]
    train_h_list = [int(x) for x in str(args.train_horizons).split(',') if x.strip()]

    # Pre evaluation
    pre_table = eval_rule_table(model, tokenizer, args.rule, seed=args.seed, chat=args.chat)
    pre_rolls = eval_rollouts(model, tokenizer, rule=args.rule, width=args.width, horizons=eval_h_list, seed=args.seed + 1, chat=args.chat)
    pre_rolls_rand = None
    if (args.eval_h_samples or 0) > 0 and (args.eval_h_min is not None) and (args.eval_h_max is not None):
        pre_rolls_rand = eval_rollouts_random(model, tokenizer, rule=args.rule, width=args.width, h_min=args.eval_h_min, h_max=args.eval_h_max, samples=args.eval_h_samples, seed=args.seed + 100, chat=args.chat)
    pre_normal = None
    if args.eval_normal:
        pre_normal = eval_normal_questions(model, tokenizer, seed=args.seed + 11, chat=args.chat)

    # GSPO training in-place
    gspo_train_inplace(
        model,
        ref_model,
        tokenizer,
        env=args.rl_task,
        width=args.width,
        height=8,
        pairs=8,
        steps=args.rl_steps,
        eta=5.0,
        beta=0.05,
        temp=0.7,
        top_p=0.95,
        samples=4,
        seed=args.seed + 2,
        rule_for_rollout=(args.rule if args.rl_task == "eca_rollout" else None),
        rollout_horizon=args.horizon,
        train_horizons=(train_h_list if args.rl_task == "eca_rollout" else None),
        train_h_min=(args.train_h_min if args.rl_task == "eca_rollout" else None),
        train_h_max=(args.train_h_max if args.rl_task == "eca_rollout" else None),
        reward_mode="acc",
        chat=args.chat,
    )

    # Post evaluation
    post_table = eval_rule_table(model, tokenizer, args.rule, seed=args.seed + 3, chat=args.chat)
    post_rolls = eval_rollouts(model, tokenizer, rule=args.rule, width=args.width, horizons=eval_h_list, seed=args.seed + 4, chat=args.chat)
    post_rolls_rand = None
    if (args.eval_h_samples or 0) > 0 and (args.eval_h_min is not None) and (args.eval_h_max is not None):
        post_rolls_rand = eval_rollouts_random(model, tokenizer, rule=args.rule, width=args.width, h_min=args.eval_h_min, h_max=args.eval_h_max, samples=args.eval_h_samples, seed=args.seed + 200, chat=args.chat)
    post_normal = None
    if args.eval_normal:
        post_normal = eval_normal_questions(model, tokenizer, seed=args.seed + 211)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        payload = {
            "model": args.model,
            "rule": args.rule,
            "width": args.width,
            "horizon": args.horizon,
            "rl_steps": args.rl_steps,
            "rl_task": args.rl_task,
            "eval_horizons": eval_h_list,
            "train_horizons": (train_h_list if args.rl_task == "eca_rollout" else []),
            "pre": {"table": pre_table, "rollouts": pre_rolls},
            "post": {"table": post_table, "rollouts": post_rolls},
        }
        if pre_rolls_rand is not None:
            payload["pre"]["rollouts_random"] = pre_rolls_rand
        if post_rolls_rand is not None:
            payload["post"]["rollouts_random"] = post_rolls_rand
        if pre_normal is not None:
            payload["pre"]["normal"] = pre_normal
        if post_normal is not None:
            payload["post"]["normal"] = post_normal
        json.dump(payload, f, indent=2)

    print("Saved results to", args.out)


if __name__ == "__main__":
    main()
