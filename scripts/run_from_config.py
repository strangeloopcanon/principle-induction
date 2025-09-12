from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict

import yaml

# Ensure repo root on path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.run_grpo_env import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="YAML config file")
    # Optional overrides
    ap.add_argument("--model", type=str)
    ap.add_argument("--ref-model", type=str)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    # Build argparse.Namespace compatible object
    defaults = dict(
        env="eca",
        model=cfg.get("model"),
        ref_model=cfg.get("ref_model", ""),
        steps=cfg.get("steps", 20),
        lr=cfg.get("lr", 5e-5),
        seed=cfg.get("seed", 42),
        pairs=cfg.get("pairs", 8),
        width=cfg.get("width", 32),
        height=cfg.get("height", 8),
        attempts=cfg.get("attempts", 1),
        reward_mode=cfg.get("reward_mode", "acc"),
        samples=cfg.get("samples", 4),
        eta=cfg.get("eta", 5.0),
        beta=cfg.get("beta", 0.05),
        temp=cfg.get("temp", 0.7),
        top_p=cfg.get("top_p", 0.95),
        lora=cfg.get("lora", False),
        soft_prompt=cfg.get("soft_prompt", False),
        log_csv=cfg.get("log_csv", ""),
    )
    # Override from CLI if provided
    if args.model:
        defaults["model"] = args.model
    if args.ref_model is not None:
        defaults["ref_model"] = args.ref_model

    ns = argparse.Namespace(**defaults)
    run(ns)


if __name__ == "__main__":
    main()

