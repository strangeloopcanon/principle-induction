# Principle Induction (CA + RL on MLX)

![CI](https://github.com/strangeloopcanon/principle-induction/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)

This repo demonstrates principle induction on Cellular Automata via two levers:
- Execution-in-the-loop with explicit “tools” (deterministic simulators).
- Rule-gleaning with RL on structured outputs (ECA rule numbers; Life B/S sets).

It provides deterministic CA tools, small RL environments, and GSPO/GRPO training loops wired to MLX and mlx-gen-parity. See `TODO.md` for the full roadmap.

## Quick Start

Install dependencies:
- `python -m pip install -r requirements.txt`
  - Optional dev tools: `python -m pip install -e .[dev]` or `pip install -r dev-requirements.txt`

(Optional) Convert an HF model to MLX once, e.g. Qwen 0.5–0.6B:
- Python: `python -c "from mlx_gen_parity.interop import convert_hf_to_mlx; convert_hf_to_mlx('Qwen/Qwen2.5-0.5B', quantize=False, local_out='mlx_qwen2_0_5b')"`
- CLI: `mlx_lm.convert --hf-path Qwen/Qwen2.5-0.5B --mlx-path mlx_qwen2_0_5b`

Run GSPO on ECA:
- `make run-gspo-eca MODEL=./mlx_qwen2_0_5b`

Run GSPO on Life:
- `make run-gspo-life MODEL=./mlx_qwen2_0_5b`

Config-driven run:
- `python scripts/run_from_config.py --config configs/rl_eca.yaml`
  - Or installed CLI: `pi-run --config configs/rl_eca.yaml`

Run tests:
- `pytest -q`

Minimal prediction-only baseline (demo)
- ECA: `make train-eca`
- Life: `make train-life`
- Notes: this is a tiny decoder-only Transformer in MLX that learns to predict next frames given the previous frame as context. It’s a simple baseline to validate datasets and training.

Cheat sheet (GSPO flags)
- `--samples K` group size; `--eta` group temperature
- `--adv-norm {softmax,zscore,rank,baseline}` advantage normalization
- `--beta` KL weight; `--beta-schedule {fixed,target}`; `--target-kl`
- `--ema-ref-decay α` EMA reference policy update (0 disables)

## Layout
- `tools/` — deterministic simulators: Life (B3/S23), ECA
- `rl/envs/` — gym-like envs for ECA/Life parameter inference
- `rl/algo/` — GRPO and GSPO updates (MLX)
- `rl/prompts.py` — prompt formatting + parsing helpers
- `datasets/` — generators for pairs/trajectories (Life/ECA)
- `models/` — minimal decoder-only Transformer (baseline)
- `scripts/` — demos and runners
- `tests/` — unit tests (7+)
- `legacy/` — older exploratory artifacts (kept for reference)

## Documentation
- See `docs/` for the docs site:
  - [Getting Started](docs/getting-started.md)
  - [Algorithms](docs/algorithms.md)
  - [Configs](docs/configs.md)
  - [Troubleshooting](docs/troubleshooting.md)

## Algorithms (brief)
- GRPO (reward-weighted NLL): single action update with REINFORCE.
- GSPO (grouped): sample K actions per prompt; minimize sum_i q_i·NLL_i + β·KL(pi||pref), with:
  - q = softmax(η·r) or advantage-normalized (z-score, rank, baseline).
  - β fixed or scheduled to a target KL.
  - Optional EMA of the reference policy.

See flags in `scripts/run_grpo_env.py` and examples in `configs/`.

## Notes
- Uses toroidal wrap consistently across tools/envs/datasets.
- mlx-gen-parity provides generation + training utilities; LoRA and SoftPrompt hooks are supported.

## Development
- Contrib guide: see `CONTRIBUTING.md`; Code of Conduct: `CODE_OF_CONDUCT.md`.
- Install editable: `make install`
- Lint/format (optional): `make lint` / `make format`

## License
- Please choose a license (MIT/Apache-2.0 are common). Add a `LICENSE` file before publishing.
Apache 2.0 — see `LICENSE` and `NOTICE`.

## Next Steps
- Scheduling for KL (β) and η.
- Add grouped batch updates for throughput; optional W&B logging.
- Mini-transformer baseline (prediction-only) and data loaders.
