# Contributing

Thanks for your interest! This repo is designed to be simple to run and easy to extend.

- Development setup
  - Python 3.9+ on macOS (Apple Silicon recommended for MLX acceleration).
  - Install base deps: `python -m pip install -r requirements.txt`
  - Optional dev deps: `python -m pip install -e .[dev]` or `pip install -r dev-requirements.txt`.

- Tests
  - Run `pytest -q`. Add tests next to the code you touch.

- Style
  - Keep functions small and documented. Prefer type annotations.
  - We use ruff + black (optional). Run `make lint` and `make format` locally.

- PR guidelines
  - One logical change per PR; include a brief Motivation and Summary.
  - If adding flags or public functions, update README and TODO.
  - Keep diffs minimal; don’t reformat unrelated code in the same PR.

- Issue triage
  - Please include: system info, exact commands, logs, and a minimal repro.

- Scope
  - This project focuses on CA tools, small RL envs, and GSPO/GRPO wiring on MLX.
  - Keep RL algorithm variants here; propose small reusable helpers for upstreaming to `mlx-gen-parity` as needed.

