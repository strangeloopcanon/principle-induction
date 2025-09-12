# Getting Started

Prerequisites
- macOS on Apple Silicon recommended (MLX optimized).
- Python 3.9+.

Install
- `python -m pip install -r requirements.txt`
- Optional dev tools: `python -m pip install -e .[dev]` or `pip install -r dev-requirements.txt`

Model (MLX)
- Convert a Hugging Face model to MLX once:
  - `python -c "from mlx_gen_parity.interop import convert_hf_to_mlx; convert_hf_to_mlx('Qwen/Qwen2.5-0.5B', quantize=False, local_out='mlx_qwen2_0_5b')"`

Run GSPO
- ECA: `make run-gspo-eca MODEL=./mlx_qwen2_0_5b`
- Life: `make run-gspo-life MODEL=./mlx_qwen2_0_5b`

Config-driven
- `python scripts/run_from_config.py --config configs/rl_eca.yaml`

Tests
- `pytest -q`

