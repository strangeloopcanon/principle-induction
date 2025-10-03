# Getting Started

Prerequisites
- macOS on Apple Silicon recommended (MLX optimized).
- Python 3.9+.

Install
- `python -m pip install -r requirements.txt`
- Optional dev tools: `python -m pip install -e .[dev]` or `pip install -r dev-requirements.txt`

Model (MLX)
- Convert a Hugging Face model to MLX once (Qwen3‑0.6B recommended):
  - `python -c "from mlx_gen_parity.interop import convert_hf_to_mlx; convert_hf_to_mlx('Qwen/Qwen3-0.6B-Instruct', quantize=False, local_out='mlx_qwen3_0_6b')"`

Run GSPO
- ECA: `make run-gspo-eca MODEL=./mlx_qwen3_0_6b`
- Life: `make run-gspo-life MODEL=./mlx_qwen3_0_6b`

Config-driven
- `python scripts/run_from_config.py --config configs/rl_eca.yaml`

Tests
- `pytest -q`

End-to-end ECA experiment
- `pi-exp-eca --model ./mlx_qwen3_0_6b --rule 110 --width 32 --horizon 10 --rl-steps 100 --rl-task eca --out runs/eca_experiment.json`
