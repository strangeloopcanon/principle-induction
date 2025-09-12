.PHONY: help demo-rl-envs run-rl-eca run-rl-life run-gspo-eca run-gspo-life test lint format install

# Default MLX model path or HF id (must be MLX-formatted)
MODEL ?= ./mlx_qwen2_0_5b

help:
	@echo "Targets:"
	@echo "  demo-rl-envs     - Run quick sanity demos for RL envs"
	@echo "  run-rl-eca       - Run GRPO loop on ECA env (edit MODEL)"
	@echo "  run-rl-life      - Run GRPO loop on Life env (edit MODEL)"
	@echo "  run-gspo-eca     - Run GSPO loop on ECA env (grouped sampling)"
	@echo "  run-gspo-life    - Run GSPO loop on Life env (grouped sampling)"
	@echo "  test             - Run unit tests (pytest)"
	@echo "  lint             - Run ruff lint (optional)"
	@echo "  format           - Run black format (optional)"
	@echo "  install          - Install this package in editable mode"
	@echo "  train-eca        - Train minimal baseline on ECA"
	@echo "  train-life       - Train minimal baseline on Life"

demo-rl-envs:
	python3 scripts/demo_rl_envs.py

run-rl-eca:
	python3 scripts/run_grpo_env.py --env eca --model $(MODEL) --steps 50 --pairs 8 --width 32 --attempts 1 --reward-mode acc --soft-prompt --seed 123 --log-csv runs/eca_grpo.csv

run-rl-life:
	python3 scripts/run_grpo_env.py --env life --model $(MODEL) --steps 50 --pairs 8 --height 8 --width 8 --attempts 1 --reward-mode acc --soft-prompt --seed 123 --log-csv runs/life_grpo.csv

run-gspo-eca:
	python3 scripts/run_grpo_env.py --env eca --model $(MODEL) --steps 50 --pairs 8 --width 32 --samples 4 --eta 5.0 --beta 0.05 --temp 0.7 --top-p 0.95 --soft-prompt --seed 123 --log-csv runs/eca_gspo.csv

run-gspo-life:
	python3 scripts/run_grpo_env.py --env life --model $(MODEL) --steps 50 --pairs 8 --height 8 --width 8 --samples 4 --eta 5.0 --beta 0.05 --temp 0.7 --top-p 0.95 --soft-prompt --seed 123 --log-csv runs/life_gspo.csv

test:
	pytest -q

lint:
	ruff check . || true

format:
	black . || true

install:
	pip install -e .

train-eca:
	python3 scripts/train_baseline.py --task eca --width 32 --horizon 8 --num-seqs 128 --steps 50 --batch 32

train-life:
	python3 scripts/train_baseline.py --task life --height 16 --width 16 --horizon 8 --num-seqs 128 --steps 50 --batch 32
