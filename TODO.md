# Principle Induction on Cellular Automata — Action Plan (Repo: principle-induction)

Goal: Build a clean, reproducible codebase to demonstrate that dual-objective training with explicit tool-use and a rule-gleaner induces algorithmic generalization on Cellular Automata (CA). Deliver a from-scratch mini-Transformer baseline, an optional Qwen3-0.6B+LoRA variant, strong OOD tests, and “no-tools retention” evaluations. Older artifacts can be archived or ported.

Scope: Life (B3/S23) and ECA (Rules 30/90/110) first; optional DNA/Wave kept as ablations after core results.

---

## Repository Layout (target)

- tools/
  - life.py — canonical, deterministic Life (B3/S23), batch-friendly
  - eca.py — batched ECA step given 8-bit rule table, plus helpers
- runtime/
  - dispatcher.py — intercepts `<call>` tokens, executes tools, injects `<result>`
- datasets/
  - life.py, eca.py — deterministic generators, IID/OOD splits with fixed seeds
- models/
  - mini_transformer.py — small decoder-only backbone
  - heads.py — latent head (counts), trace head, param head, code head
- losses/
  - dual.py — L_final, L_prog, L_exec, invariants, tool-budget penalty
- rl/
  - code_pg.py — policy-gradient for code head (GRPO/REINFORCE + KL)
  - envs/ — lightweight RL envs for rule inference (ECA, Life)
- eval/
  - ood.py — size/horizon sweeps; no-tools toggle; invariant checks
- scripts/
  - train.py — CLI to run tasks; can be a thin wrapper around modules
  - eval.py — evaluation and report generation
- notebooks/
  - archive/ — move existing notebooks here; keep for exploration
- configs/ (optional)
  - *.yaml — preset configs for experiments

---

## Milestones, Tasks, and Acceptance Criteria

### M0 — Repo Grooming and Deterministic Simulators

- [ ] Create `tools/life.py` implementing canonical B3/S23 (no randomness).
  - DoD: Unit tests for still lifes, oscillators, and glider step correctness; rotation/translation equivariance checks pass within tolerance.
- [ ] Create `tools/eca.py` with batched step using an 8-bit table for radius-1 neighborhoods.
  - DoD: Unit tests for Rules 30/90/110 vs known truth tables; invariance under left/right shifts for periodic boundary.
- [ ] Decide fate of existing files: `eca.py`, `gameoflife.py`, `dna_ca.py`, `wavefn.py`, `simple.py`.
  - Plan: port core logic into `tools/` where useful; move originals to `notebooks/archive/` or `#Archive/` for reference; fix or deprecate `gameoflife.py` (non-canonical now).

### M1 — Datasets with IID/OOD Splits

- [ ] `datasets/life.py`: generators for 16×16 training, 32×32 OOD; gliders, blinkers, guns, rotations; fixed RNG seeds; rollouts 1–32 (train), 64–100 (OOD).
- [ ] `datasets/eca.py`: sequences for rules 30/90/110; variable lengths; extrapolate in space/time for OOD.
- [ ] Compact encoding: boards as tokens {0,1}; neighbor counts as {0..8}. RLE optional later.
  - DoD: Deterministic regeneration given seed; sharded writing optional; simple checksum of sample batches.

### M2 — Mini-Transformer (Prediction-Only Baseline)

- [ ] `models/mini_transformer.py`: ~120M decoder (12×512, 8 heads) with positional encodings compatible with grids flattened to sequences.
- [ ] `scripts/train.py`: `--task {life,eca}`; prediction-only mode; CE loss on next frame; logging basics.
- [ ] Metrics: IoU/accuracy at 1–10 and 50-step rollouts; size generalization 16→32.
  - DoD: Baseline converges on IID; OOD degrades in expected ways; unit tests and a short smoke-training pass succeed on CPU/GPU.

### M2.5 — RL Environments for Rule Inference

- [x] `rl/envs/eca_param_env.py`: Discrete(256) action to propose ECA rule; obs = (N,2,W) pairs; reward = delta accuracy; toroidal wrap.
- [x] `rl/envs/life_param_env.py`: MultiBinary(18) action to propose B/S sets; obs = (N,2,H,W); reward = delta accuracy; toroidal wrap.
- [x] `rl/envs/spaces.py` and `rl/envs/base.py`: minimal gym-like spaces and StepResult.
  - DoD: Envs reset/step deterministically with a fixed seed; perfect actions reach reward 1.0 cumulative (acc mode) or 0.0 residual (delta mode); quick demo runs.

### M3 — Tool Traces + Dispatcher (Execution-in-the-Loop)

- [ ] Define special tokens: `<call>`, `<result>`, `<end>`; reserve IDs in tokenizer for both backbones.
- [ ] `runtime/dispatcher.py`: deterministic execution of tool calls; inject results as compact tokens (counts 0–8 for Life, single bits for ECA steps or small tensors chunked).
- [ ] Trace supervision: teacher-forced tool traces in training batches; `L_prog` for traces; `L_exec` for final prediction conditioned on injected results.
- [ ] Update `train.py`: `--use-tools {0,1}` and schedule to anneal teacher forcing to ~50%.
  - DoD: End-to-end step produces identical results to pure simulation when the model follows the trace; dispatcher is deterministic (unit test).

### M4 — No-Tools Retention Evaluation

- [ ] Add a flag to disable tools at test time after tool-trained models.
- [ ] Measure drop in accuracy/IoU; expect small degradation if rules internalized.
  - DoD: Report with IID/OOD comparisons logged to disk; script reproduces results from a fixed seed.

### M5 — Rule-Gleaner (Parametric)

- [ ] `models/heads.py`: param head predicting Life B/S sets and ECA 8-bit table from short contexts.
- [ ] Training: supervised CE on parameters + CE on final next-frame via simulator with predicted params.
- [ ] Integrate with `train.py`: `--gleaner {none,param,code}`; switch to param head when set.
  - DoD: Exact ECA table recovery improves with context; Life B/S set recovered on held-out sequences; next-step metrics match simulator when params correct.

### M6 — Rule-Gleaner (Code) + RL

- [ ] Define a tiny whitelisted grammar/AST for step functions (no imports; fixed ops only).
- [ ] Unit test harness that validates code against a canonical set of inputs/outputs; reject unsafe or failing code.
- [ ] `rl/code_pg.py`: SFT on templates → policy-gradient (GRPO/REINFORCE) with reward = held-out next-step accuracy − λ·code length − μ·runtime; KL control vs SFT.
- [ ] Integrate: accepted code conditions the predictor; train code head only with RL, rest with supervised losses.
  - DoD: Non-trivial pass@k on unit tests; accuracy improves with code conditioning; rejection path is safe and deterministic.

### M6.5 — RL Harness via mlx-gen-parity

- [x] Integrate `mlx-gen-parity` (installed via pip) for model loading, generation, and training utilities.
- [x] Add GRPO-style single-step RL update using MLX (`rl/algo/grpo.py`), compatible with LoRA and SoftPrompt hooks.
- [x] Provide runnable script `scripts/run_grpo_env.py` to connect LLM -> env (ECA/Life) with reward-weighted NLL updates.
  - DoD: Script runs end-to-end with a local MLX model path or a valid HF `mlx-community/<model>`; prints step, reward, acc, and sampled action text.

### M6.6 — GSPO (Grouped Structured Policy Optimization)

- [x] Implement grouped update objective with reference-policy KL using MLX:
  - `rl/algo/gspo.py`: `gspo_group_step(model, ref_model, tokenizer, prompt, actions, rewards, ...)`.
  - Loss: sum_i softmax(η r_i) · NLL_i + β · KL_i, where NLL is sequence NLL over action tokens, KL is token-level KL(pi||pref) vs frozen reference.
- [x] Extend runner to sample K actions per prompt and apply GSPO update once per group.
  - `scripts/run_grpo_env.py`: flags `--samples`, `--eta`, `--beta`, `--temp`, `--top-p`; loads a frozen reference model.
  - Logs group mean reward/acc and best action text.
  - DoD: Runs with a local MLX model; group sampling works; stable updates with soft-prompt or LoRA.

### M7 — Qwen3-0.6B + LoRA (Optional but Recommended)

- [ ] Integrate HF `Qwen3-0.6B-Base` with LoRA adapters; register special tokens.
- [ ] Mirror training modes: prediction-only, tools, gleaner(param/code).
  - DoD: Replicate M2–M6 flows on Qwen; report parity or deltas vs mini-Transformer.

### M8 — Evaluation, Reproducibility, Packaging

- [ ] `eval/ood.py`: consolidated runs over sizes, horizons; invariant checks; no-tools toggle.
- [ ] Seeds/configs: YAML or CLI dumps; write reports and JSON metrics to `runs/`.
- [ ] Unit tests: Life/ECA determinism, dispatcher determinism, param recovery, tokenization invertibility, grammar sandbox.
- [ ] Optional CI hook (pytest) and a `Makefile`/shell scripts for common tasks.
  - DoD: Fresh clone → run a baseline and one tool-trained run with a single command; results match a small reference table.

---

## Technical Decisions (concrete defaults)

- Tokenization/encoding
  - Life board: flatten H×W into length HW; values in {0,1}.
  - Life neighbor counts: encode 0..8 as tokens; batch-friendly int tensors.
  - ECA: 1D bit arrays; truth table is 8-bit integer or 8 Bernoulli logits.
  - Special tokens: `<call>`, `<result>`, `<end>` reserved early to avoid remapping later.
- Boundaries
  - Life: zero or toroidal? Choose toroidal for analysis parity; document choice and keep consistent across tools and data.
  - ECA: toroidal wrap; document explicitly.
- Losses and schedules
  - `L_final` (answer), `L_prog` (trace CE), `L_exec` (answer when results injected), invariants, and tool-budget penalty.
  - Anneal teacher forcing on traces from 100% to ~50% after warm-up.
- Metrics
  - IoU/accuracy at 1–10 and 50–100 step rollouts; exact rule recovery (ECA 8-bit, Life B/S).
  - Invariant errors (e.g., Life rotation/translation equivariance); tool-ablation sensitivity.
- Logging
  - Minimal: stdout/CSV logs; optional Weights & Biases hooks behind a flag.

---

## Cleanup Plan for Existing Artifacts

- Move exploratory notebooks (`CAT_*.ipynb`, `test-1.ipynb`) to `notebooks/archive/`.
- Port useful code:
  - `eca.py` → `tools/eca.py` (batch ops, tests).
  - Replace/fix `gameoflife.py` with `tools/life.py` (canonical B3/S23, no randomness).
  - Keep `dna_ca.py`, `wavefn.py`, `simple.py` as optional ablations; don’t block core milestones.

---

## Execution Toggles (CLI)

`python scripts/train.py \
  --task {life,eca} \
  --use-tools {0,1} \
  --gleaner {none,param,code} \
  --backbone {mini,qwen3-0.6b} \
  --lora {on,off} \
  --seed 123 \
  --steps 100000 \
  --batch 64 \
  --save runs/exp1`

`python scripts/eval.py --load runs/exp1 --ood --no-tools`

---

## Risks and Mitigations

- Code-head RL instability → fall back to param-gleaner (M5) for core results; isolate RL updates with KL control and unit-tested grammar.
- Tool overuse → budget penalty and scheduled teacher forcing.
- OOD variance → multiple seeds and fixed evaluation grids.
- Performance on 32×32 → micro-batching; efficient kernels in `tools/`.

---

## What Happens Next

Once you approve, I will:

1) Stand up `tools/` (Life/ECA) with tests and replace `gameoflife.py` usage.
2) Add `datasets/` and a minimal `train.py` for prediction-only (M0–M2).
3) Implement `runtime/dispatcher.py` and tool-trace training (M3–M4).
4) Add param-gleaner (M5), then code-gleaner + RL (M6).
5) Optionally integrate Qwen3-0.6B+LoRA and replicate (M7).
6) Package evals, seeds, and scripts for reproducibility (M8).

Definitions of Done: Each milestone has explicit unit tests and a short, scripted run that produces stable metrics on a fixed seed. Final deliverable includes a small reference results table and commands to reproduce.

---

## Refactor Pass (completed)

- Extracted prompt formatting/parsing into `rl/prompts.py` for reuse and testing.
- Split `scripts/run_grpo_env.py` logic into a reusable `run(args)` function and a thin `main()`.
- Added `scripts/run_from_config.py` to launch runs from YAML with optional CLI overrides.
- Consolidated env logic and sampling; added `--ref-model` for KL anchor control.
- Centralized dependencies in `requirements.txt`.
