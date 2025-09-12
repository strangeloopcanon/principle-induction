# Troubleshooting

Common issues
- Cannot load model: ensure you converted to MLX (`mlx_lm.convert` or `convert_hf_to_mlx`) and point `--model` to the MLX path.
- `ModuleNotFoundError: rl`: run from repo root or install editable: `make install`.
- Slow sampling: reduce `--samples`, `--pairs`, or use a smaller MLX model.
- Unstable KL: use `--beta-schedule target --target-kl 0.05`, and/or lower `--eta`.
- Actions not parseable: outputs constrained with `force_words_ids`; ensure the chosen model supports the tokens and adjust `--temp`/`--top-p`.

Debug tips
- Print env info: add `print(res.info)` in the runner.
- Log more: write metrics/params to CSV; group entropy and KL are informative.
- Determinism: set `--seed`; envs and sampling both use seeds.

