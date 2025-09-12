# Configs

YAML fields (scripts/run_from_config.py)
- env: eca|life
- model: path or model id (MLX format)
- ref_model: optional reference policy id/path; defaults to same as model
- steps: number of GSPO steps
- pairs: number of (input, output) pairs per episode (env specific)
- width/height: grid sizes (env specific)
- attempts: attempts per episode (env specific)
- reward_mode: acc|delta (env specific)
- seed: RNG seed
- samples: K samples per prompt (GSPO group size)
- eta: weight temperature for group weights
- beta: KL weight
- beta_schedule: fixed|target
- target_kl: desired token-level KL if schedule=target
- temp: sampling temperature for action generation
- top_p: top-p sampling
- soft_prompt: true|false (enable SoftPrompt hooks)
- adv_norm: softmax|zscore|rank|baseline
- ema_ref_decay: 0.0 disables; otherwise EMA decay for ref policy
- log_csv: output CSV path

