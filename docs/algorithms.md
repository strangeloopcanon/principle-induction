# Algorithms: GRPO and GSPO

GRPO (single-action)
- Samples one action per prompt.
- Update: minimize reward-weighted NLL(action | prompt).
- Optional: KL regularization to a reference policy.

GSPO (grouped)
- Samples K actions per prompt.
- We minimize: sum_i q_i · NLL_i + β · KL(pi || pref)
  - q_i: group weights
    - softmax: q_i = softmax(η · r_i)
    - zscore: softmax(η · z_i) where z is standardized reward
    - rank: softmax over normalized ranks
    - baseline: mean-subtracted rewards (can be negative; handled with mean aggregation)
  - KL: token-level KL over supervised action tokens vs a reference policy.
- Reference policy options
  - Frozen copy of initial model
  - EMA of current policy with decay α (optional)
- KL scheduling
  - fixed: constant β
  - target: adjust β multiplicatively to match a target KL

Implementation hooks
- We rely on mlx-gen-parity for:
  - loss_forward (forward logits with hooks)
  - stable_log_softmax, utils, and steering hooks (LoRA, SoftPrompt)

Flags (runner)
- `--samples`: group size K
- `--eta`: weight temperature
- `--adv-norm`: softmax|zscore|rank|baseline
- `--beta`: KL weight
- `--beta-schedule`: fixed|target; `--target-kl`
- `--ema-ref-decay`: EMA decay for reference policy

