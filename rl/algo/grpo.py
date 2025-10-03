from __future__ import annotations

from typing import List, Optional, Tuple, Dict, Any


def grpo_step(
    model: Any,
    tokenizer: Any,
    prompt: str,
    action_text: str,
    reward: float,
    optimizer: Any,
    *,
    hooks: Optional[List[Any]] = None,
    dtype: Optional[str] = None,
    loss_scale: float = 1.0,
    label_smoothing: float = 0.0,
) -> float:
    """Single GRPO-style policy update using REINFORCE on an action string.

    Minimizes loss = reward * CE(action_tokens | prompt), i.e., reward-weighted NLL.
    Positive reward increases log-prob of the emitted action; negative reward decreases it.

    Returns the scalar loss value.
    """
    import mlx.core as mx
    import mlx.nn as nn
    try:
        from third_party.mlx_gen_compat import loss_forward as _loss_forward, as_mx_array as _as_mx_array
    except Exception:
        _loss_forward = None
        _as_mx_array = None
    # local fallback xent
    from models.mini_transformer import xent_loss  # reuse existing

    # Tokenize prompt and action
    prompt_ids = tokenizer.encode(prompt)
    action_ids = tokenizer.encode(action_text)
    # Build a single sequence: [prompt + action]
    tokens = prompt_ids + action_ids
    # Prepare labels: ignore prompt positions, supervise action positions
    ignore_index = -100
    labels = [ignore_index] * len(prompt_ids) + action_ids

    tok_arr = (_as_mx_array(tokens) if _as_mx_array is not None else mx.array(tokens)).reshape(1, -1)
    lab_arr = (_as_mx_array(labels) if _as_mx_array is not None else mx.array(labels)).reshape(1, -1)

    def loss_fn():
        logits = (_loss_forward(model, tok_arr, hooks=hooks) if _loss_forward is not None else model(tok_arr))
        # Predict next token: align logits to labels
        logits_shifted = logits[:, :-1, :]
        labels_shifted = lab_arr[:, 1:]
        ce = xent_loss(logits_shifted, labels_shifted, ignore_index=ignore_index, label_smoothing=label_smoothing)
        # Reward-weighted CE
        return ce * float(reward)

    if dtype == "bf16":
        # Mixed precision compute
        fp32_params = model.trainable_parameters()
        bf16_params = mx.tree_map(lambda p: p.astype(mx.bfloat16), fp32_params)
        model.update(bf16_params)
        val_and_grad = nn.value_and_grad(model, lambda: loss_fn() * loss_scale)
        loss, grads = val_and_grad()
        model.update(fp32_params)
        grads = mx.tree_map(lambda g: g.astype(mx.float32) / loss_scale, grads)
        loss = loss / loss_scale
    else:
        val_and_grad = nn.value_and_grad(model, loss_fn)
        loss, grads = val_and_grad()

    # Clip and apply grads
    from mlx.optimizers import clip_grad_norm

    clipped, _ = clip_grad_norm(grads, 1.0)
    optimizer.update(model, clipped)
    return float(loss.item())
