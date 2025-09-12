from __future__ import annotations

from typing import List, Optional, Tuple, Any


def _batch_tokens_and_labels(tokenizer, prompt: str, actions: List[str], ignore_index: int = -100):
    """Build batched token arrays and label masks for action-only supervision.

    Returns (tokens, labels, prompt_len), where labels use ignore_index for
    prompt positions and action tokens are supervised.
    """
    prompt_ids = tokenizer.encode(prompt)
    pL = len(prompt_ids)
    action_ids = [tokenizer.encode(a) for a in actions]
    max_a = max(len(a) for a in action_ids) if action_ids else 0
    B = len(actions)
    # Sequence = prompt + action; allow variable action lengths by right-padding labels with ignore_index
    tokens = []
    labels = []
    for a in action_ids:
        seq = prompt_ids + a
        # Pad sequence to same length across batch
        seq_pad = seq + [ignore_index] * (pL + max_a - len(seq))
        # labels: ignore for prompt ids, then the action ids; pad to align
        lab = [ignore_index] * pL + a + [ignore_index] * (max_a - len(a))
        tokens.append(seq_pad)
        labels.append(lab)
    return tokens, labels, pL


def _sequence_logprob(model, tokens_arr, labels_arr, hooks=None):
    """Return per-sample sum log-prob over supervised labels (action tokens)."""
    import mlx.core as mx
    from mlx_gen_parity.training import loss_forward
    from mlx_gen_parity.utils import stable_log_softmax

    logits = loss_forward(model, tokens_arr, hooks=hooks)  # [B, L, V]
    # Shift for next-token prediction
    logits = logits[:, :-1, :]
    labels = labels_arr[:, 1:]  # align with next-token targets

    logp = stable_log_softmax(logits)
    B, T = labels.shape
    # Gather log-probs at label positions (ignore_index filtered later)
    labels_clamped = mx.maximum(labels, 0)
    chosen = mx.take_along_axis(logp, labels_clamped.reshape(B, T, 1), axis=-1).reshape(B, T)
    mask = (labels != -100)
    chosen = chosen * mask
    # Sum over time
    seq_logprob = chosen.sum(axis=1)  # [B]
    return seq_logprob


def _token_kl(model, ref_model, tokens_arr, labels_arr, hooks=None):
    """Compute per-sample token-level KL(pi||pref) averaged over supervised positions.

    Returns kl_per_seq with shape [B].
    """
    import mlx.core as mx
    from mlx_gen_parity.training import loss_forward
    from mlx_gen_parity.utils import stable_log_softmax

    logits = loss_forward(model, tokens_arr, hooks=hooks)[:, :-1, :]
    ref_logits = loss_forward(ref_model, tokens_arr, hooks=None)[:, :-1, :]
    labels = labels_arr[:, 1:]
    mask = (labels != -100)

    logp = stable_log_softmax(logits)
    logq = stable_log_softmax(ref_logits)
    p = mx.exp(logp)
    kl = (p * (logp - logq)).sum(axis=-1)  # [B, T]
    kl = kl * mask
    # Mean over supervised positions to avoid length bias; add epsilon to denom
    denom = mask.sum(axis=1).astype(mx.float32) + 1e-8
    kl_seq = kl.sum(axis=1) / denom
    return kl_seq


def compute_group_weights(
    rewards: List[float],
    *,
    mode: str = "softmax",
    eta: float = 1.0,
) -> Tuple[Any, Any]:
    """Compute group weights for GSPO.

    Returns (weights, extras) where weights is a numpy array shape [K], sums to 1,
    and extras may contain diagnostics like ranks or standardized rewards.
    """
    import numpy as np

    r = np.asarray(rewards, dtype=np.float32)
    K = r.shape[0]
    eps = 1e-8
    extras = {}
    if mode == "softmax":
        w = np.exp(eta * (r - r.max()))
        q = w / (w.sum() + eps)
    elif mode in ("zscore", "standardize"):
        mu = float(r.mean())
        sigma = float(r.std())
        sigma = sigma if sigma > 1e-6 else 1.0
        z = (r - mu) / sigma
        extras["z"] = z
        w = np.exp(eta * (z - z.max()))
        q = w / (w.sum() + eps)
    elif mode == "rank":
        ranks = r.argsort().argsort().astype(np.float32)  # 0..K-1
        ranks = ranks / max(1.0, K - 1)
        extras["ranks"] = ranks
        w = np.exp(eta * (ranks - ranks.max()))
        q = w / (w.sum() + eps)
    elif mode == "baseline":
        # Baseline-subtracted advantages; allow negatives. Normalize by L1 for scale.
        adv = r - float(r.mean())
        extras["adv"] = adv
        denom = np.sum(np.abs(adv)) + eps
        q = adv / denom  # can be negative; caller must handle usage
    else:
        raise ValueError(f"Unknown adv normalization mode: {mode}")
    return q, extras


def gspo_group_step(
    model: Any,
    ref_model: Any,
    tokenizer: Any,
    prompt: str,
    actions: List[str],
    rewards: List[float],
    optimizer: Any,
    *,
    eta: float = 1.0,
    beta: float = 0.05,
    adv_norm: str = "softmax",  # softmax, zscore, rank, baseline
    hooks: Optional[List[Any]] = None,
    dtype: Optional[str] = None,
    loss_scale: float = 1.0,
) -> Tuple[float, dict]:
    """GSPO-like grouped update.

    - Build a group of K candidate actions for the same prompt.
    - Compute normalized target weights q = softmax(eta * rewards).
    - Minimize L = sum_i q_i * NLL_i + beta * KL_i, where
        NLL_i = -log p_theta(action_i | prompt),
        KL_i = mean token-level KL(pi_theta || pi_ref) over action tokens.

    Returns (loss_value, metrics_dict).
    """
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_gen_parity.utils import as_mx_array

    assert len(actions) == len(rewards) and len(actions) > 0

    tokens, labels, _ = _batch_tokens_and_labels(tokenizer, prompt, actions)
    tokens_arr = as_mx_array(tokens).astype(mx.int32)
    labels_arr = as_mx_array(labels).astype(mx.int32)

    # Target weights q over the group
    import numpy as np

    q, extras = compute_group_weights(rewards, mode=adv_norm, eta=eta)
    q_arr = as_mx_array(q).reshape(-1)

    def loss_fn():
        # Negative sequence log-prob for each candidate
        seq_logp = _sequence_logprob(model, tokens_arr, labels_arr, hooks=hooks)  # [B]
        nll = -seq_logp  # [B]
        # KL penalty per candidate
        kl = _token_kl(model, ref_model, tokens_arr, labels_arr, hooks=hooks)  # [B]
        # Weighted sum
        if adv_norm == "baseline":
            # q_arr may be negative; use mean to stabilize scale
            nll_term = (nll * q_arr).mean()
            kl_term = kl.mean()
        else:
            # q_arr is a proper distribution
            nll_term = (nll * q_arr).sum()
            kl_term = (kl * q_arr).sum()
        return nll_term + (beta * kl_term)

    if dtype == "bf16":
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

    from mlx.optimizers import clip_grad_norm

    clipped, _ = clip_grad_norm(grads, 1.0)
    optimizer.update(model, clipped)
    # Diagnostics: return mean KL and weights entropy
    from mlx_gen_parity.utils import try_import_mlx
    mx, _ = try_import_mlx()
    # Recompute for metrics (reuse logits to avoid double compute? acceptable small overhead for now)
    kl_seq = _token_kl(model, ref_model, tokens_arr, labels_arr, hooks=hooks)
    # Convert q to numpy for metrics
    q_np = q if isinstance(q, np.ndarray) else np.array(q)
    q_np = np.abs(q_np)
    q_np = q_np / (q_np.sum() + 1e-8)
    entropy = float(-(q_np * (np.log(q_np + 1e-8))).sum())
    metrics = dict(
        loss=float(loss.item()),
        kl_mean=float(kl_seq.mean().item()),
        nll_mean=float((_sequence_logprob(model, tokens_arr, labels_arr, hooks=hooks) * (-1)).mean().item()),
        weight_entropy=entropy,
        adv_mode=adv_norm,
    )
    return float(loss.item()), metrics
