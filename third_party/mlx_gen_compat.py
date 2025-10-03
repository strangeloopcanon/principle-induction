from __future__ import annotations

"""Compatibility shim for mlx-genkit and mlx-gen-parity.

This module provides a unified import surface for:
- GenerationConfig, generate
- apply_lora, SoftPromptHook (if available; otherwise no-ops)
- training helpers: loss_forward, sequence_logprob, token_kl (if available)
- utils: as_mx_array, stable_log_softmax, try_import_mlx, ema_update

Falls back gracefully if some symbols are missing; callers should handle None.
"""

from typing import Any, Optional


# Generation API
GenerationConfig = None  # type: ignore
generate = None  # type: ignore
apply_lora = None  # type: ignore
SoftPromptHook = None  # type: ignore

# Training helpers
loss_forward = None  # type: ignore
sequence_logprob = None  # type: ignore
token_kl = None  # type: ignore

# Utils
as_mx_array = None  # type: ignore
stable_log_softmax = None  # type: ignore
try_import_mlx = None  # type: ignore
ema_update = None  # type: ignore


_sources = []

try:
    from mlx_genkit import GenerationConfig as _GC, generate as _gen

    GenerationConfig = _GC
    generate = _gen
    _sources.append("mlx_genkit.gen")
    try:
        from mlx_genkit import apply_lora as _apply_lora, SoftPromptHook as _SoftPromptHook

        apply_lora = _apply_lora
        SoftPromptHook = _SoftPromptHook
        _sources.append("mlx_genkit.hooks")
    except Exception:
        pass
    try:
        from mlx_genkit.training import loss_forward as _lf, sequence_logprob as _slp, token_kl as _tkl

        loss_forward = _lf
        sequence_logprob = _slp
        token_kl = _tkl
        _sources.append("mlx_genkit.training")
    except Exception:
        pass
    try:
        from mlx_genkit.utils import as_mx_array as _asarr, stable_log_softmax as _sls, try_import_mlx as _tim, ema_update as _ema

        as_mx_array = _asarr
        stable_log_softmax = _sls
        try_import_mlx = _tim
        ema_update = _ema
        _sources.append("mlx_genkit.utils")
    except Exception:
        pass
except Exception:
    pass

try:
    # Fallback to mlx-gen-parity
    if GenerationConfig is None or generate is None:
        from mlx_gen_parity import GenerationConfig as _GC, generate as _gen

        GenerationConfig = _GC
        generate = _gen
        _sources.append("mlx_gen_parity.gen")
    if apply_lora is None or SoftPromptHook is None:
        from mlx_gen_parity import apply_lora as _apply_lora, SoftPromptHook as _SoftPromptHook

        apply_lora = _apply_lora
        SoftPromptHook = _SoftPromptHook
        _sources.append("mlx_gen_parity.hooks")
    if loss_forward is None or sequence_logprob is None or token_kl is None:
        from mlx_gen_parity.training import loss_forward as _lf, sequence_logprob as _slp, token_kl as _tkl

        loss_forward = _lf
        sequence_logprob = _slp
        token_kl = _tkl
        _sources.append("mlx_gen_parity.training")
    if as_mx_array is None or stable_log_softmax is None or try_import_mlx is None or ema_update is None:
        from mlx_gen_parity.utils import as_mx_array as _asarr, stable_log_softmax as _sls, try_import_mlx as _tim, ema_update as _ema

        as_mx_array = _asarr
        stable_log_softmax = _sls
        try_import_mlx = _tim
        ema_update = _ema
        _sources.append("mlx_gen_parity.utils")
except Exception:
    pass


def have_generation() -> bool:
    return GenerationConfig is not None and generate is not None


def have_hooks() -> bool:
    return apply_lora is not None and SoftPromptHook is not None


def have_training_helpers() -> bool:
    return loss_forward is not None

