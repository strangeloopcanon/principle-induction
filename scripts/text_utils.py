from __future__ import annotations

import math
import re
from typing import Iterable, List

import mlx.core as mx

from third_party.mlx_gen_compat import (
    as_mx_array,
    loss_forward,
    stable_log_softmax,
)


_ASSISTANT_TAG = "<|im_start|>assistant"
_END_TAG = "<|im_end|>"


def apply_chat_template(tokenizer, prompt: str, *, chat: bool) -> str:
    if not chat or not hasattr(tokenizer, "apply_chat_template"):
        return prompt
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return prompt


def clean_llm_output(text: str) -> str:
    """Strip chat template tokens and <think> traces from model output."""

    if not isinstance(text, str):
        return text

    cleaned = text

    if _ASSISTANT_TAG in cleaned:
        _, after = cleaned.rsplit(_ASSISTANT_TAG, 1)
        if _END_TAG in after:
            after = after.split(_END_TAG, 1)[0]
        cleaned = after

    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<\|/?im_[^>]*\|>", "", cleaned)

    return cleaned.strip()


def _suffix_logprob(model, tokenizer, prefix_tokens: List[int], full_tokens: List[int], hooks=None) -> float:
    if len(full_tokens) <= 1:
        return float("-inf")
    arr = as_mx_array([full_tokens]).astype(mx.int32)
    logits = loss_forward(model, arr, hooks=hooks)[0]  # [L, V]
    logp = stable_log_softmax(logits)
    total = 0.0
    prefix_len = len(prefix_tokens)
    for idx in range(prefix_len, len(full_tokens)):
        if idx == 0:
            continue
        tok_id = full_tokens[idx]
        total += float(logp[idx - 1, tok_id].item())
    return total


def _greedy_completion(
    model,
    tokenizer,
    prompt: str,
    *,
    chat: bool,
    length: int,
    alphabet: Iterable[str],
    hooks=None,
) -> str:
    context = apply_chat_template(tokenizer, prompt, chat=chat)
    prefix_text = context
    prefix_tokens = tokenizer.encode(prefix_text)
    result = []
    for _ in range(length):
        best_char = None
        best_lp = -math.inf
        best_tokens = None
        for ch in alphabet:
            candidate_text = prefix_text + ch
            full_tokens = tokenizer.encode(candidate_text)
            if full_tokens[: len(prefix_tokens)] != prefix_tokens:
                continue
            lp = _suffix_logprob(model, tokenizer, prefix_tokens, full_tokens, hooks=hooks)
            if lp > best_lp:
                best_lp = lp
                best_char = ch
                best_tokens = full_tokens
        if best_char is None or best_tokens is None:
            default = next(iter(alphabet))
            result.append(default)
            prefix_text += default
            prefix_tokens = tokenizer.encode(prefix_text)
        else:
            result.append(best_char)
            prefix_text += best_char
            prefix_tokens = best_tokens
    return "".join(result)


def greedy_binary_completion(model, tokenizer, prompt: str, *, chat: bool, digits: int, hooks=None) -> str:
    return _greedy_completion(model, tokenizer, prompt, chat=chat, length=digits, alphabet=("0", "1"), hooks=hooks)


def greedy_decimal_completion(model, tokenizer, prompt: str, *, chat: bool, max_digits: int = 3, hooks=None) -> str:
    digits = _greedy_completion(model, tokenizer, prompt, chat=chat, length=max_digits, alphabet=tuple(str(i) for i in range(10)), hooks=hooks)
    digits = digits.lstrip()
    digits = digits.lstrip("0")
    return digits or "0"
