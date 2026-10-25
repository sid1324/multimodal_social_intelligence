"""One-token next-label objective (plan A3): full-vocabulary FP32 log_softmax at the last
nonpadding prompt position; CE = -sum_c q_c log p(answer_token_c | input).

No EOS/rationale target, no classifier head, no renormalized training loss, no distillation.
All inputs are validated; nonfinite logits, targets or losses raise instead of propagating.
"""

from __future__ import annotations

import math
from typing import Sequence


class ObjectiveError(ValueError):
    pass


def last_nonpad_index(attention_mask):
    """Index of the last nonpadding position per row (right padding). All-padding rows raise."""
    import torch

    am = attention_mask.to(torch.long)
    if am.ndim != 2:
        raise ObjectiveError("attention_mask must be [B,T]")
    if bool((am.sum(dim=1) == 0).any()):
        raise ObjectiveError("attention_mask has an all-padding row")
    positions = torch.arange(am.shape[1], device=am.device).unsqueeze(0).expand_as(am)
    return torch.where(am.bool(), positions, torch.full_like(positions, -1)).max(dim=1).values


def check_answer_ids(answer_ids: Sequence[int], vocab_size: int) -> list[int]:
    ids = [int(i) for i in answer_ids]
    if not ids or len(set(ids)) != len(ids):
        raise ObjectiveError("answer IDs must be nonempty and distinct")
    if any(not 0 <= i < vocab_size for i in ids):
        raise ObjectiveError("answer ID outside vocabulary")
    return ids


def check_vote_q(q, n: int) -> list[float]:
    vals = [float(x) for x in q]
    if len(vals) != n:
        raise ObjectiveError(f"vote distribution has {len(vals)} entries, expected {n}")
    if any(not math.isfinite(x) or x < 0 for x in vals) or abs(sum(vals) - 1.0) > 1e-6:
        raise ObjectiveError("vote distribution must be finite, nonnegative and sum to 1")
    return vals


def answer_logprobs(logits_last, answer_ids: Sequence[int]):
    """logits_last: [V] or [B,V]. FP32 full-vocabulary log-probs at the answer IDs."""
    import torch

    ids = check_answer_ids(answer_ids, logits_last.shape[-1])
    if not bool(torch.isfinite(logits_last).all()):
        raise ObjectiveError("nonfinite logits")
    logp = torch.log_softmax(logits_last.float(), dim=-1)
    return logp.index_select(-1, torch.as_tensor(ids, device=logp.device))


def vote_ce(logits_last, answer_ids: Sequence[int], q):
    """Empirical vote-distribution CE for one example (logits_last: [V])."""
    import torch

    if logits_last.ndim != 1:
        raise ObjectiveError("vote_ce expects a single [V] logits row")
    qv = torch.tensor(check_vote_q(q, len(answer_ids)), dtype=torch.float32, device=logits_last.device)
    loss = -(qv * answer_logprobs(logits_last, answer_ids)).sum()
    if not bool(torch.isfinite(loss)):
        raise ObjectiveError("nonfinite loss")
    return loss


def conditional_from_full(full_logprobs):
    """Conditional label log-probs p_c / sum_valid p, plus log valid mass."""
    import torch

    log_mass = torch.logsumexp(full_logprobs, dim=-1, keepdim=True)
    return full_logprobs - log_mass, log_mass.squeeze(-1)


def final_row_logits(model, model_inputs, final_row: int, logits_to_keep: bool = False):
    """Batch-1 native forward (use_cache=False) returning the [V] logits at the final prompt row.

    logits_to_keep=True asks the native model for the last position only, avoiding a prompt-wide
    vocabulary tensor. Use it only after `logits_to_keep_parity` has passed for the pinned model,
    and only when the final prompt row is the last input position (no padding)."""
    if logits_to_keep:
        seq_len = model_inputs["input_ids"].shape[-1]
        if final_row != seq_len - 1:
            raise ObjectiveError("logits_to_keep requires the final prompt row to be the last position")
        logits = model(**model_inputs, use_cache=False, logits_to_keep=1).logits
        if logits.shape[0] != 1 or logits.shape[1] != 1:
            raise ObjectiveError("unexpected logits_to_keep output shape")
        return logits[0, -1]
    logits = model(**model_inputs, use_cache=False).logits
    if logits.shape[0] != 1:
        raise ObjectiveError("batch-1 forward expected")
    return logits[0, final_row]


def logits_to_keep_parity(model, model_inputs) -> float:
    """Max |full-logits final row - logits_to_keep final row| (FP32). Caller compares to its floor."""
    import torch

    final = model_inputs["input_ids"].shape[-1] - 1
    with torch.no_grad():
        a = final_row_logits(model, model_inputs, final, logits_to_keep=False).float()
        b = final_row_logits(model, model_inputs, final, logits_to_keep=True).float()
    return float((a - b).abs().max())
