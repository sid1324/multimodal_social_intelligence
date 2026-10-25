"""Frozen layer-19 native-probability selector, independent of labels/answers."""
from dataclasses import dataclass
import numpy as np
from ..constants import SELECTOR_LAYER, FA_NUM_Q_HEADS
from ..schedules import topk_support, random_support
from .alignment import AlignmentError, align


class SystematicAlignmentFailure(RuntimeError):
    """Caller/audit identified systemic failure: stop, without a numeric fallback threshold."""


@dataclass(frozen=True)
class Support:
    supported: bool
    reason: str | None
    targeted: tuple[int, ...] = ()
    random: tuple[int, ...] = ()
    patches: tuple[int, ...] = ()
    rows: tuple[int, ...] = ()
    scores: tuple[float, ...] = ()
    clean_all_arms: bool = True


def select(serialized, probabilities, *, layer=SELECTOR_LAYER, systemic_alignment_failure=False, **alignment_ids):
    """Accept actual native P[head,query,key] or completed-row P[head,key].

    Reference acquisition must use the frozen base eval/no-grad model. No weights are
    loaded here. Native probabilities must retain their full key axis (no patch renormalization).
    """
    if systemic_alignment_failure:
        raise SystematicAlignmentFailure('systematic alignment failure; stop per frozen 5.2')
    if layer != SELECTOR_LAYER:
        raise ValueError('reference must be full-attention layer 19')
    try:
        a = align(serialized, **alignment_ids)
    except AlignmentError as exc:
        return Support(False, str(exc))
    if len(a.patches) < 2:
        return Support(False, 'fewer than two image tokens')
    native_dtype = str(getattr(probabilities, 'dtype', 'float64'))
    # Native eager softmax computes in FP32, then casts to query dtype.
    # Account for FP32 exp/division rounding and an n-term denominator sum
    # using gamma_(n+3), in addition to output dtype quantization. This only
    # validates mass; probabilities/scores are never renormalized or changed.
    eps = {'torch.bfloat16': 2**-7, 'torch.float16': 2**-10,
           'torch.float32': 2**-23, 'float32': 2**-23, 'float16': 2**-10}.get(native_dtype, 2**-52)
    n = len(serialized.input_ids)
    fp32_u = np.finfo(np.float32).eps / 2
    gamma = (n + 3) * fp32_u / (1 - (n + 3) * fp32_u)
    normalization_bound = (1 + gamma) * (1 + eps / (2 - eps)) - 1 + 16 * np.finfo(np.float64).eps
    if hasattr(probabilities, 'detach'):
        probabilities = probabilities.detach().float().cpu().numpy()
    p = np.asarray(probabilities, dtype=np.float64)
    n = len(serialized.input_ids)
    if p.shape == (FA_NUM_Q_HEADS, n, n):
        p = p[:, serialized.completed_action_row, :]
    if p.shape != (FA_NUM_Q_HEADS, n):
        raise ValueError('native probabilities require all 16 heads and full prompt key axis')
    if not np.isfinite(p).all():
        return Support(False, 'nonfinite reference probabilities')
    if (p < 0).any() or (p > 1).any() or not np.all(np.abs(p.sum(-1) - 1) <= normalization_bound):
        raise ValueError('expected native attention probabilities')
    if np.any(p[:, serialized.completed_action_row+1:] != 0):
        raise ValueError('reference probabilities violate completed-row causality')
    scores = p[:, a.patches].mean(0)
    return Support(True, None, tuple(topk_support(scores)),
                   tuple(random_support(serialized.example_id, len(scores))), a.patches,
                   a.rows, tuple(float(x) for x in scores), False)


def generic_serialized(serializer, image_id, image):
    """Use the serializer's exact generic text and shared system framing."""
    return serializer.selector_generic(image_id, image)
