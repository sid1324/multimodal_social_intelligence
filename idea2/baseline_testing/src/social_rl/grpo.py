"""Group-relative clipped policy objective for the R³-FDT pilot."""

import torch


def group_advantages(rewards):
    """Population-standardized outcome rewards; constant groups have no signal."""
    values = torch.as_tensor(rewards, dtype=torch.float32)
    if values.ndim != 1 or values.numel() < 2:
        raise ValueError("GRPO needs at least two rewards from one question")
    centered = values - values.mean()
    scale = values.std(unbiased=False)
    return centered / scale if scale > 0 else torch.zeros_like(values)


def clipped_token_objective(logps, old_logps, ref_logps, advantage, clip_epsilon, kl_beta):
    """Return minimized loss, sampled KL, and fraction outside clipping range."""
    if logps.shape != old_logps.shape or logps.shape != ref_logps.shape:
        raise ValueError("Current, old, and reference token log-probs must align")
    if logps.numel() == 0 or not (0 < clip_epsilon < 1):
        raise ValueError("Completion and clip epsilon must be valid")
    ratio = torch.exp((logps - old_logps).clamp(-20, 20))
    clipped = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon)
    surrogate = torch.minimum(ratio * advantage, clipped * advantage)
    d = ref_logps - logps
    kl = torch.exp(d.clamp(-20, 20)) - d - 1
    loss = -surrogate.mean() + kl_beta * kl.mean()
    clipped_fraction = ((ratio < 1 - clip_epsilon) | (ratio > 1 + clip_epsilon)).float().mean()
    return loss, kl.mean().detach(), clipped_fraction.detach()
