"""Detached native endpoint evidence. These measurements never alter the operator."""
import math
import torch


@torch.no_grad()
def endpoint_record(*, site, family, ordinal, source, rows, clean_readout,
                    excluded_readout, clean_postgate, excluded_postgate,
                    clean_output, excluded_output, output, eta=.25,
                    clean_normalized=None, excluded_normalized=None, native_epsilon=None):
    if family not in ('FA', 'D') or not isinstance(site, int) or site < 0 or not isinstance(ordinal, int) or ordinal < 0:
        raise ValueError('site, family and nonnegative forward ordinal required')
    if source.dtype != torch.bool or rows.dtype != torch.bool or source.shape != rows.shape or not rows.any():
        raise ValueError('nonempty boolean rows and source provenance required')
    def checked(x):
        x = x.detach().float()
        if not torch.isfinite(x).all():
            raise FloatingPointError('nonfinite diagnostic evidence')
        return x
    def selected(x):
        return checked(x)[rows]
    def rms(x):
        return x.square().mean(-1).sqrt()
    def norm(x):
        return x.norm(dim=-1)
    def mix(a, b):
        return a + eta * (b-a)
    y, e = selected(clean_readout), selected(excluded_readout)
    ratio = norm(e) / norm(y).clamp_min(1e-8)
    result = dict(site=site, family=family, ordinal=ordinal,
                  source=source.detach().cpu().tolist(), rows=rows.detach().cpu().tolist(),
                  readout_norm_ratio=ratio, excluded_rms=rms(e))
    for name, a, b, measure in (
        ('postgate_rms', clean_postgate, excluded_postgate, rms),
        ('projected_norm', clean_output, excluded_output, norm),
    ):
        a, b = selected(a), selected(b)
        result[name] = dict(clean=measure(a), excluded=measure(b), mixture=measure(mix(a,b)))
    c, o = selected(clean_output), selected(output)
    result['projected_norm']['mixture'] = norm(o)
    result['displacement'] = (o-c).norm() / c.norm().clamp_min(1e-8)
    if family == 'D':
        if native_epsilon is None or not math.isfinite(native_epsilon) or native_epsilon <= 0:
            raise ValueError('native positive epsilon required')
        a, b = selected(clean_normalized), selected(excluded_normalized)
        result['normalized_rms'] = dict(clean=rms(a), excluded=rms(b), mixture=rms(mix(a,b)))
        result.update(native_epsilon=native_epsilon, low_ratio=ratio < 1e-3,
                      low_rms=rms(e) <= math.sqrt(native_epsilon),
                      joint_low_signal=(ratio < 1e-3) & (rms(e) <= math.sqrt(native_epsilon)))
    def plain(v):
        if isinstance(v, torch.Tensor):
            if not torch.isfinite(v).all():
                raise FloatingPointError('nonfinite derived diagnostic')
            return v.cpu().tolist()
        if isinstance(v, dict):
            return {k:plain(x) for k,x in v.items()}
        return v
    return plain(result)
