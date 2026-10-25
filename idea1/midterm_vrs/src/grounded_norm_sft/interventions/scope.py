"""Exception-safe single-module bridge scope; no global dispatch mutation."""
from contextlib import contextmanager
from contextvars import ContextVar

from grounded_norm_sft.arms import require_trainable_arm
from grounded_norm_sft.constants import INTERVENTION_ARMS

_active = ContextVar("srer_active_site", default=None)


@contextmanager
def one_site(module, bridge, *, arm):
    """Temporarily replace exactly one module's forward with a tested bridge.

    The caller supplies native integration; this is not itself a native adapter.
    Scopes must cover checkpoint recomputation/backward until integration proves
    a different lifecycle safe. Never save/export while this scope is active.
    """
    require_trainable_arm(arm)
    if arm not in INTERVENTION_ARMS:
        raise ValueError("only intervention arms may install a bridge")
    if _active.get() is not None:
        raise RuntimeError("only one SRER site may be active")
    owned = "forward" in module.__dict__
    original = module.__dict__.get("forward")
    native = module.forward
    token = _active.set(module)
    try:
        module.forward = lambda *args, **kwargs: bridge(native, *args, **kwargs)
        yield module
    finally:
        if owned:
            module.forward = original
        else:
            del module.forward
        _active.reset(token)


def dispatch(native_forward, intervention_forward, *args, enabled=True, eta=0.25, protocol=None, **kwargs):
    """Disabled/eta-zero forwards call the literal supplied native callable."""
    if not enabled or eta == 0:
        return native_forward(*args, **kwargs)
    from .eta050 import require_dose
    require_dose(eta, protocol)
    return intervention_forward(*args, **kwargs)
