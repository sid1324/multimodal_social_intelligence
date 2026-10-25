"""Isolated SRER native operators, supplied-model trainer, cache and potency CLI."""

__all__ = ["delta_endpoint", "full_attention_endpoint", "interpolate"]


def __getattr__(name):
    if name in __all__:
        from . import endpoints

        value = getattr(endpoints, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
