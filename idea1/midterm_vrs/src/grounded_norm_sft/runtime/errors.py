"""Runtime error types. Every stop names the gate or contract that failed."""

from __future__ import annotations


class RuntimeContractError(RuntimeError):
    """A runtime invariant from the plan failed; stop rather than approximate."""


class GateError(RuntimeContractError):
    """One or more launch gates are missing/invalid. Raised before any model load."""

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__("launch refused; missing/invalid gate(s):\n  " + "\n  ".join(self.problems))


class ArmRefused(GateError):
    pass


class PinError(RuntimeContractError):
    """Installed runtime / model files / processor do not match the immutable pins."""


class InventoryError(RuntimeContractError):
    """Module, LoRA or parameter inventory differs from the specification."""


class NonfiniteError(RuntimeContractError):
    """Nonfinite loss/logits/gradient: hard stop (plan 5, 9)."""


class MissingImageError(RuntimeContractError):
    """A referenced image file does not exist (never substituted)."""


class ImageIdentityError(RuntimeContractError):
    """Decoded RGB identity differs from the manifest hash."""


class CheckpointError(RuntimeContractError):
    """Checkpoint missing, tampered, untrusted, or inconsistent with the run."""


class ExportError(RuntimeContractError):
    """Native export/fresh-reload verification failed."""
