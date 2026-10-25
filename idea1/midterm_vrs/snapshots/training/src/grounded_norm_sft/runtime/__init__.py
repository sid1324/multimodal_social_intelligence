"""Baseline runtime slice (plan FINAL v1.2, sections 6-9, A3, A6): B0 native evaluation and B1
clean SFT for Qwen3.5-9B. Entry point: ``python -m grounded_norm_sft.runtime COMMAND``.

In scope: immutable local-only model/processor load with source/config/backend assertions,
all-32-layer native-mixer LoRA (exact 18,112,512 FP32 trainable parameters on a BF16 base),
the B1 objective/accumulation/optimizer/scheduler, pickle-free checkpoint/resume, epoch-boundary
dev selection, native evaluation artifacts and fresh-reload export verification, plus launch
gates that refuse before any model load.

Out of scope (refused, not stubbed): intervention arms H-T/H-R/F-T/D-T (need reviewed SRER
integration and Seal B), H-Rmatch (stopped by plan 10.2), and B0-cal (the existing
``grounded_norm_sft.metrics.calibration`` module; no forward). No EOS/rationale targets exist.

Nothing here has been executed against real weights; see docs/BASELINE_RUNTIME.md.
"""

RUNTIME_VERSION = "baseline_runtime_v1"
