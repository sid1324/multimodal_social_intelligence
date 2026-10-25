"""Grounded Social Norm Reasoning (plan: provenance/hybrid_norm_sft_plan_FINAL_v1.2.md).

Heavy dependencies (torch, transformers, peft, PIL, scipy, av, imagehash) are imported
lazily inside the functions that need them so that CPU unit tests run with numpy only.
"""

__version__ = "0.1.0"
PLAN_FILE = "provenance/hybrid_norm_sft_plan_FINAL_v1.2.md"
