"""Locked constants from plan FINAL v1.2. Changing any value is a new experiment version."""

from __future__ import annotations

# ---------------------------------------------------------------- project / compute
PROJECT_NAME = "grounded_norm_sft"
ALLOCATION = "cis260338p"  # the ONLY authorized allocation
REMOTE_ROOT = "/ocean/projects/cis260338p/ssingh40/grounded_norm_sft"

# ---------------------------------------------------------------- backbone (inspected, not pins)
MODEL_REPO = "Qwen/Qwen3.5-9B"
MODEL_CLASS = "Qwen3_5ForConditionalGeneration"
NUM_LAYERS = 32
FULL_ATTENTION_LAYERS = (3, 7, 11, 15, 19, 23, 27, 31)
DELTA_LAYERS = tuple(i for i in range(NUM_LAYERS) if i not in FULL_ATTENTION_LAYERS)
HIDDEN_SIZE = 4096
FA_NUM_Q_HEADS = 16
FA_NUM_KV_HEADS = 4
FA_HEAD_DIM = 256
DELTA_NUM_K_HEADS = 16
DELTA_NUM_V_HEADS = 32
DELTA_HEAD_DIM = 128
DELTA_CONV_WIDTH = 4

# ---------------------------------------------------------------- adapters (section 6.1)
FA_LORA_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
DELTA_LORA_MODULES = ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj")
FA_MIXER_ATTR = "self_attn"
DELTA_MIXER_ATTR = "linear_attn"
FUSED_DELTA_NAMES = ("in_proj_qkvz", "in_proj_ba")  # presence => stop (different adapter capacity)
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.0
EXPECTED_LORA_PARAMS_PER_FA_LAYER = 491_520
EXPECTED_LORA_PARAMS_PER_DELTA_LAYER = 590_848
EXPECTED_LORA_PARAMS_TOTAL = 18_112_512

# ---------------------------------------------------------------- SRER (sections 5, 5.2, 5.3)
ETA = 0.25
SUPPORT_FRACTION = 0.25
SELECTOR_LAYER = 19
SELECTOR_GENERIC_QUERY = "Describe what is visible in this image."
SCHEDULE_SEED_BASE = 20261022
RANDOM_SUPPORT_TAG = "random_support_v1"
INTERVENTION_SCHEDULE_TAG = "intervention_schedule_v1"
POTENCY_SCHEDULE_TAG = "potency_schedule_v1"
POTENCY_SORT_PREFIX = "hybrid_potency20261022|"
POTENCY_MAX_INPUTS = 128
POTENCY_MIN_INPUTS = 64
POTENCY_FLOOR_INPUTS = 4
POTENCY_FLOOR_REPEATS = 10
GATE_HEALTH_INPUTS = 32
LOW_SIGNAL_RATIO = 1e-3
NORM_EPS_FLOOR = 1e-8
POTENCY_R_MIN = 1e-5
POTENCY_KL_MIN = 1e-4
POTENCY_FLOOR_MULTIPLIER = 100.0
INERT_UPDATE_FRACTION = 0.80
JACCARD_WARNING_MEDIAN = 0.7
HRMATCH_MAX_ETA = 0.5

# ---------------------------------------------------------------- arms
ARMS = ("B0", "B0-cal", "B1", "H-T", "H-R", "F-T", "D-T", "H-Rmatch")
TRAINABLE_ARMS = ("B1", "H-T", "H-R", "F-T", "D-T")
INTERVENTION_ARMS = ("H-T", "H-R", "F-T", "D-T")
DIAGNOSTIC_ONLY_ARMS = ("H-Rmatch",)

# ---------------------------------------------------------------- optimization (section 7, A3)
LR = 1e-4
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
WARMUP_FRACTION = 0.05
EFFECTIVE_BATCH = 16
NORMBANK_LOSS_WEIGHT = 0.25
CLEAN_WEIGHT_INTERVENTION = 0.5
PERTURBED_WEIGHT_INTERVENTION = 0.5
PILOT_SEED = 17
PAPER_SEEDS = (17, 29, 43)
ALLOWED_EPOCHS = (1, 3)

# ---------------------------------------------------------------- labels / prompts
NORMLENS_CLASSES = (0, 1, 2)
NORMLENS_CLASS_NAMES = ("Wrong", "Okay", "Impossible")
NORMLENS_LETTERS = ("A", "B", "C")
NORMBANK_CLASS_NAMES = ("taboo", "normal", "expected")
NORMBANK_LETTERS = ("D", "E", "F")
NORMBANK_LABEL_MAP = {"taboo": 0, "normal": 1, "expected": 2}

MIN_PIXELS = 65_536
MAX_PIXELS = 262_144
VISUAL_TOKENS_MIN = 64
VISUAL_TOKENS_MAX = 256
MAX_PROMPT_TOKENS = 1024

# ---------------------------------------------------------------- data / split (A2)
SPLIT_SALT = "20261021|"
SPLIT_TRAIN_UPPER = 0.70
SPLIT_DEV_UPPER = 0.85
PHASH_MAX_HAMMING = 4
SHUFFLE_SEED = 20261022
NORMBANK_SPLIT_PRIORITY = {"test": 3, "dev": 2, "train": 1}
RETENTION_SORT_PREFIX = "retention20261021|"
RETENTION_MAX = 2000

# ---------------------------------------------------------------- evaluation
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20261021
BOOTSTRAP_PERCENTILES = (2.5, 97.5)
ENTROPY_BINS = ((0.0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.0))
CALIBRATION_L2 = 0.01
CALIBRATION_GTOL_ACCEPT = 1e-8
EXPORT_LOGPROB_ABS_MIN = 1e-6
EXPORT_FLOOR_MULTIPLIER = 10.0
EXPORT_FLOOR_STOP = 1e-4

# ---------------------------------------------------------------- EgoNormia (A5, A5.1, App. B)
# Production Ego source (user correction, 2026-10-22; supersedes plan v1.2 A1's inspected tuple).
EGO_SOURCE_SCHEMA = "egonormia_final_data_1853_v1"
EGO_SOURCE_FILE = "analysis1_verified/data/egonormia/final_data.json"
EGO_SOURCE_SHA256 = "0f19bc41374bf13b359a55fb65b4ba3e5db5a217c31727bc9ff5b1fa6d32e225"
EGO_SOURCE_RECORDS = 1853
EGO_MEDIA_REVISION = "2937df7fa96d8515417e7b5122fbf8eaeeaeee86"
EGO_PROFILE_IDS = 5
# Historical only (plan v1.2 A1 inspection: 1,743-record HF parquet). Never a production source.
EGO_HISTORICAL_HF_REVISION = "bc43d57b09af2ee694cb99a555e101ef197cfa83"
EGO_HISTORICAL_HF_RECORDS = 1743
EGO_EVALUATOR_BLOB = "788f4a1a7b678312a2042e8cda37911882954416"
EGO_SCORER_BLOB = "72d7c204d0b9c2252210c5a0de00440b67e65cd2"
EGO_SORT_PREFIX = "ego_midterm20261021|"
EGO_MAX_ITEMS = 400
EGO_DIAG_COHORT = 32
EGO_NUM_OPTIONS = 5
EGO_MAX_NEW_TOKENS = 256
EGO_FRAMES = 8
EGO_TILE = 256
EGO_SHEET_W = 1024
EGO_SHEET_H = 512
EGO_MAX_PIXELS = 524_288
EGO_EXPECTED_VISUAL_TOKENS = 512
EGO_CANDIDATE_MASS_TOL = 1e-6
EGO_INCIDENTAL_EXPOSURE = (
    "01ccf6b0-5e95-4b18-acd4-a8cc1fd31ddf_248-81",
    "01ccf6b0-5e95-4b18-acd4-a8cc1fd31ddf_849-71",
)

# ---------------------------------------------------------------- numeric tolerances (9.B.4)
TOL_FP32_OUT = dict(atol=2e-5, rtol=2e-4)
TOL_FP32_GRAD = dict(atol=5e-5, rtol=5e-4)
TOL_FP64_FD = dict(eps=1e-5, atol=1e-5, rtol=1e-3)
TOL_BF16_OUT = dict(atol=0.02, rtol=0.02, nrms=0.02)
TOL_BF16_GRAD = dict(atol=0.05, rtol=0.05, nrms=0.05)


def layer_family(layer: int) -> str:
    if not 0 <= layer < NUM_LAYERS:
        raise ValueError(f"layer {layer} out of range")
    return "FA" if layer in FULL_ATTENTION_LAYERS else "D"
