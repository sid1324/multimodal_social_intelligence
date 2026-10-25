"""Content-addressed, create-only support records with verified provenance on reads."""
import json
from dataclasses import asdict
from pathlib import Path
from ..serializer import to_list
from ..hashing import canonical_json_bytes, sha256_bytes
from ..constants import SELECTOR_LAYER, FA_NUM_Q_HEADS
from .selection import select

VERSION = 'frozen_support_v1'


def digest(value):
    return sha256_bytes(canonical_json_bytes(value))


def provenance(serialized, *, model_sha256, processor_sha256, template_sha256,
               image_sha256, **alignment_ids):
    """Hashes identify immutable model/processor/template/image artifacts, not repo aliases."""
    pins = dict(model_sha256=model_sha256, processor_sha256=processor_sha256,
                template_sha256=template_sha256, image_sha256=image_sha256)
    for key, value in pins.items():
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError(f'{key} must be an immutable SHA256')
    # Preserve actual processor grid even on unsupported input; never serialize metadata/answers.
    grid = serialized.model_inputs.get('image_grid_thw')
    if hasattr(grid, 'tolist'):
        grid = grid.tolist()
    return dict(version=VERSION, **pins, example_id=serialized.example_id, task=serialized.task,
                prompt_ids=list(serialized.input_ids), prompt_sha256=serialized.prompt_sha256,
                attention_mask=to_list(serialized.model_inputs.get('attention_mask', [1]*len(serialized.input_ids))),
                image_positions=list(serialized.image_positions), num_visual_tokens=serialized.num_visual_tokens,
                grid=grid, grid_sha256=digest(grid), alignment_ids=alignment_ids,
                action_positions=list(serialized.action_token_positions),
                completed_action_row=serialized.completed_action_row,
                final_prompt_row=serialized.final_prompt_row,
                selector_layer=SELECTOR_LAYER, query_heads=FA_NUM_Q_HEADS)


def build_record(serialized, probabilities, *, pins, alignment_ids):
    """Produce a complete usable supported/clean-unsupported record from frozen reference P."""
    identity = provenance(serialized, **pins, **alignment_ids)
    support = select(serialized, probabilities, **alignment_ids)
    payload = json.loads(canonical_json_bytes(dict(provenance=identity, support=asdict(support))))
    return dict(payload=payload, sha256=digest(payload))


def write_record(directory, record):
    """No overwrite; identical reruns reuse verified bytes, changed identity has another key."""
    payload = record['payload']
    if digest(payload) != record['sha256']:
        raise ValueError('record checksum mismatch')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (digest(payload['provenance']) + '.json')
    raw = canonical_json_bytes(record)
    try:
        with path.open('xb') as stream:
            stream.write(raw)
    except FileExistsError:
        if path.read_bytes() != raw:
            raise ValueError('immutable support cache conflict')
    return path


def read_record(path, *, expected_provenance):
    record = json.loads(Path(path).read_bytes())
    if digest(record['payload']) != record['sha256']:
        raise ValueError('support cache checksum mismatch')
    if canonical_json_bytes(record['payload']['provenance']) != canonical_json_bytes(expected_provenance):
        raise ValueError('support cache provenance mismatch')
    return record
