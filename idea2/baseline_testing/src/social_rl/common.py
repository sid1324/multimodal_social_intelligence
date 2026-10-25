import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STORE = Path(os.environ.get("SOCIAL_RL_STORE", "/ocean/projects/cis260338p/apatawar/social_norm_rl"))

def config():
    return json.loads((ROOT / "configs/midterm.json").read_text())

def stable_hash(value, seed=777):
    return int(hashlib.sha256(f"{seed}:{value}".encode()).hexdigest(), 16)

def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)

def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)

def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
