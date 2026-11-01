"""Fetch the pinned public NormLens archive to Ocean for final-project research.

The archive is a possible secondary visual-norm training source. It is not
EgoNormia and is never used to update an EgoNormia evaluation partition.
"""
import hashlib
import os
from pathlib import Path

import requests

from social_rl.common import STORE


URL = ("https://media.githubusercontent.com/media/wade3han/normlens/"
       "2663c3240ac2f325950f5b4fe36ae196ed4d46be/normlens_dataset.zip")
SHA256 = "b92f190bca09b748f97a94b2fe058908a93fa3e4ca0025d90f324cd0ed08eba7"
TARGET = STORE / "data/external/normlens_dataset.zip"


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    if TARGET.exists() and digest(TARGET) == SHA256:
        print(f"Verified existing {TARGET}", flush=True)
        return
    partial = TARGET.with_suffix(f".{os.getpid()}.part")
    try:
        with requests.get(URL, stream=True, timeout=(30, 120)) as r:
            r.raise_for_status()
            with open(partial, "wb") as f:
                for chunk in r.iter_content(4 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)
        observed = digest(partial)
        if observed != SHA256:
            raise ValueError(f"NormLens archive SHA256 mismatch: {observed}")
        partial.replace(TARGET)
        print(f"Verified {TARGET} ({TARGET.stat().st_size} bytes)", flush=True)
    finally:
        partial.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
