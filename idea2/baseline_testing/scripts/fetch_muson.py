"""Fetch the public MUSON v1.0 archive for an independent data audit.

The archive is stored on Ocean only. This script does not use EgoNormia and
does not add MUSON material to Git.
"""
import hashlib
import json
import os
import zipfile
from pathlib import Path

import requests

from social_rl.common import STORE, write_json


URL = "https://github.com/MUSON-dataset/MUSON/releases/download/v1.0/MUSON.zip"
RELEASE_API = "https://api.github.com/repos/MUSON-dataset/MUSON/releases/tags/v1.0"
TARGET = STORE / "data/external/MUSON-v1.0.zip"
EXPECTED_SIZE = 1196748572


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    metadata = requests.get(RELEASE_API, timeout=30)
    metadata.raise_for_status()
    release = metadata.json()
    assets = [a for a in release["assets"] if a["name"] == "MUSON.zip"]
    if len(assets) != 1 or assets[0]["size"] != EXPECTED_SIZE:
        raise ValueError("Release asset differs from pinned MUSON v1.0 size")
    partial = TARGET.with_suffix(".part")
    if not TARGET.exists() or TARGET.stat().st_size != EXPECTED_SIZE:
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > EXPECTED_SIZE:
            partial.unlink()
            offset = 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        with requests.get(URL, headers=headers, stream=True, timeout=(30, 120)) as r:
            r.raise_for_status()
            if offset and r.status_code != 206:
                raise IOError("Server did not honor resume byte range")
            with open(partial, "ab" if offset else "wb") as f:
                for chunk in r.iter_content(4 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)
        if partial.stat().st_size != EXPECTED_SIZE:
            raise IOError(f"Incomplete MUSON archive: {partial.stat().st_size}/{EXPECTED_SIZE}")
        os.replace(partial, TARGET)
    with zipfile.ZipFile(TARGET) as z:
        bad = z.testzip()
        if bad:
            raise IOError(f"MUSON ZIP CRC failure: {bad}")
        members = len(z.namelist())
    audit = {"source": URL, "release_tag": release["tag_name"],
             "published_at": release["published_at"], "asset_size": TARGET.stat().st_size,
             "sha256": sha256(TARGET), "zip_members": members,
             "license_in_github_repo": None}
    write_json(STORE / "data/external/MUSON-v1.0-fetch-audit.json", audit)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
