"""Download the pinned public release and official evaluation annotations."""
import concurrent.futures
import hashlib
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/EgoNormia"
DATA.mkdir(parents=True, exist_ok=True)

def fetch(url, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                payload = response.read()
            temp = dest.with_suffix(dest.suffix + ".part")
            temp.write_bytes(payload)
            temp.replace(dest)
            return
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)

def main():
    from huggingface_hub import HfApi
    api = HfApi()
    manifest_path = DATA / "download_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        revision = api.dataset_info("open-social-world/EgoNormia").sha
        entries = api.list_repo_tree("open-social-world/EgoNormia", repo_type="dataset", recursive=True, revision=revision)
        manifest = {"revision": revision, "files": [{"path": x.path, "size": x.size} for x in entries if hasattr(x, "size")]}
        manifest_path.write_text(json.dumps(manifest, indent=2))
    provenance_path = DATA / "annotations/provenance.json"
    github_revision = json.loads(provenance_path.read_text())["github_revision"] if provenance_path.exists() else json.load(urllib.request.urlopen("https://api.github.com/repos/Open-Social-World/EgoNormia/commits/release"))["sha"]
    for name in ["final_data.json", "verified_split.json", "final_dataset_guide.md"]:
        destination = DATA / "annotations" / name
        if not destination.exists():
            fetch(f"https://raw.githubusercontent.com/Open-Social-World/EgoNormia/{github_revision}/src/final_dataset/{name}", destination)
    if not provenance_path.exists():
        provenance_path.write_text(json.dumps({"github_revision": github_revision, "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2))
    tasks = [(x, DATA / x["path"]) for x in manifest["files"] if not (DATA / x["path"]).exists() or (DATA / x["path"]).stat().st_size != x["size"]]
    print(f"Downloading {len(tasks)} files, {sum(x['size'] for x, _ in tasks)/1e9:.2f} GB", flush=True)
    def one(task):
        entry, path = task
        fetch(f"https://huggingface.co/datasets/open-social-world/EgoNormia/resolve/{manifest['revision']}/{entry['path']}", path)
        if path.stat().st_size != entry["size"]:
            raise ValueError(f"Size mismatch: {path}")
        return entry["size"]
    total = 0
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(one, t): t[0]["path"] for t in tasks}
        for i, f in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                total += f.result()
            except Exception as e:
                errors.append({"path": futures[f], "error": str(e)})
            if i % 100 == 0 or i == len(tasks):
                print(f"{i}/{len(tasks)} files; {total/1e9:.2f} GB; errors={len(errors)}", flush=True)
    (DATA / "download_errors.json").write_text(json.dumps(errors, indent=2))
    missing = [x["path"] for x in manifest["files"] if not (DATA/x["path"]).exists() or (DATA/x["path"]).stat().st_size != x["size"]]
    verified = {"expected_files": len(manifest["files"]), "missing_or_size_mismatch": missing, "expected_bytes": sum(x["size"] for x in manifest["files"]), "revision": manifest["revision"]}
    (DATA / "download_verification.json").write_text(json.dumps(verified, indent=2))
    print(json.dumps(verified), flush=True)
    if missing:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
