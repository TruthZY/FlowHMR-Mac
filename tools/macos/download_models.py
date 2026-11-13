"""Download FlowHMR models on macOS, including the login-gated SMPL-H / MANO pack.

The licensed body models are read from the Hugging Face mirror copy of
``lithiumice/models_hub/4_SMPLhub`` instead of the MANO sign-in page:

    python tools/macos/download_models.py                     # everything
    python tools/macos/download_models.py --only sam smplh    # selected groups
    python tools/macos/download_models.py --skip vggt         # keep VGGT for later

Every file is resumable; pinned files are verified against the SHA-256 that
``flowhmr.model_sources`` publishes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flowhmr import model_files as files
from flowhmr import model_sources as sources


SMPLHUB = "https://hf-mirror.com/lithiumice/models_hub/resolve/main/4_SMPLhub"
SMPLHUB_FILES = (
    # HF mirror path, destination, minimum size
    ("smplh/neutral/model.npz", files.SMPLH_FILES["neutral"], 60 << 20),
    ("SMPLH/X_pkl/SMPLH_male.pkl", files.SMPLH_FILES["male"], 60 << 20),
    ("SMPLH/X_pkl/SMPLH_female.pkl", files.SMPLH_FILES["female"], 60 << 20),
    ("MANO/pkl/MANO_LEFT.pkl", files.MANO_FILES[0], 1 << 20),
    ("MANO/pkl/MANO_RIGHT.pkl", files.MANO_FILES[1], 1 << 20),
)
GROUPS = ("flowhmr", "yolox", "regressor", "sam", "vggt", "smplh", "mano")


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, target, *, size=None, sha256=None, minimum=1):
    destination = files.resolve_model(target)
    if destination.is_file():
        present = destination.stat().st_size
        if (size is None or present == size) and present >= minimum:
            if sha256 is None or checksum(destination) == sha256:
                print(f"Present: {target}", flush=True)
                return
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_suffix(destination.suffix + ".part")
    offset = part.stat().st_size if part.exists() else 0
    request = urllib.request.Request(
        url, headers={"Accept-Encoding": "identity", "User-Agent": "FlowHMR downloader",
                      **({"Range": f"bytes={offset}-"} if offset else {})})
    print(f"Downloading: {target}" + (f" (resume at {offset / (1 << 20):.0f} MiB)" if offset else ""),
          flush=True)
    with urllib.request.urlopen(request, timeout=600) as response:
        append = offset > 0 and response.status == 206
        if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
            raise RuntimeError("Invalid download resume response")
        written = offset if append else 0
        next_report = written + 256 * 1024 * 1024
        with part.open("ab" if append else "wb") as output:
            while chunk := response.read(8 * 1024 * 1024):
                output.write(chunk)
                written += len(chunk)
                if written >= next_report:
                    print(f"  {target}: {written / (1 << 20):.0f} MiB", flush=True)
                    next_report = written + 256 * 1024 * 1024
    actual = part.stat().st_size
    if size is not None and actual != size:
        raise RuntimeError(f"size mismatch for {target}: {actual} != {size}")
    if actual < minimum:
        raise RuntimeError(f"unexpectedly small download: {target}")
    if sha256 is not None and checksum(part) != sha256:
        raise RuntimeError(f"SHA-256 mismatch for {target}; remove {part} and retry")
    part.replace(destination)
    print(f"Saved: {target} ({actual:,} bytes)", flush=True)


def download_flowhmr():
    for host in (sources.HF_MIRROR, sources.HF_OFFICIAL):
        try:
            with urllib.request.urlopen(
                    f"{host}/api/models/{sources.FLOWHMR_REPO}?blobs=true", timeout=30) as response:
                manifest = json.load(response)
        except (OSError, ValueError) as error:
            print(f"{host} unavailable: {type(error).__name__}: {error}", flush=True)
            continue
        revision = manifest["sha"]
        for item in manifest["siblings"]:
            name = item["rfilename"]
            # Only the checkpoint folder is consumed; repository metadata files are
            # served through a mirror cache path that rejects plain urllib requests.
            if not name.startswith("flowhmr_latest/"):
                continue
            download(f"{host}/{sources.FLOWHMR_REPO}/resolve/{revision}/{name}", name,
                     size=item["size"], sha256=item.get("lfs", {}).get("sha256"))
        (files.MODEL_ROOT / "flowhmr_latest/download-revision.txt").write_text(
            f"{sources.FLOWHMR_REPO}\n{revision}\n", encoding="utf-8")
        return
    raise RuntimeError("FlowHMR checkpoint mirror unavailable")


def run(group):
    if group == "flowhmr":
        download_flowhmr()
    elif group == "yolox":
        asset = sources.PUBLIC_ASSETS["yolox"][0]
        last = None
        for url in asset.urls:
            try:
                download(url, asset.relative, size=asset.size, sha256=asset.sha256)
                return
            except (OSError, urllib.error.URLError, RuntimeError) as error:
                last = error
                print(f"  source failed ({url}): {type(error).__name__}: {error}", flush=True)
        raise RuntimeError(f"YOLOX download failed: {last}")
    elif group == "regressor":
        asset = sources.PUBLIC_ASSETS["joint-regressor"][0]
        last = None
        for url in asset.urls:
            try:
                download(url, asset.relative, size=asset.size, sha256=asset.sha256)
                return
            except (OSError, urllib.error.URLError, RuntimeError) as error:
                last = error
                print(f"  source failed ({url}): {type(error).__name__}: {error}", flush=True)
        raise RuntimeError(f"joint regressor download failed: {last}")
    elif group in ("sam", "vggt"):
        for repo, name, target, size, sha in sources.PERCEPTION_DOWNLOADS:
            wanted = target.startswith("sam-3d-body") if group == "sam" else target.startswith("vggt-omega")
            if wanted:
                download(sources.modelscope_file(repo, name), target, size=size, sha256=sha)
    elif group == "smplh":
        for remote, target, minimum in SMPLHUB_FILES[:3]:
            download(f"{SMPLHUB}/{remote}", target, minimum=minimum)
    elif group == "mano":
        for remote, target, minimum in SMPLHUB_FILES[3:]:
            download(f"{SMPLHUB}/{remote}", target, minimum=minimum)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=GROUPS, help="download only these groups")
    parser.add_argument("--skip", nargs="+", choices=GROUPS, default=[], help="skip these groups")
    args = parser.parse_args()
    files.ensure_model_folders()
    selected = [g for g in (args.only or GROUPS) if g not in args.skip]
    failures = []
    for group in selected:
        print(f"\n== {group} ==", flush=True)
        try:
            run(group)
        except Exception as error:  # keep going with the remaining groups
            failures.append(group)
            print(f"Failed {group}: {type(error).__name__}: {error}", flush=True)
    if failures:
        print("\nUnfinished groups: " + ", ".join(failures), flush=True)
        return 1
    print("\nAll selected models are installed in " + str(files.MODEL_ROOT), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
