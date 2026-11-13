#!/bin/zsh
set -euo pipefail
cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-python3}"
if ! "$PYTHON" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))' 2>/dev/null; then
  echo "Python 3.10+ is required (set PYTHON=/path/to/python3)." >&2
  exit 1
fi

ROOT="$PWD"

# The device fixes ship as patches because the submodules are pinned to upstream
# commits that only target CUDA. Re-running setup is a no-op once they apply.
apply_patch() {
  local repo="$1" patch="$ROOT/third_party/patches/$2"
  if git -C "$ROOT/$repo" apply --check "$patch" 2>/dev/null; then
    git -C "$ROOT/$repo" apply "$patch"
    echo "applied $2"
  elif ! git -C "$ROOT/$repo" apply --check -R "$patch" 2>/dev/null; then
    echo "$2 does not apply to $repo; reset the submodule and retry" >&2
    exit 1
  fi
}

git submodule update --init --recursive
apply_patch third_party/sam-3d-body 0005-sam3d-body-device-agnostic.patch
apply_patch third_party/YOLOX 0006-yolox-source-build-any-platform.patch

# SAM's DINOv3 backbone loads Meta's hub code; vendoring it keeps first-run
# inference off GitHub. Without it torch.hub fetches the same code on demand.
if [[ ! -f third_party/dinov3/hubconf.py ]]; then
  DINOv3_URL=https://github.com/facebookresearch/dinov3
  TMP="$(mktemp -d)"
  if git clone --depth 1 -q "$DINOv3_URL" "$TMP/dinov3" 2>/dev/null \
    || git clone --depth 1 -q "https://ghfast.top/$DINOv3_URL" "$TMP/dinov3" 2>/dev/null; then
    rsync -a "$TMP/dinov3/" third_party/dinov3/
    git -C "$TMP/dinov3" rev-parse HEAD > third_party/dinov3/SOURCE_REVISION
    echo "vendored DINOv3 hub code ($(cat third_party/dinov3/SOURCE_REVISION))"
  else
    echo "note: could not fetch DINOv3 hub code; torch.hub will download it on first run" >&2
  fi
  rm -rf "$TMP"
fi

"$PYTHON" -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/pip install -r requirements-macos.txt
# Opt-in MLX SAM backbone: WITH_MLX=1 tools/macos/setup.sh
if [[ "${WITH_MLX:-0}" == "1" ]]; then
  .venv/bin/pip install -r requirements-macos-mlx.txt
fi
.venv/bin/pip install --no-build-isolation --no-deps -e third_party/YOLOX
.venv/bin/python tools/macos/download_models.py "$@"
.venv/bin/python tools/macos/check_install.py
