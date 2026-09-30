#!/bin/bash
# One-time environment setup on a RunPod GPU pod (image: runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04).
# Usage (on the pod): bash experiments/pod_setup.sh
# Creates a .venv inside the repo (under /workspace, so it survives pod restarts) on top of the image's
# python3 with --system-site-packages, so the image's CUDA torch is reused instead of downloaded again.
# Only requirements-pod.txt is installed; transitive deps are pinned to requirements.txt via constraints.

set -euo pipefail
cd "$(dirname "$0")/.."
# Keep pip's download cache on the volume so reinstalling on a fresh pod is fast
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/.cache/pip}"

SYS_PY="$(command -v python3)"
if ! "${SYS_PY}" -c 'import torch' 2>/dev/null; then
  echo "${SYS_PY} has no torch. This script expects the runpod/pytorch image." >&2
  exit 1
fi
SYS_PY_VER="$("${SYS_PY}" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"

# An old .venv (e.g. the previous Python 3.11 one) would shadow or miss the image's torch
if [ -e .venv ]; then
  if ! grep -q '^include-system-site-packages = true' .venv/pyvenv.cfg 2>/dev/null \
    || [ "$(.venv/bin/python -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)" != "${SYS_PY_VER}" ]; then
    echo "Existing .venv was not built on the image's Python ${SYS_PY_VER} with system site-packages. Remove it (rm -rf .venv) and rerun." >&2
    exit 1
  fi
else
  if ! "${SYS_PY}" -c 'import ensurepip' 2>/dev/null; then
    apt-get update && apt-get install -y "python${SYS_PY_VER}-venv"
  fi
  "${SYS_PY}" -m venv --system-site-packages .venv
fi

source .venv/bin/activate
pip install --upgrade pip

# Constraints: requirements.txt without the torch/CUDA stack, plus the image's own torch/torchvision
# versions so no dependency can pull a different torch into the venv
CONSTRAINTS="$(mktemp)"
trap 'rm -f "${CONSTRAINTS}"' EXIT
grep -vE '^(torch|torchvision|triton|nvidia-[a-z0-9-]+)==' requirements.txt > "${CONSTRAINTS}"
# Use the installed metadata version (what pip matches against), not torch.__version__,
# which can carry a local tag like +cu121 that the metadata lacks
python -c "from importlib.metadata import version; print(f'torch=={version(\"torch\")}'); print(f'torchvision=={version(\"torchvision\")}')" >> "${CONSTRAINTS}"
pip install -r requirements-pod.txt -c "${CONSTRAINTS}"

python - <<'EOF'
import sys, torch
assert ".venv" not in torch.__file__, f"torch was reinstalled into the venv: {torch.__file__}"
assert torch.cuda.is_available(), "CUDA not available"
import transformers, nnsight, dictionary_learning, utils  # noqa: F401
print("Python", sys.version.split()[0], "| torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0))
EOF
