#!/bin/bash
# One-time environment setup on a RunPod GPU pod.
# Usage (on the pod): bash experiments/pod_setup.sh
# Creates a Python 3.11 .venv inside the repo (under /workspace, so it survives pod restarts) and
# installs the original requirements.txt, which includes the CUDA build of torch 2.1.2.

set -euo pipefail
cd "$(dirname "$0")/.."
# Keep pip's download cache on the volume so reinstalling on a fresh pod is fast
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/.cache/pip}"

# Same Python version as the local venv; torch 2.1.2 has no wheels for 3.12+
if ! command -v python3.11 >/dev/null; then
  apt-get update && apt-get install -y python3.11 python3.11-venv
fi

if [ -x .venv/bin/python ] && ! .venv/bin/python -c 'import sys; sys.exit(sys.version_info[:2] != (3, 11))'; then
  echo "Existing .venv is not Python 3.11 ($(.venv/bin/python --version)). Remove it (rm -rf .venv) and rerun." >&2
  exit 1
fi
if [ ! -x .venv/bin/python ]; then
  python3.11 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('Python', __import__('sys').version.split()[0], '| torch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0))"
