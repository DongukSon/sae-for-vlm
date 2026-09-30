#!/bin/bash
# Per-session environment on a RunPod pod. Source it in every new shell (after pod_setup.sh has run once):
#   source experiments/pod_env.sh
# Everything outside /workspace (the network volume) is wiped when the pod stops, so dataset archives,
# caches, secrets and the git SSH key all live under /workspace.
# Secrets go in /workspace/.pod_secrets (not in git), e.g.:
#   export WANDB_API_KEY=...
#   export HF_TOKEN=...

VOLUME="${VOLUME:-/workspace}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Dataset archives stay on the volume; each pod extracts them to the faster container disk
# (${HOME}/data, wiped on stop) and reads images from there
export ARCHIVE_ROOT="${VOLUME}/data"
export DATA_ROOT="${HOME}/data"
export HF_HOME="${VOLUME}/.cache/huggingface"
export TORCH_HOME="${VOLUME}/.cache/torch"
export PIP_CACHE_DIR="${VOLUME}/.cache/pip"
export WANDB_DIR="${REPO_DIR}"
mkdir -p "${ARCHIVE_ROOT}" "${DATA_ROOT}" "${HF_HOME}" "${TORCH_HOME}" "${PIP_CACHE_DIR}"

# SSH key for git pull/push, kept on the volume (~/.ssh does not survive a pod restart).
# The network volume may not honor chmod (files stay 0666), and ssh rejects such keys,
# so copy it to the container disk with 600 permissions and use that copy.
if [ -f "${VOLUME}/.ssh/id_ed25519" ]; then
  mkdir -p ~/.ssh && chmod 700 ~/.ssh
  install -m 600 "${VOLUME}/.ssh/id_ed25519" ~/.ssh/id_ed25519_github
  export GIT_SSH_COMMAND="ssh -i ${HOME}/.ssh/id_ed25519_github -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"
fi

if [ -f "${VOLUME}/.pod_secrets" ]; then
  source "${VOLUME}/.pod_secrets"
fi

# The venv's python symlinks to the image's python3 and takes torch from its site-packages,
# so this only fails if the venv is missing or was built on a different image
if "${REPO_DIR}/.venv/bin/python" -c 'import torch' 2>/dev/null; then
  source "${REPO_DIR}/.venv/bin/activate"
else
  echo "venv not usable on this pod. Run: bash experiments/pod_setup.sh && source experiments/pod_env.sh" >&2
fi
