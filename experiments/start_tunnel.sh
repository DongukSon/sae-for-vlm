#!/bin/bash
# Initialize a RunPod workspace and start a persistent VS Code Tunnel.
# This configures SSH, tmux, persistent Claude/Git/VS Code data, and the tunnel.

set -euo pipefail

# SSH key: volume ignores chmod, so copy to container disk with 600
mkdir -p /root/.ssh && chmod 700 /root/.ssh
install -m 600 /workspace/.ssh/id_ed25519 /root/.ssh/id_ed25519
grep -q github.com /root/.ssh/known_hosts 2>/dev/null || \
  ssh-keyscan github.com >> /root/.ssh/known_hosts 2>/dev/null

# tmux for long experiments
command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux)

# Persistent config on the volume (inherited by the tunnel and VS Code server below)
export CLAUDE_CONFIG_DIR=/workspace/.claude
export GIT_CONFIG_GLOBAL=/workspace/.gitconfig
mkdir -p "$CLAUDE_CONFIG_DIR"
touch "$GIT_CONFIG_GLOBAL"
# Also set them for interactive shells
grep -q CLAUDE_CONFIG_DIR /root/.bashrc || cat >> /root/.bashrc << 'RC'
export CLAUDE_CONFIG_DIR=/workspace/.claude
export GIT_CONFIG_GLOBAL=/workspace/.gitconfig
RC

# VS Code extensions live on the volume.
# Must run before the first VS Code connection, or VS Code creates a real dir here instead.
mkdir -p /workspace/.vscode-server
if [ -d /root/.vscode-server ] && [ ! -L /root/.vscode-server ]; then
  echo "WARNING: /root/.vscode-server is a real directory; extensions won't persist."
else
  [ -e /root/.vscode-server ] || ln -s /workspace/.vscode-server /root/.vscode-server
fi

# VS Code tunnel (skip if already running)
if pgrep -f "code tunnel" >/dev/null; then
  echo "Tunnel already running."
else
  nohup /workspace/code tunnel \
    --name runpod \
    --accept-server-license-terms \
    --cli-data-dir /workspace/.vscode-cli \
    > /workspace/tunnel.log 2>&1 &
  echo "Tunnel started. Check log: cat /workspace/tunnel.log"
fi
