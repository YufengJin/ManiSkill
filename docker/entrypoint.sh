#!/usr/bin/env bash
set -e

cd /workspace

# Install ManiSkill in editable mode when source is mounted (for in-container dev)
MANISKILL_ROOT=""
if [ -f /workspace/ManiSkill/setup.py ]; then
  MANISKILL_ROOT="/workspace/ManiSkill"
elif [ -f /workspace/setup.py ]; then
  MANISKILL_ROOT="/workspace"
fi

if [ -n "${MANISKILL_ROOT}" ]; then
  echo "Installing ManiSkill from ${MANISKILL_ROOT} (editable)..."
  micromamba run -n maniskill pip install -e "${MANISKILL_ROOT}"
fi

if [ $# -eq 0 ]; then
  exec bash
else
  exec micromamba run -n maniskill "$@"
fi
