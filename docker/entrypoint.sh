#!/usr/bin/env bash
set -e

cd /workspace

# Determine ManiSkill project root (may be /workspace/ManiSkill when parent is mounted)
MANISKILL_ROOT=""
if [ -f /workspace/ManiSkill/setup.py ]; then
  MANISKILL_ROOT="/workspace/ManiSkill"
elif [ -f /workspace/setup.py ]; then
  MANISKILL_ROOT="/workspace"
fi

# Install ManiSkill in editable mode when source is mounted
if [ -n "${MANISKILL_ROOT}" ]; then
  echo "Installing ManiSkill from ${MANISKILL_ROOT} (editable)..."
  micromamba run -n maniskill pip install -e "${MANISKILL_ROOT}"

  if ! micromamba run -n maniskill python -c "import policy_websocket" 2>/dev/null; then
    echo "Installing policy-websocket (for run_demo / run_eval / replay_demo)..."
    micromamba run -n maniskill python -m pip install -q \
      "policy-websocket @ git+https://github.com/YufengJin/policy_websocket.git"
  fi

  ASSET_DIR="${MS_ASSET_DIR:-${MANISKILL_ROOT}/.maniskill}"
  if [ ! -d "${ASSET_DIR}" ] || [ -z "$(ls -A "${ASSET_DIR}" 2>/dev/null)" ]; then
    echo "[INFO] No assets found under ${ASSET_DIR}."
    echo "       Example: python -m mani_skill.utils.download_asset all -y"
    echo "       Demos:   python -m mani_skill.utils.download_demo PickCube-v1"
  fi
fi

if [ $# -eq 0 ]; then
  exec bash
else
  exec micromamba run -n maniskill "$@"
fi
