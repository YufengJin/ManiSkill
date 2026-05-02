#!/usr/bin/env bash
# Check host prerequisites for this project's Docker environment.
# Idempotent. Run before `docker compose build` on a new machine.
#
#   bash docker/check_prereqs.sh
#
# Exits non-zero with install links if anything is missing. Does NOT install
# system packages — those require sudo and are host-specific.

set -e

if [ -t 1 ]; then
  R='\033[0;31m'; G='\033[0;32m'; N='\033[0m'
else
  R=''; G=''; N=''
fi
ok()  { printf "${G}[OK]${N}   %s\n" "$*"; }
err() { printf "${R}[FAIL]${N} %s\n" "$*" >&2; }

if ! command -v docker &>/dev/null; then
  err "docker not found."
  err "  Install: https://docs.docker.com/engine/install/"
  exit 1
fi
ok "docker: $(docker --version)"

if ! docker compose version &>/dev/null; then
  err "docker compose plugin not found."
  err "  Install: apt install docker-compose-plugin"
  exit 1
fi
ok "docker compose: $(docker compose version --short 2>/dev/null || echo present)"

# Verify current user can talk to the Docker daemon WITHOUT sudo.
# docker --version and docker compose version are pure client commands and
# don't prove daemon access; build/up/exec will fail without it.
if ! docker info >/dev/null 2>&1; then
  err "Cannot reach the Docker daemon as user '$(whoami)'."
  err "  - Is the daemon running?  systemctl status docker"
  err "  - Are you in the docker group?  groups \$USER"
  err "  - If not, add and re-login:  sudo usermod -aG docker \$USER && newgrp docker"
  err "  (Otherwise every docker/compose command will need sudo.)"
  exit 1
fi
ok "Docker daemon: reachable without sudo"

if ! command -v nvidia-smi &>/dev/null; then
  err "nvidia-smi not found. NVIDIA GPU driver must be installed on the host."
  err "  Install: https://www.nvidia.com/en-us/drivers/"
  exit 1
fi
ok "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

if ! docker info 2>/dev/null | grep -qi nvidia; then
  err "NVIDIA Container Toolkit is not registered with Docker daemon."
  err "  Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
  err "  Then: sudo systemctl restart docker"
  exit 1
fi
ok "NVIDIA Container Toolkit: registered"

echo
ok "All host prerequisites satisfied."
