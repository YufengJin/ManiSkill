# Docker Setup for ManiSkill

## Prerequisites

Run the prerequisite checker — it verifies Docker, Compose plugin, and the NVIDIA driver + Container Toolkit:

```bash
bash docker/check_prereqs.sh
```

If any check fails, the script prints an install link and exits non-zero.

Manual install references:

- Docker Engine — https://docs.docker.com/engine/install/
- NVIDIA Container Toolkit — https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
- NVIDIA GPU driver — https://www.nvidia.com/en-us/drivers/

## Dependencies (uv)

Runtime versions are pinned in `pyproject.toml` and `uv.lock`. Refresh the lockfile after editing dependencies:

```bash
uv lock
```

Docker installs the locked environment with `uv sync --frozen --no-install-project` (see `docker/Dockerfile`).

## Build

From project root:

```bash
docker compose -f docker/docker-compose.headless.yaml build
```

## Volume mounts

Update paths in the compose file to match your system.

| Container path | Purpose |
|----------------|---------|
| `../` → `/workspace/maniskill` | Project (editable install) |
| `${HOME}/.cache/huggingface` | HF model cache |

## Usage

### Headless (training / serving)

```bash
docker compose -f docker/docker-compose.headless.yaml up -d
docker exec -it maniskill-headless bash
```

### X11 (GUI)

```bash
xhost +local:
docker compose -f docker/docker-compose.x11.yaml up -d
docker exec -it maniskill-gui bash
```

## Entrypoint

On each start:

1. Uses uv-managed venv at `/opt/venv` (Python 3.9)
2. When `pyproject.toml` is present: `uv pip install -e .` (deps already synced in the image)
