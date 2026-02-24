# ManiSkill Docker Guide

This document describes how to build and run ManiSkill containers.

## Prerequisites

- Docker and Docker Compose installed
- NVIDIA Container Toolkit configured (for GPU support)
- For X11 mode: X11 server running on the host

## Build Image

From the project root:

```bash
cd docker
docker-compose -f docker-compose.x11.yaml build
# or
docker-compose -f docker-compose.headless.yaml build
```

With custom image name:

```bash
IMAGE=maniskill:custom docker-compose -f docker-compose.x11.yaml build
```

## Start Container

### X11 mode (GUI display)

For visualization (e.g. `--render-mode=human`):

```bash
cd docker
DISPLAY=${DISPLAY} docker-compose -f docker-compose.x11.yaml up -d
```

Or foreground:

```bash
DISPLAY=${DISPLAY} docker-compose -f docker-compose.x11.yaml up
```

### Headless mode

For training, batch evaluation, or other non-GUI use:

```bash
cd docker
docker-compose -f docker-compose.headless.yaml up -d
```

Or foreground:

```bash
docker-compose -f docker-compose.headless.yaml up
```

## Attach to Container

```bash
docker exec -it maniskill_container bash
```

## Test Installation

Inside the container, activate the environment and run the demo:

```bash
python -m mani_skill.examples.demo_random_action
```

With rendering and video recording:

```bash
python -m mani_skill.examples.demo_random_action -e PickCube-v1 --render-mode=rgb_array --record-dir=videos
```

With GUI (X11 mode only):

```bash
python -m mani_skill.examples.demo_random_action -e PickCube-v1 --render-mode=human
```

## Stop Container

```bash
cd docker
docker-compose -f docker-compose.x11.yaml down
# or
docker-compose -f docker-compose.headless.yaml down
```

## Assets and Demos (avoid re-download)

`MS_ASSET_DIR` is set to `/workspace/.maniskill` (project root `.maniskill`). With `../:/workspace` mounted, all assets/demos persist on the host at `ManiSkill/.maniskill` and are reused across container runs.

### Asset categories (download_asset.py)

| Category       | Examples |
|----------------|----------|
| **scene**      | ReplicaCAD, ReplicaCADRearrange, AI2THOR, RoboCasa |
| **robot**      | ur10e, anymal_c, unitree_h1, unitree_g1, unitree_go2, stompy, widowx250s, googlerobot, robotiq_2f, xarm6, widowxai, xlerobot |
| **task_assets**| ycb, pick_clutter_ycb_configs, assembling_kits, panda_avoid_obstacles, bridge_v2_real2sim, oakink-v2 |
| **objects**    | partnet_mobility (cabinet, chair, bucket, faucet) |

```bash
# List assets by category
python -m mani_skill.utils.download_asset -l scene
python -m mani_skill.utils.download_asset -l robot
python -m mani_skill.utils.download_asset -l task_assets

# Download (e.g. all scenes)
python -m mani_skill.utils.download_asset all -y
```

### Demo datasets (download_demo.py)

Source: `https://huggingface.co/datasets/haosulab/ManiSkill_Demonstrations`

Env IDs: PickCube-v1, PushCube-v1, StackCube-v1, PegInsertionSide-v1, PlugCharger-v1, etc. (see `python -m mani_skill.utils.download_demo` for full list).

```bash
python -m mani_skill.utils.download_demo PickCube-v1
python -m mani_skill.utils.download_demo all   # all demos
```

---

## Why Vulkan Is Required

ManiSkill uses [SAPIEN](https://sapien.ucsd.edu/) as its simulation engine. SAPIEN relies on **Vulkan** for GPU-accelerated rendering:

- **State-based simulation**: Does not require Vulkan (CPU/GPU physics only)
- **Rendering**: RGB, depth, segmentation, and `--render-mode=human` visualization all require Vulkan

Inside Docker, the NVIDIA driver is provided by the NVIDIA Container Toolkit. Vulkan needs additional configuration so the Vulkan loader can find the NVIDIA Vulkan implementation. This image installs `libvulkan1`, `vulkan-tools`, and copies `nvidia_icd.json` / `nvidia_layers.json` to enable Vulkan in the container.

## Configuration

### Container name
- Fixed name: `maniskill_container`

### GPU
- Uses all available NVIDIA GPUs by default
- Set `GPU` env var to override (default: `all`)

### Working directory
- Container workdir: `/workspace`

### Network
- Uses `host` network mode

### Environment variables
- **DISPLAY** (X11 only): X11 display
- **GPU**: GPU selection (default: `all`)
- **MS_ASSET_DIR**: Directory for ManiSkill assets (default: `~/.maniskill/data`)
- **MS_SKIP_ASSET_DOWNLOAD_PROMPT**: Set to `1` to skip interactive asset download prompts

## Troubleshooting

### X11 permission denied

```bash
xhost +local:root
```

### GPU not detected

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi
```

### pkg_resources / setuptools

If you see `ModuleNotFoundError: No module named 'pkg_resources'`, the Dockerfile pins `setuptools<75` because SAPIEN requires `pkg_resources` (removed in setuptools 82+). Ensure the Dockerfile has this pin.

### Vulkan initialization failed

Ensure the host has an NVIDIA driver with Vulkan support. Common errors:
- `vk::Instance::enumeratePhysicalDevices: ErrorInitializationFailed`
- `Some required Vulkan extension is not present`

See the [ManiSkill installation docs](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#vulkan) for host-side Vulkan setup.

### Container name conflict

```bash
docker stop maniskill_container
docker rm maniskill_container
```

## Notes

- X11 mode requires a running X server and `xhost` access for Docker
- First build can take a while (PyTorch, ManiSkill, PhysX)
- When the ManiSkill source is mounted at `/workspace`, the entrypoint installs it in editable mode (`pip install -e .`)
