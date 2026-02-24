"""Shared utilities for run_eval.py and run_demo.py (WebSocket policy scripts).
Client sends ManiSkill obs formatted for policy compatibility."""

import os
import random
from typing import Any, Dict, Optional, Union

import gymnasium as gym
import numpy as np
import torch

# Import to trigger env registration
import mani_skill.envs  # noqa: F401
from mani_skill.utils.registration import REGISTERED_ENVS

# ---------------------------------------------------------------------------
# Task metadata
# ---------------------------------------------------------------------------
EVAL_TASK_BLACKLIST = frozenset([
    "Empty-v1",
    "CustomEnv-v1",
    "FrankaMoveBenchmark-v1",
    "FrankaPickCubeBenchmark-v1",
    "CartpoleBalanceBenchmark-v1",
])

TASK_DESCRIPTIONS: Dict[str, str] = {
    "PickCube-v1": "Pick the red cube and place it on the green sphere.",
    "PickCubeSO100-v1": "Pick the red cube and place it on the green sphere.",
    "PickCubeWidowXAI-v1": "Pick the red cube and place it on the green sphere.",
    "StackCube-v1": "Stack the red cube on top of the green cube.",
    "PushCube-v1": "Push the red cube to the target green region.",
    "PegInsertionSide-v1": "Insert the peg into the hole from the side.",
    "PlugCharger-v1": "Plug the charger into the receptacle.",
    "TurnFaucet-v1": "Turn the faucet to the target angle.",
    "LiftPegUpright-v1": "Lift the peg upright.",
    "DrawTriangle-v1": "Draw a triangle on the canvas.",
}


def get_task_max_steps(env_id: str, default_horizon: int = 500) -> int:
    """Return max_episode_steps (horizon) for a ManiSkill task.

    Reads from REGISTERED_ENVS.
    """
    if env_id not in REGISTERED_ENVS:
        return default_horizon
    spec = REGISTERED_ENVS[env_id]
    if spec.max_episode_steps is None:
        return default_horizon
    return spec.max_episode_steps


def get_available_tasks() -> list:
    """Return list of task env_ids suitable for eval/demo (excludes blacklist)."""
    return [eid for eid in REGISTERED_ENVS if eid not in EVAL_TASK_BLACKLIST]


def set_seed_everywhere(seed: int, deterministic: bool = True) -> None:
    """Set global random seeds for reproducibility."""
    if deterministic:
        os.environ["DETERMINISTIC"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_task_description(env_id: str) -> str:
    """Return natural language description for env_id if available."""
    return TASK_DESCRIPTIONS.get(env_id, "")


def get_success_from_info(info: dict, num_envs: int = 1) -> bool:
    """Extract success from info dict. Handles scalar/batched torch.Tensor."""
    if "success" not in info:
        return False
    s = info["success"]
    if isinstance(s, torch.Tensor):
        s = s.cpu().numpy()
        if s.ndim > 0:
            s = bool(s[0])
        else:
            s = bool(s)
    return bool(s)


def _to_numpy(x: Any) -> np.ndarray:
    """Convert tensor/array to numpy (H, W, C) for single env."""
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    x = np.asarray(x)
    if x.ndim == 4:
        x = x[0]
    return x.astype(np.uint8) if x.dtype != np.uint8 else x


def obs_to_policy_format(obs: dict, task_description: str = "") -> dict:
    """Convert ManiSkill obs to policy-expected format (RoboCasa-compatible keys).

    Maps sensor_data[camera][rgb] to:
    - robot0_agentview_left_image
    - robot0_agentview_right_image
    - robot0_eye_in_hand_image

    For single-camera tasks, the same image is used for all three.
    Returns only serializable (numpy) values for WebSocket/msgpack.
    """
    out = {}

    sensor_data = obs.get("sensor_data") or {}
    cameras = [k for k in sorted(sensor_data.keys()) if isinstance(sensor_data[k], dict)]

    def get_rgb(cam_name: str) -> Optional[np.ndarray]:
        if cam_name not in sensor_data:
            return None
        data = sensor_data[cam_name]
        if isinstance(data, dict) and "rgb" in data:
            return _to_numpy(data["rgb"])
        return None

    if len(cameras) >= 3:
        left = get_rgb(cameras[0])
        right = get_rgb(cameras[1])
        wrist = get_rgb(cameras[2])
    elif len(cameras) >= 1:
        img = get_rgb(cameras[0])
        left = right = wrist = img
    else:
        left = right = wrist = None

    if left is not None:
        out["robot0_agentview_left_image"] = left
    if right is not None:
        out["robot0_agentview_right_image"] = right
    if wrist is not None:
        out["robot0_eye_in_hand_image"] = wrist

    out["task_description"] = task_description
    if "action_dim" in obs:
        out["action_dim"] = int(obs["action_dim"])
    if "action_low" in obs:
        out["action_low"] = np.asarray(obs["action_low"], dtype=np.float64)
    if "action_high" in obs:
        out["action_high"] = np.asarray(obs["action_high"], dtype=np.float64)
    if "task_name" in obs:
        out["task_name"] = str(obs["task_name"])

    return out


def get_action_spec(env: gym.Env) -> tuple:
    """Return (action_dim, action_low, action_high) for policy init.
    Extracts from env.action_space (Box). Uses single_action_space if available."""
    from gymnasium import spaces
    space = getattr(env, "single_action_space", None) or env.action_space
    if not hasattr(space, "shape"):
        raise ValueError("ManiSkill run_demo/run_eval requires Box action space")
    if isinstance(space, spaces.Box):
        low = np.array(space.low).flatten()
        high = np.array(space.high).flatten()
        return low.shape[0], low.astype(np.float64), high.astype(np.float64)
    raise ValueError(f"Unsupported action space: {type(space)}")


def create_maniskill_env(
    env_id: str,
    robot_uids: str = "panda",
    obs_mode: str = "rgbd",
    img_res: int = 224,
    render_mode: Optional[str] = None,
    num_envs: int = 1,
    seed: Optional[int] = None,
    control_mode: Optional[str] = None,
    sensor_configs: Optional[dict] = None,
    **kwargs,
) -> gym.Env:
    """Create a ManiSkill environment for run_eval / run_demo."""
    if env_id not in REGISTERED_ENVS:
        raise KeyError(f"Unknown env_id: {env_id}. Available: {list(REGISTERED_ENVS.keys())}")

    cfg = dict(
        obs_mode=obs_mode,
        num_envs=num_envs,
        render_mode=render_mode,
        robot_uids=robot_uids,
        control_mode=control_mode,
        **kwargs,
    )
    if sensor_configs is None:
        sensor_configs = {}
    sensor_configs = {**sensor_configs, "width": img_res, "height": img_res}
    cfg["sensor_configs"] = sensor_configs

    env = gym.make(env_id, **cfg)
    return env
