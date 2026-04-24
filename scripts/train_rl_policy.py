#!/usr/bin/env python3
"""
train_rl_policy.py — ManiSkill RL training via stable-baselines3 (sb3 mode).

Purpose
-------
Unified entrypoint that selects an sb3 algorithm by Hydra config (--config-name
or algo=... override) and trains it on any registered ManiSkill env. Generated
by the `benchmark-env-generator` skill (RL category, sb3 mode — ManiSkill ships
training examples but no in-tree baselines library, so sb3 is the generic
fallback). Supports PPO, SAC, TD3, DDPG, DQN, A2C out of the box.

Config files live under `<repo>/conf/{ppo,sac,td3}.yaml` (Hydra default).

Example
-------
    # Short smoke run (L3_RL tier; takes ~1 minute, no wandb)
    python scripts/train_rl_policy.py --config-name ppo \\
        env_id=PickCube-v1 total_timesteps=5000 n_envs=1 wandb=null

    # Full run (SAC, 1M steps, override from CLI)
    python scripts/train_rl_policy.py --config-name sac \\
        env_id=PickCube-v1 total_timesteps=1_000_000

    # Switch algo at runtime
    python scripts/train_rl_policy.py --config-name ppo algo=ppo

Caveat: ManiSkill envs are GPU sim and return torch tensors by default. The
training script wraps `gym.make(...)` in a cpu-tensor adapter so sb3's numpy
pipeline works without modification.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Type

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

# Ensure repo root importable when run as `python scripts/train_rl_policy.py`
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import gymnasium as gym

from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


SB3_CLASS_MAP: dict[str, Type[BaseAlgorithm]] = {
    "ppo": PPO, "sac": SAC, "td3": TD3, "ddpg": DDPG, "dqn": DQN, "a2c": A2C,
}


class _ManiSkillCPUAdapter(gym.Wrapper):
    """ManiSkill returns torch cuda tensors with a leading num_envs batch dim
    even for `num_envs=1`. Convert to numpy and drop the batch dim so sb3's
    DummyVecEnv pipeline (which expects a plain gym.Env) works unmodified."""

    def __init__(self, env):
        super().__init__(env)
        from gymnasium.spaces import Box
        obs_space = env.observation_space
        act_space = env.action_space
        if isinstance(obs_space, Box) and len(obs_space.shape) >= 2 and obs_space.shape[0] == 1:
            low = np.asarray(obs_space.low).reshape(obs_space.shape)[0]
            high = np.asarray(obs_space.high).reshape(obs_space.shape)[0]
            self.observation_space = Box(low=low, high=high, dtype=obs_space.dtype)
        else:
            self.observation_space = obs_space
        if isinstance(act_space, Box) and len(act_space.shape) >= 2 and act_space.shape[0] == 1:
            low = np.asarray(act_space.low).reshape(act_space.shape)[0]
            high = np.asarray(act_space.high).reshape(act_space.shape)[0]
            self.action_space = Box(low=low, high=high, dtype=act_space.dtype)
        else:
            self.action_space = act_space

    def _to_numpy(self, obs):
        import torch
        if isinstance(obs, dict):
            return {k: self._to_numpy(v) for k, v in obs.items()}
        if isinstance(obs, torch.Tensor):
            return obs.detach().cpu().numpy()
        return obs

    def _unbatch(self, obs):
        if isinstance(obs, dict):
            return {k: self._unbatch(v) for k, v in obs.items()}
        if isinstance(obs, np.ndarray) and obs.ndim > 0 and obs.shape[0] == 1:
            return obs[0]
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._unbatch(self._to_numpy(obs)), {}

    def step(self, action):
        # sb3 sends numpy shape (action_dim,); ManiSkill expects (n_envs, action_dim)
        if isinstance(action, np.ndarray) and action.ndim == 1:
            action = action[None, :]
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._unbatch(self._to_numpy(obs))
        reward = float(self._to_numpy(reward).reshape(-1)[0])
        terminated = bool(np.asarray(self._to_numpy(terminated)).reshape(-1)[0])
        truncated = bool(np.asarray(self._to_numpy(truncated)).reshape(-1)[0])
        return obs, reward, terminated, truncated, {}


def make_env_fn(env_id: str):
    """Build a fresh env instance. Called once per subprocess in SubprocVecEnv."""
    def _make():
        import mani_skill.envs  # noqa: F401 — registers envs
        env = gym.make(env_id, obs_mode="state", control_mode="pd_joint_delta_pos",
                       render_mode=None, num_envs=1)
        return _ManiSkillCPUAdapter(env)
    return _make


@hydra.main(version_base=None, config_path="../conf", config_name="ppo")
def main(cfg: DictConfig):
    algo = cfg.get("algo") or hydra.core.hydra_config.HydraConfig.get().job.config_name
    if algo not in SB3_CLASS_MAP:
        print(f"[error] Unknown algo '{algo}'. Supported: {list(SB3_CLASS_MAP.keys())}",
              file=sys.stderr)
        sys.exit(2)
    algo_cls = SB3_CLASS_MAP[algo]

    env_id = str(cfg.get("env_id", "PickCube-v1"))
    n_envs = int(cfg.get("n_envs", 1))
    total_timesteps = int(cfg.get("total_timesteps", 1_000_000))

    env_fn = make_env_fn(env_id)
    vec_env = SubprocVecEnv([env_fn for _ in range(n_envs)]) if n_envs > 1 else DummyVecEnv([env_fn])

    reserved = {"algo", "env_id", "n_envs", "total_timesteps", "checkpoint_dir",
                "wandb", "log_dir"}
    raw = OmegaConf.to_container(cfg, resolve=True)
    model_kwargs = {k: v for k, v in raw.items() if k not in reserved and v is not None}

    model = algo_cls(env=vec_env, verbose=1, **model_kwargs)

    print(f"[train] algo={algo} env_id={env_id} n_envs={n_envs} "
          f"total_timesteps={total_timesteps}", flush=True)
    try:
        model.learn(total_timesteps=total_timesteps)
    except KeyboardInterrupt:
        print("[train] interrupted — saving partial checkpoint")

    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_root = Path(cfg.get("checkpoint_dir", f"/workspace/.sb3_checkpoints/{algo}"))
    ckpt_dir = ckpt_root / f"{algo}--{env_id}--{date_str}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{algo}_{total_timesteps}.zip"
    model.save(str(ckpt_path))
    print(f"[train] saved checkpoint to {ckpt_path}")
    print(f"L3_RL OK: train_rl_policy completed")


if __name__ == "__main__":
    main()
