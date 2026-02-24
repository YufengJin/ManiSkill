#!/usr/bin/env python3
"""
replay_demo.py — ManiSkill demo replay policy server (WebSocket).

Loads downloaded ManiSkill demos and replays them as a policy server.
Connect run_demo.py to this server to visualize demo replay in simulation.

Usage:
    1. Download demos: python -m mani_skill.utils.download_demo PickCube-v1
    2. Start replay server: python scripts/replay_demo.py --env_id PickCube-v1 --port 8000
    3. Run client: python scripts/run_demo.py --env_id PickCube-v1 --policy_server_addr localhost:8000

Both replay_demo and run_demo MUST use the same --env_id.

Container test (run_demo without GUI, saves video):
    # Terminal 1 - start server:
    docker exec -it maniskill_container /opt/conda/envs/maniskill/bin/python scripts/replay_demo.py --env_id PickCube-v1 --port 8000
    # Terminal 2 - run client:
    docker exec -it maniskill_container /opt/conda/envs/maniskill/bin/python scripts/run_demo.py --env_id PickCube-v1 --policy_server_addr localhost:8000 --num_resets 2

    Or use: ./scripts/test_replay_in_container.sh
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure mani_skill is importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import h5py
import numpy as np

from mani_skill import DEMO_DIR
from mani_skill.trajectory.replay_trajectory import sanity_check_and_format_seed
from mani_skill.utils.download_demo import DATASET_SOURCES
from mani_skill.utils.io_utils import load_json

try:
    from policy_websocket import BasePolicy, WebsocketPolicyServer
except ImportError:
    raise ImportError(
        "policy-websocket is required for replay_demo. "
        "Install: pip install 'policy-websocket @ git+https://github.com/YufengJin/policy_websocket.git'"
    )

logger = logging.getLogger(__name__)


def find_demo_h5_files(env_id: str, demo_dir: Path) -> list:
    """Find .h5 trajectory files for env_id under demo_dir.
    Returns list of (h5_path, json_path) tuples.
    ManiSkill demos are in subdirs: demo_dir/env_id/{rl,motionplanning,teleop}/*.h5"""
    demo_dir = Path(demo_dir)
    if not demo_dir.exists():
        return []

    candidates = []
    for h5_path in demo_dir.rglob("*.h5"):
        json_path = h5_path.with_suffix(".json")
        if not json_path.exists():
            continue
        try:
            data = load_json(str(json_path))
            file_env_id = data.get("env_info", {}).get("env_id")
            if file_env_id == env_id or (
                file_env_id is None and env_id in str(h5_path)
            ):
                candidates.append((str(h5_path), str(json_path)))
        except Exception:
            pass

    seen = set()
    out = []
    for h5, js in candidates:
        if h5 not in seen:
            seen.add(h5)
            out.append((h5, js))
    return out


class ReplayPolicy(BasePolicy):
    """Policy that replays actions from ManiSkill demo trajectories in sequence.
    Multiple demos (h5 files and episodes) are replayed one after another."""

    def __init__(self, env_id: str, demo_dir: Path, traj_index: int = 0):
        self.env_id = env_id
        self.demo_dir = Path(demo_dir)
        self._traj_index = traj_index
        self._h5_file = None
        self._current_h5_path = None
        self._current_traj_actions = None
        self._step_index = 0
        self._action_dim = 7
        # (h5_path, episode_id) for all trajectories, in order
        self._all_trajectories = []
        self._load_all_trajectories()

    def _load_all_trajectories(self):
        files = find_demo_h5_files(self.env_id, self.demo_dir)
        if not files:
            raise FileNotFoundError(
                f"No demo files found for {self.env_id} under {self.demo_dir}. "
                f"Run: python -m mani_skill.utils.download_demo {self.env_id}"
            )
        for h5_path, json_path in sorted(files):
            json_data = load_json(json_path)
            episodes = json_data.get("episodes", [])
            for ep in sorted(episodes, key=lambda x: x["episode_id"]):
                self._all_trajectories.append((h5_path, json_path, ep["episode_id"], ep))
        if not self._all_trajectories:
            raise ValueError(f"No episodes found in demo files for {self.env_id}")
        self._traj_index = self._traj_index % len(self._all_trajectories)
        self._load_current_trajectory()
        logger.info(f"Loaded {len(self._all_trajectories)} trajectories from {len(files)} demo file(s)")

    def _load_current_trajectory(self):
        h5_path, _json_path, episode_id, _ep = self._all_trajectories[self._traj_index]
        if self._current_h5_path != h5_path:
            if self._h5_file is not None:
                self._h5_file.close()
            self._h5_file = h5py.File(h5_path, "r")
            self._current_h5_path = h5_path
        traj = self._h5_file[f"traj_{episode_id}"]
        self._current_traj_actions = np.array(traj["actions"])
        self._step_index = 0

    def _get_current_episode_meta(self) -> dict:
        """Return reset_kwargs and control_mode for the current trajectory."""
        _h5_path, _json_path, _episode_id, episode = self._all_trajectories[self._traj_index]
        ep_copy = dict(episode)
        reset_kwargs = dict(ep_copy.get("reset_kwargs", {"options": {}, "seed": ep_copy.get("episode_seed", 0)}))
        ep_copy["reset_kwargs"] = reset_kwargs
        sanity_check_and_format_seed(ep_copy)
        return {
            "reset_kwargs": ep_copy["reset_kwargs"],
            "control_mode": ep_copy.get("control_mode", "pd_joint_pos"),
        }

    def infer(self, obs: dict) -> dict:
        # Replay mode: client requests reset params at start of each episode
        # Advance to next traj (except first call) then return current traj's params
        if obs.get("__get_reset_params__") is True:
            if getattr(self, "_get_reset_params_called", False):
                self.reset()  # advance to next trajectory for this new episode
            self._get_reset_params_called = True
            meta = self._get_current_episode_meta()
            return {
                "reset_kwargs": meta["reset_kwargs"],
                "control_mode": meta["control_mode"],
                "actions": np.zeros(self._action_dim, dtype=np.float64),
                "replay_mode": True,
            }

        if "task_name" in obs and obs["task_name"] != self.env_id:
            logger.warning(
                f"replay_demo env_id={self.env_id}, but client sent task_name={obs['task_name']}. "
                "They must match."
            )
        if "action_dim" in obs:
            self._action_dim = int(obs["action_dim"])

        is_init = "robot0_agentview_left_image" not in obs
        if self._current_traj_actions is None or self._step_index >= len(
            self._current_traj_actions
        ):
            action = np.zeros(self._action_dim, dtype=np.float64)
        else:
            a = self._current_traj_actions[self._step_index]
            action = np.asarray(a, dtype=np.float64).flatten()
            if action.shape[0] != self._action_dim:
                action = np.pad(
                    action,
                    (0, max(0, self._action_dim - action.shape[0])),
                    mode="constant",
                    constant_values=0,
                )[: self._action_dim]
            if not is_init:
                self._step_index += 1
        return {"actions": action}

    def reset(self) -> None:
        self._step_index = 0
        self._traj_index = (self._traj_index + 1) % len(self._all_trajectories)
        self._load_current_trajectory()

    def close(self):
        if self._h5_file is not None:
            try:
                self._h5_file.close()
            except Exception:
                pass
            self._h5_file = None


def parse_args():
    available = list(DATASET_SOURCES.keys())
    parser = argparse.ArgumentParser(
        description="ManiSkill demo replay policy server"
    )
    parser.add_argument(
        "--env_id",
        type=str,
        default="PickCube-v1",
        choices=available,
        help="Environment ID (must match run_demo --env_id)",
    )
    parser.add_argument(
        "--demo_dir",
        type=str,
        default=None,
        help=f"Demo directory. Default: {{DEMO_DIR}} (~/.maniskill/demos)",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    args = parse_args()
    demo_dir = Path(args.demo_dir) if args.demo_dir else DEMO_DIR

    policy = ReplayPolicy(env_id=args.env_id, demo_dir=demo_dir)
    metadata = {
        "policy_name": "ReplayPolicy",
        "action_dim": 7,
        "env_id": args.env_id,
        "replay_mode": True,
    }

    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=metadata,
    )
    print(f"Starting ReplayPolicy server for {args.env_id} on ws://{args.host}:{args.port}")
    print(f"Demo dir: {demo_dir}")
    print("Use same --env_id in run_demo. Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    if hasattr(policy, "close"):
        policy.close()
    print("Server stopped.")


if __name__ == "__main__":
    main()
