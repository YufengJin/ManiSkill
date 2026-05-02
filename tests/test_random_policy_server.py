#!/usr/bin/env python3
"""
Test policy server for ManiSkill — returns random 8-D actions via WebSocket.

Purpose
-------
Smoke-test companion for `scripts/run_demo.py` and `scripts/run_eval.py`.
Both scripts are WebSocket clients that need a server at `--policy_server_addr`.
The random policy never solves the task; it exists purely to exercise the
end-to-end wiring (handshake, obs serialization, action deserialization,
env stepping, video encoding) so a fresh `docker compose up` is immediately
runnable.

Headless by construction: no env, no dataset, no GUI — pure WebSocket server.

Example
-------
    # 1. Start the server (this file)
    python tests/test_random_policy_server.py --port 8765

    # 2. In another terminal (or a second `docker exec`), run clients
    python scripts/run_demo.py --policy_server_addr localhost:8765 \
        --env_id PickCube-v1 --num_resets 1
    python scripts/run_eval.py --policy_server_addr localhost:8765 \
        --env_id PickCube-v1 --num_trials 1

The `action_dim` advertised in metadata MUST match what the env expects
(env.action_space.shape[0]), or clients will raise on the handshake guard.
ManiSkill's default PickCube-v1 + Panda + pd_joint_delta_pos controller has
action_dim=8 (7 joint deltas + 1 gripper). Override the ACTION_DIM constant
below if you switch controller or robot.
"""

import argparse
import logging
from typing import Dict

import numpy as np

from policy_websocket import BasePolicy, WebsocketPolicyServer


logger = logging.getLogger(__name__)

ACTION_DIM = 8


def _sample_action() -> np.ndarray:
    """Produce a safe in-range action of shape (ACTION_DIM,).

    ManiSkill PickCube-v1 + Panda + pd_joint_delta_pos accepts deltas in
    roughly [-1, 1]; a small uniform sample never exceeds joint limits.
    """
    return np.random.uniform(-0.1, 0.1, ACTION_DIM).astype(np.float32)


class RandomPolicy(BasePolicy):
    """Returns safe random actions for ManiSkill via `_sample_action`."""

    def __init__(self) -> None:
        pass

    def infer(self, obs: Dict) -> Dict:
        # Handle the init / handshake payload that run_demo/run_eval send
        # before any rollout. Clients pass `action_dim`/`task_name`/... as
        # the first obs; just echo an action of the right shape.
        action = np.asarray(_sample_action(), dtype=np.float32)
        assert action.shape[0] == ACTION_DIM, (
            f"RandomPolicy produced action of shape {action.shape}; "
            f"expected ({ACTION_DIM},)"
        )
        return {"actions": action}

    def reset(self) -> None:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="ManiSkill test policy server (random actions, headless)"
    )
    parser.add_argument("--host", default="0.0.0.0",
                        help="Bind address; default 0.0.0.0 (all interfaces)")
    parser.add_argument("--port", type=int, default=8765,
                        help="TCP port; default 8765 (matches policy_websocket library default)")
    args = parser.parse_args()

    policy = RandomPolicy()
    metadata = {"policy_name": "RandomPolicy(ManiSkill)", "action_dim": ACTION_DIM}

    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=metadata,
    )
    print(f"Starting ManiSkill RandomPolicy server on ws://{args.host}:{args.port}")
    print(f"Advertising action_dim={ACTION_DIM}. Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    print("Server stopped, port released.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    main()
