#!/usr/bin/env python3
"""
run_eval.py — ManiSkill evaluation client (WebSocket).

Runs the ManiSkill simulation loop and delegates action inference to a remote
Policy Server over WebSocket. Client sends ManiSkill obs converted to policy format.
Logs and videos are written to: <log_dir>/<env_id>--<YYYYMMDD_HHMMSS>/

Usage:
    python scripts/run_eval.py --env_id PickCube-v1 --policy_server_addr localhost:8000
"""

import argparse
import atexit
import sys
from pathlib import Path

# Ensure mani_skill is importable when run from scripts/
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import os
import signal
import time
from datetime import datetime

import imageio
import numpy as np

from mani_skill.utils.run_utils import (
    create_maniskill_env,
    get_action_spec,
    get_available_tasks,
    get_task_description,
    get_task_max_steps,
    obs_to_policy_format,
    set_seed_everywhere,
)

try:
    from policy_websocket import WebsocketClientPolicy
except ImportError:
    raise ImportError(
        "policy-websocket is required for run_eval. "
        "Install with: pip install policy-websocket "
        "or: pip install 'policy-websocket @ git+https://github.com/YufengJin/policy_websocket.git'"
    )


def log(msg: str, log_file=None):
    """Print a message and optionally write it to a log file."""
    print(msg)
    if log_file is not None:
        log_file.write(msg + "\n")
        log_file.flush()


def _create_env(args, seed=None, episode_idx=None):
    """Create a ManiSkill environment for eval (no GUI)."""
    return create_maniskill_env(
        env_id=args.env_id,
        robot_uids=args.robot_uids,
        obs_mode=args.obs_mode,
        img_res=args.img_res,
        render_mode=None,
        num_envs=1,
        seed=seed,
    )


def save_rollout_video(
    primary_images,
    secondary_images,
    wrist_images,
    episode_idx,
    success,
    task_description,
    output_dir,
):
    """Save a concatenated MP4 of primary | secondary | wrist camera views."""
    os.makedirs(output_dir, exist_ok=True)
    tag = (
        task_description.lower()
        .replace(" ", "_")
        .replace("\n", "_")
        .replace(".", "_")[:40]
    )
    filename = f"episode={episode_idx}--success={success}--task={tag}.mp4"
    mp4_path = os.path.join(output_dir, filename)
    writer = imageio.get_writer(mp4_path, fps=30, format="FFMPEG", codec="libx264")
    for p, s, w in zip(primary_images, secondary_images, wrist_images):
        frame = np.concatenate([p, s, w], axis=1)
        writer.append_data(frame)
    writer.close()
    print(f"Saved rollout video: {mp4_path}")
    return mp4_path


def run_episode(args, env, task_description, policy, episode_idx, log_file=None):
    """Run a single evaluation episode via WebSocket policy.infer()."""
    NUM_WAIT_STEPS = 10
    action_dim, action_low, action_high = get_action_spec(env)
    dummy = np.zeros(action_dim, dtype=np.float64)
    for _ in range(NUM_WAIT_STEPS):
        obs, _, _, _, _ = env.step(dummy)

    max_steps = get_task_max_steps(args.env_id, default_horizon=500)
    success = False
    episode_length = 0
    replay_primary, replay_secondary, replay_wrist = [], [], []

    for t in range(max_steps):
        observation = obs_to_policy_format(obs, task_description)
        if "robot0_agentview_left_image" in observation:
            p = observation["robot0_agentview_left_image"]
            s = observation.get("robot0_agentview_right_image", p)
            w = observation.get("robot0_eye_in_hand_image", p)
            replay_primary.append(np.array(p, copy=True))
            replay_secondary.append(np.array(s, copy=True))
            replay_wrist.append(np.array(w, copy=True))

        start = time.time()
        result = policy.infer(observation)
        action = result["actions"]
        if hasattr(action, "ndim") and action.ndim > 1:
            action = action[0]
        query_time = time.time() - start

        if t % 50 == 0:
            log(f"  t={t}: infer {query_time:.3f}s, action[:4]={action[:4]}", log_file)

        action = np.asarray(action, dtype=np.float64)
        if action.shape[-1] != action_dim:
            action = np.pad(
                action,
                (0, max(0, action_dim - action.shape[-1])),
                mode="constant",
                constant_values=0,
            )[:action_dim]

        obs, reward, terminated, truncated, info = env.step(action)
        episode_length += 1

        if info.get("success") is not None:
            s = info["success"]
            if hasattr(s, "cpu"):
                s = s.cpu().numpy()
            success = bool(np.asarray(s).ravel()[0])
            if success:
                log(f"  Success at t={t}!", log_file)
                break

        if terminated or truncated:
            break

    log(
        f"  Episode {episode_idx}: {'SUCCESS' if success else 'FAILURE'} "
        f"(length={episode_length})",
        log_file,
    )
    return success, episode_length, replay_primary, replay_secondary, replay_wrist


def run_task(args, policy, log_file=None):
    """Evaluate a task over multiple episodes and report success rate."""
    log(f"\nEvaluating task: {args.env_id}", log_file)

    successes = []
    lengths = []

    for ep_idx in range(args.num_trials):
        log(f"\n--- Episode {ep_idx + 1}/{args.num_trials} ---", log_file)

        seed = args.seed * (ep_idx + 1) * 256 if args.deterministic else None
        env = _create_env(args, seed=seed, episode_idx=ep_idx)
        obs, _ = env.reset(seed=seed)

        task_description = get_task_description(args.env_id)
        log(f"Task description: {task_description}", log_file)

        policy.reset()
        action_dim, action_low, action_high = get_action_spec(env)
        init_obs = {
            "action_dim": action_dim,
            "action_low": action_low,
            "action_high": action_high,
            "task_name": args.env_id,
            "task_description": task_description,
        }
        policy.infer(init_obs)

        success, length, rep_p, rep_s, rep_w = run_episode(
            args, env, task_description, policy, ep_idx, log_file
        )
        successes.append(success)
        lengths.append(length)

        if args.save_video and rep_p and rep_s and rep_w:
            save_rollout_video(
                rep_p,
                rep_s,
                rep_w,
                ep_idx,
                success,
                task_description,
                output_dir=args.log_dir,
            )

        env.close()

        sr = sum(successes) / len(successes) * 100
        log(
            f"Running success rate: {sum(successes)}/{len(successes)} ({sr:.1f}%)",
            log_file,
        )

    success_rate = np.mean(successes)
    avg_length = np.mean(lengths)
    log("\n" + "=" * 60, log_file)
    log("FINAL RESULTS", log_file)
    log("=" * 60, log_file)
    log(f"Policy:           {args.policy}", log_file)
    log(f"Task:             {args.env_id}", log_file)
    log(f"Success rate:     {success_rate:.4f} ({int(success_rate * 100)}%)", log_file)
    log(f"Avg ep length:    {avg_length:.1f}", log_file)
    log(f"Total episodes:   {len(successes)}", log_file)
    log(f"Total successes:  {sum(successes)}", log_file)
    log("=" * 60, log_file)
    return success_rate


def parse_args():
    all_tasks = get_available_tasks()
    parser = argparse.ArgumentParser(
        description="ManiSkill WebSocket evaluation client",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--policy_server_addr",
        type=str,
        default="localhost:8000",
        help="Address of the WebSocket policy server (host:port)",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="randomPolicy",
        help="Policy name for logging (e.g. randomPolicy, cosmos)",
    )
    parser.add_argument(
        "--env_id",
        type=str,
        default="PickCube-v1",
        choices=all_tasks,
        help="ManiSkill environment ID",
    )
    parser.add_argument(
        "--robot_uids",
        type=str,
        default="panda",
        help="Robot UID (e.g. panda, xarm6_robotiq)",
    )
    parser.add_argument(
        "--obs_mode",
        type=str,
        default="rgbd",
        help="Observation mode (rgbd, rgb, state, etc.)",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=5,
        help="Number of evaluation episodes per task",
    )
    parser.add_argument(
        "--img_res",
        type=int,
        default=224,
        help="Camera image resolution (square)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=195,
        help="Random seed",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic seeding",
    )
    parser.add_argument(
        "--no_deterministic",
        action="store_false",
        dest="deterministic",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./eval_logs",
        help="Directory for logs and rollout videos",
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        default=True,
        help="Save rollout videos",
    )
    parser.add_argument(
        "--no_save_video",
        action="store_false",
        dest="save_video",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    all_tasks = set(get_available_tasks())
    if args.env_id not in all_tasks:
        raise ValueError(
            f"Unknown env_id: {args.env_id}. Available: {sorted(all_tasks)}"
        )

    set_seed_everywhere(args.seed, deterministic=args.deterministic)
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.log_dir, f"{args.env_id}--{date_str}")
    os.makedirs(run_dir, exist_ok=True)
    args.log_dir = run_dir

    log_path = os.path.join(run_dir, "eval.log")
    log_file = open(log_path, "w")
    log("=" * 60, log_file)
    log("ManiSkill WebSocket Eval Run", log_file)
    log("=" * 60, log_file)
    log(f"  policy:           {args.policy}", log_file)
    log(f"  env_id:           {args.env_id}", log_file)
    log(f"  num_trials:       {args.num_trials}", log_file)
    log(f"  seed:             {args.seed}", log_file)
    log(f"  deterministic:    {args.deterministic}", log_file)
    log(f"  policy_server:    {args.policy_server_addr}", log_file)
    log(f"  log_dir (run_dir): {run_dir}", log_file)
    log(f"  img_res:          {args.img_res}", log_file)
    log(f"  robot_uids:       {args.robot_uids}", log_file)
    log("=" * 60, log_file)
    log("", log_file)

    addr = args.policy_server_addr
    if ":" in addr:
        host, port = addr.rsplit(":", 1)
        port = int(port)
    else:
        host, port = addr, 8000

    log(f"Connecting to policy server at ws://{host}:{port} ...", log_file)
    policy = WebsocketClientPolicy(host=host, port=port)
    metadata = policy.get_server_metadata()
    log(f"Server metadata: {metadata}", log_file)

    def _cleanup(signum=None, frame=None):
        print("\nCleaning up ...", flush=True)
        policy.close()
        if not log_file.closed:
            log_file.close()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1 if signum else 0)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)
    atexit.register(policy.close)

    try:
        success_rate = run_task(args, policy, log_file)
        log(f"\nLog saved to: {log_path}", log_file)
        print(f"\nLog saved to: {log_path}")
        print(f"Run directory (logs + videos): {run_dir}")
        return success_rate
    finally:
        policy.close()
        if not log_file.closed:
            log_file.close()


if __name__ == "__main__":
    main()
