#!/usr/bin/env python3
"""
run_demo.py — Run policy in ManiSkill simulation (no eval).

Connects to a Policy Server over WebSocket for action inference.
Client sends ManiSkill obs converted to policy-expected format.
For demo only: no eval logs or success-rate tracking. Default: headless, saves videos to demo_log/.

Runnable task count is dynamic (REGISTERED_ENVS minus EVAL_TASK_BLACKLIST in mani_skill.utils.run_utils).
Print the current list with: python scripts/run_demo.py --list_tasks

Usage:
    python scripts/run_demo.py --env_id PickCube-v1 --policy_server_addr localhost:8765
    python scripts/run_demo.py --gui --env_id PickCube-v1 --policy_server_addr localhost:8765
    python scripts/run_demo.py --list_tasks

Example
-------
    python scripts/run_demo.py --policy_server_addr localhost:8765 \
        --env_id PickCube-v1 --num_resets 1
"""

import argparse
import sys
from pathlib import Path

# Ensure mani_skill is importable when run from scripts/
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import atexit
import os
import signal
import sys
import time
from datetime import datetime

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

def _websocket_policy_class():
    try:
        from policy_websocket import WebsocketClientPolicy
    except ImportError as e:
        raise ImportError(
            "policy-websocket is required for run_demo. "
            "Install with: pip install policy-websocket "
            "or: pip install 'policy-websocket @ git+https://github.com/YufengJin/policy_websocket.git'"
        ) from e
    return WebsocketClientPolicy


def log(msg: str, log_file=None):
    """Print a message and optionally write it to a log file (flush after each write)."""
    print(msg)
    if log_file is not None:
        log_file.write(msg + "\n")
        log_file.flush()


def _create_env(
    args,
    seed=None,
    episode_idx=None,
    use_gui: bool = False,
    control_mode=None,
):
    """Create a ManiSkill environment for demo (optionally with GUI)."""
    render_mode = "human" if use_gui else None
    return create_maniskill_env(
        env_id=args.env_id,
        robot_uids=args.robot_uids,
        obs_mode=args.obs_mode,
        img_res=args.img_res,
        render_mode=render_mode,
        num_envs=1,
        seed=seed,
        control_mode=control_mode,
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
    import imageio

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


def run_episode(
    args,
    env,
    task_description,
    policy,
    episode_idx,
    use_gui: bool,
    save_video: bool = False,
    num_wait_steps: int = 10,
    init_obs=None,
):
    """Run one episode: obs -> policy -> step loop with optional rendering."""
    action_dim, action_low, action_high = get_action_spec(env)
    dummy = np.zeros(action_dim, dtype=np.float64)
    obs = init_obs
    for _ in range(num_wait_steps):
        obs, _, _, _, _ = env.step(dummy)

    max_steps = get_task_max_steps(args.env_id, default_horizon=500)
    success = False
    episode_length = 0
    replay_primary, replay_secondary, replay_wrist = [], [], []

    for t in range(max_steps):
        observation = obs_to_policy_format(obs, task_description)
        # Inject mandatory __meta__ envelope (per benchmark contract).
        observation["__meta__"] = {
            "v": 1,
            "benchmark": "maniskill",
            "task": str(args.env_id),
            "task_description": task_description,
            "phase": "step",
        }
        if save_video and "robot0_agentview_left_image" in observation:
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
        if t % 50 == 0:
            print(f"  t={t}: infer {time.time() - start:.3f}s")

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

        if use_gui:
            env.render()

        if info.get("success") is not None:
            s = info["success"]
            if hasattr(s, "cpu"):
                s = s.cpu().numpy()
            success = bool(np.asarray(s).ravel()[0])
            if success:
                print(f"  Success at t={t}!")
                break
        if terminated or truncated:
            break

    print(
        f"  Episode {episode_idx}: {'SUCCESS' if success else 'FAILURE'} (length={episode_length})"
    )
    return success, episode_length, replay_primary, replay_secondary, replay_wrist


def parse_args():
    all_tasks = get_available_tasks()
    parser = argparse.ArgumentParser(
        description="ManiSkill demo: run policy in sim via WebSocket (no eval)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--policy_server_addr",
        type=str,
        default="localhost:8765",
        help="WebSocket policy server address host:port (default 8765 = policy_websocket lib default)",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="randomPolicy",
        help="Policy name (for display)",
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
        "--num_resets",
        type=int,
        default=10,
        help="Number of scene resets",
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
    )
    parser.add_argument(
        "--no_deterministic",
        action="store_false",
        dest="deterministic",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Enable interactive GUI rendering (default: headless no_gui)",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs",
        help="Per-run demo dir: <log_dir>/demo/<env_id>--<YYYYMMDD_HHMMSS>/",
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        default=True,
        help="Save MP4 in no_gui mode (default on; requires imageio[ffmpeg])",
    )
    parser.add_argument(
        "--no_save_video",
        action="store_false",
        dest="save_video",
        help="Disable MP4 capture",
    )
    parser.add_argument(
        "--list_tasks",
        action="store_true",
        help="Print runnable env_id count and sorted list, then exit",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    all_tasks = sorted(get_available_tasks())
    if args.list_tasks:
        print(f"ManiSkill runnable tasks: {len(all_tasks)} (see EVAL_TASK_BLACKLIST in run_utils.py)")
        for eid in all_tasks:
            print(f"  {eid}")
        return
    all_tasks_set = set(all_tasks)
    if args.env_id not in all_tasks_set:
        raise ValueError(
            f"Unknown env_id: {args.env_id}. Available: {all_tasks}"
        )

    set_seed_everywhere(args.seed, deterministic=args.deterministic)
    use_gui = args.gui
    save_video = args.save_video and not use_gui
    task_description = get_task_description(args.env_id)

    addr = args.policy_server_addr
    if ":" in addr:
        host, port = addr.rsplit(":", 1)
        port = int(port)
    else:
        host, port = addr, 8765

    # Per-run log dir (always created, contains demo.log + MP4s when --save_video on)
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.log_dir, "demo", f"{args.env_id}--{date_str}")
    os.makedirs(run_dir, exist_ok=True)
    args._run_dir = run_dir
    log_path = os.path.join(run_dir, "demo.log")
    log_file = open(log_path, "w", encoding="utf-8")

    log("=" * 60, log_file)
    log("ManiSkill Demo (run policy in sim, no eval)", log_file)
    log("=" * 60, log_file)
    log(f"  env_id:        {args.env_id}", log_file)
    log(f"  num_resets:    {args.num_resets}", log_file)
    log(f"  policy:        {args.policy}", log_file)
    log(f"  robot_uids:    {args.robot_uids}", log_file)
    log(f"  obs_mode:      {args.obs_mode}", log_file)
    log(f"  img_res:       {args.img_res}", log_file)
    log(f"  seed:          {args.seed}", log_file)
    log(f"  deterministic: {args.deterministic}", log_file)
    log(f"  policy_server: ws://{host}:{port}", log_file)
    log(f"  GUI:           {'on (--gui)' if use_gui else 'off (headless)'}", log_file)
    log(f"  save_video:    {save_video}", log_file)
    log(f"  log_dir:       {run_dir}", log_file)
    log(f"  log_file:      {log_path}", log_file)
    log("=" * 60, log_file)

    WebsocketClientPolicy = _websocket_policy_class()
    policy = WebsocketClientPolicy(host=host, port=port)
    metadata = policy.get_server_metadata()
    log(f"Server metadata: {metadata}", log_file)
    replay_mode = metadata.get("replay_mode") or metadata.get("policy_name") == "ReplayPolicy"
    if replay_mode:
        log("Replay mode: using trajectory reset_kwargs, control_mode, num_wait_steps=0", log_file)
    if metadata.get("env_id") and metadata["env_id"] != args.env_id:
        log(
            f"WARNING: Server env_id={metadata['env_id']} but run_demo env_id={args.env_id}. "
            "They must match for replay_demo.",
            log_file,
        )

    env = None
    _log_closed = {"closed": False}

    def _close_log():
        if not _log_closed["closed"]:
            try:
                log_file.close()
            except Exception:
                pass
            _log_closed["closed"] = True

    def _cleanup(signum=None, frame=None):
        log("\nCleaning up ...", log_file if not _log_closed["closed"] else None)
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        try:
            policy.close()
        except Exception:
            pass
        _close_log()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1 if signum else 0)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)
    atexit.register(_close_log)
    atexit.register(policy.close)

    try:
        for ep_idx in range(args.num_resets):
            print(f"\n--- Reset {ep_idx + 1}/{args.num_resets} ---")

            if replay_mode:
                # Get trajectory's reset_kwargs and control_mode (server advances to next traj)
                reset_resp = policy.infer(
                    {"__get_reset_params__": True, "task_name": args.env_id}
                )
                reset_kwargs = reset_resp["reset_kwargs"]
                control_mode = reset_resp["control_mode"]
                env = _create_env(
                    args,
                    seed=None,
                    episode_idx=ep_idx,
                    use_gui=use_gui,
                    control_mode=control_mode,
                )
                obs, _ = env.reset(**reset_kwargs)
                num_wait_steps = 0
            else:
                seed = args.seed * (ep_idx + 1) * 256 if args.deterministic else None
                env = _create_env(
                    args, seed=seed, episode_idx=ep_idx, use_gui=use_gui
                )
                obs, _ = env.reset(seed=seed)
                num_wait_steps = 10
                policy.reset()

            print(f"Task: {task_description}")

            action_dim, action_low, action_high = get_action_spec(env)
            init_obs = {
                "action_dim": action_dim,
                "action_low": action_low,
                "action_high": action_high,
                "task_name": args.env_id,
                "task_description": task_description,
                "__meta__": {
                    "v": 1,
                    "benchmark": "maniskill",
                    "task": str(args.env_id),
                    "task_description": task_description,
                    "phase": "init",
                },
            }
            policy.infer(init_obs)

            success, ep_len, rep_p, rep_s, rep_w = run_episode(
                args,
                env,
                task_description,
                policy,
                ep_idx,
                use_gui,
                save_video=save_video,
                num_wait_steps=num_wait_steps,
                init_obs=obs,
            )
            if save_video and rep_p and rep_s and rep_w:
                save_rollout_video(
                    rep_p,
                    rep_s,
                    rep_w,
                    ep_idx,
                    success,
                    task_description,
                    output_dir=getattr(args, "_run_dir", args.log_dir),
                )
            env.close()
            env = None
    finally:
        try:
            policy.close()
        except Exception:
            pass
        _close_log()

    log("\nDone.", log_file if not _log_closed["closed"] else None)


if __name__ == "__main__":
    main()
