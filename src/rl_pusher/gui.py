from __future__ import annotations

import argparse
import select
import sys
import time
from pathlib import Path

import gymnasium as gym
<<<<<<< HEAD
from stable_baselines3 import SAC

try:
    from .config import Config
    from .utils import checkpoint_paths, compute_final_distance, load_config_snapshot, resolve_run_dir
except ImportError:
    from config import Config
=======

try:
    from .config import Config
    from .custom_sac import CustomSACAgent
    from .utils import checkpoint_paths, compute_final_distance, load_config_snapshot, resolve_run_dir
except ImportError:
    from config import Config
    from custom_sac import CustomSACAgent
>>>>>>> d65a859ed45d7f3d5bdc83cb1279b522b7a16e68
    from utils import checkpoint_paths, compute_final_distance, load_config_snapshot, resolve_run_dir


def parse_args() -> argparse.Namespace:
<<<<<<< HEAD
    parser = argparse.ArgumentParser(description="Minimal interactive viewer for Pusher-v5.")
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--checkpoint", choices=["best", "last"], default="last")
=======
    parser = argparse.ArgumentParser(description="Minimal interactive viewer for custom PyTorch SAC on Pusher-v5.")
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--checkpoint", choices=["best", "last"], default="last")
    parser.add_argument("--max-episode-steps", type=int, default=None)
>>>>>>> d65a859ed45d7f3d5bdc83cb1279b522b7a16e68
    return parser.parse_args()


def load_config_for_run(run_dir: Path) -> Config:
    config_dict = load_config_snapshot(run_dir)
    if not config_dict:
        raise FileNotFoundError(f"config.json not found in {run_dir}")
    return Config(**config_dict)


def read_non_blocking_command() -> str | None:
    ready, _, _ = select.select([sys.stdin], [], [], 0.01)
    if not ready:
        return None
    return sys.stdin.readline().strip().lower()


def format_action(action) -> str:
    return "[" + ", ".join(f"{float(value): .3f}" for value in action) + "]"


<<<<<<< HEAD
def step_policy(model: SAC, env: gym.Env, obs, cumulative_reward: float) -> tuple[object, float, bool]:
    action, _ = model.predict(obs, deterministic=True)
=======
def step_policy(agent: CustomSACAgent, env: gym.Env, obs, cumulative_reward: float) -> tuple[object, float, bool]:
    action = agent.select_action(obs, deterministic=True)
>>>>>>> d65a859ed45d7f3d5bdc83cb1279b522b7a16e68
    next_obs, reward, terminated, truncated, _ = env.step(action)
    cumulative_reward += float(reward)
    distance = compute_final_distance(next_obs)
    print(
        f"reward={float(reward): .3f} "
        f"cumulative_reward={cumulative_reward: .3f} "
        f"distance={distance: .4f} "
        f"action={format_action(action)}"
    )
    done = bool(terminated or truncated)
    if done:
        print("Episode finished. Auto mode disabled.")
    return next_obs, cumulative_reward, done


def main() -> None:
    args = parse_args()
    run_dir = resolve_run_dir(args.run_dir, Config().artifacts_root)
    cfg = load_config_for_run(run_dir)
<<<<<<< HEAD
=======
    if args.max_episode_steps is not None:
        cfg.max_episode_steps = args.max_episode_steps
>>>>>>> d65a859ed45d7f3d5bdc83cb1279b522b7a16e68
    model_path = checkpoint_paths(run_dir, args.checkpoint)["model"]
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

<<<<<<< HEAD
    env = gym.make(cfg.env_id, render_mode="human")
    model = SAC.load(str(model_path), env=env, device=cfg.device)
=======
    env = gym.make(
        cfg.env_id,
        render_mode="human",
        max_episode_steps=cfg.max_episode_steps,
    )
    agent = CustomSACAgent(env.observation_space, env.action_space, cfg)
    agent.load_checkpoint(model_path)
>>>>>>> d65a859ed45d7f3d5bdc83cb1279b522b7a16e68
    obs, _ = env.reset(seed=cfg.seed)
    cumulative_reward = 0.0
    auto_mode = False

    print("Commands: s=step, a=toggle auto, r=reset, q=quit")
    try:
        while True:
            if auto_mode:
                command = read_non_blocking_command()
                if command == "a":
                    auto_mode = False
                    print("Auto mode disabled.")
                    continue
                if command == "r":
                    obs, _ = env.reset()
                    cumulative_reward = 0.0
                    auto_mode = False
                    print("Environment reset.")
                    continue
                if command == "q":
                    break
<<<<<<< HEAD
                obs, cumulative_reward, done = step_policy(model, env, obs, cumulative_reward)
=======
                obs, cumulative_reward, done = step_policy(agent, env, obs, cumulative_reward)
>>>>>>> d65a859ed45d7f3d5bdc83cb1279b522b7a16e68
                if done:
                    auto_mode = False
                time.sleep(0.05)
                continue

            command = input("> ").strip().lower()
            if command == "q":
                break
            if command == "r":
                obs, _ = env.reset()
                cumulative_reward = 0.0
                print("Environment reset.")
                continue
            if command == "a":
                auto_mode = True
                print("Auto mode enabled. Press Enter+a to stop, Enter+r to reset, Enter+q to quit.")
                continue
            if command == "s":
<<<<<<< HEAD
                obs, cumulative_reward, _ = step_policy(model, env, obs, cumulative_reward)
=======
                obs, cumulative_reward, _ = step_policy(agent, env, obs, cumulative_reward)
>>>>>>> d65a859ed45d7f3d5bdc83cb1279b522b7a16e68
                continue
            print("Unknown command. Use s, a, r or q.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
