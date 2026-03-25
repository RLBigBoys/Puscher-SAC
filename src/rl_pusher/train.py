from __future__ import annotations

import argparse
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import gymnasium as gym
import numpy as np

try:
    from .config import Config
    from .custom_sac import CustomSACAgent, ReplayBuffer
    from .eval_custom import evaluate_checkpoints
    from .model import describe_policy
    from .utils import (
        checkpoint_paths,
        compute_final_distance,
        compute_objective_metric,
        copy_checkpoint_if_exists,
        create_run_dir,
        load_config_snapshot,
        load_json,
        plot_training_curve,
        rolling_mean_std,
        resolve_run_dir,
        save_config_snapshot,
        save_json,
    )
except ImportError:
    from config import Config
    from custom_sac import CustomSACAgent, ReplayBuffer
    from eval_custom import evaluate_checkpoints
    from model import describe_policy
    from utils import (
        checkpoint_paths,
        compute_final_distance,
        compute_objective_metric,
        copy_checkpoint_if_exists,
        create_run_dir,
        load_config_snapshot,
        load_json,
        plot_training_curve,
        rolling_mean_std,
        resolve_run_dir,
        save_config_snapshot,
        save_json,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a custom PyTorch SAC agent on Pusher-v5.")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--load-mode", choices=["none", "best", "last"], default="none")
    parser.add_argument("--source-run-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> tuple[Config, Path | None]:
    source_run_dir: Path | None = None
    if args.load_mode == "none":
        cfg = Config()
    else:
        source_run_dir = resolve_run_dir(args.source_run_dir, Config().artifacts_root)
        source_config = load_config_snapshot(source_run_dir)
        cfg = Config(**source_config)
    cfg.load_mode = args.load_mode
    cfg.source_run_dir = str(source_run_dir) if source_run_dir else None
    cfg.run_name = args.run_name or ("custom_sac_pusher" if args.load_mode == "none" else cfg.run_name)
    if args.timesteps is not None:
        cfg.total_timesteps = args.timesteps
    if args.seed is not None:
        cfg.seed = args.seed
    if args.eval_episodes is not None:
        cfg.n_eval_episodes = args.eval_episodes
    if args.max_episode_steps is not None:
        cfg.max_episode_steps = args.max_episode_steps
    if args.no_video:
        cfg.record_video = False
    return cfg, source_run_dir


def make_train_env(cfg: Config) -> gym.Env:
    env = gym.make(cfg.env_id, max_episode_steps=cfg.max_episode_steps)
    env.reset(seed=cfg.seed)
    env.action_space.seed(cfg.seed)
    return env


def make_replay_buffer(env: gym.Env, cfg: Config) -> ReplayBuffer:
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    return ReplayBuffer(obs_dim=obs_dim, action_dim=action_dim, capacity=cfg.buffer_size)


def load_training_state(
    agent: CustomSACAgent,
    replay_buffer: ReplayBuffer,
    source_run_dir: Path,
    load_mode: str,
    save_replay_buffer: bool,
) -> dict[str, float | int]:
    paths = checkpoint_paths(source_run_dir, load_mode)
    if not paths["model"].exists():
        raise FileNotFoundError(f"Checkpoint not found: {paths['model']}")
    state = agent.load_checkpoint(paths["model"])
    if save_replay_buffer and paths["replay_buffer"].exists():
        replay_buffer.load(paths["replay_buffer"])
    return state


def save_checkpoint(
    agent: CustomSACAgent,
    replay_buffer: ReplayBuffer,
    run_dir: Path,
    tag: str,
    global_step: int,
    episode_idx: int,
    update_step: int,
    best_reward: float,
    save_replay_buffer: bool,
) -> None:
    paths = checkpoint_paths(run_dir, tag)
    agent.save_checkpoint(
        path=paths["model"],
        global_step=global_step,
        episode_idx=episode_idx,
        update_step=update_step,
        best_reward=best_reward,
    )
    agent.save_policy(paths["policy"])
    if save_replay_buffer:
        replay_buffer.save(paths["replay_buffer"])


def ensure_best_checkpoint(run_dir: Path) -> None:
    best_paths = checkpoint_paths(run_dir, "best")
    if best_paths["model"].exists():
        return
    last_paths = checkpoint_paths(run_dir, "last")
    for key, path in last_paths.items():
        if path.exists():
            shutil.copy2(path, best_paths[key])


def plot_training_curves(history: list[dict], run_dir: Path, cfg: Config) -> None:
    architecture = describe_policy(cfg)
    plot_training_curve(
        history=history,
        value_key="episode_reward",
        mean_key="reward_mean",
        std_key="reward_std",
        title=f"Custom SAC on {cfg.env_id} | {architecture} | Episode Reward",
        ylabel="Reward",
        path=run_dir / "reward_curve.png",
        window=cfg.rolling_window,
    )
    plot_training_curve(
        history=history,
        value_key="objective_metric",
        mean_key="objective_mean",
        std_key="objective_std",
        title=f"Custom SAC on {cfg.env_id} | {architecture} | Objective Metric",
        ylabel="Objective Metric",
        path=run_dir / "objective_curve.png",
        window=cfg.rolling_window,
    )


def print_progress(
    global_step: int,
    target_global_step: int,
    episode_idx: int,
    start_time: float,
    history: list[dict],
    update_metrics: dict[str, float],
) -> None:
    elapsed = max(time.time() - start_time, 1e-6)
    fps = global_step / elapsed
    if history:
        last = history[-1]
        reward_mean = float(last["reward_mean"])
        objective_mean = float(last["objective_mean"])
        episode_length = int(last["episode_length"])
    else:
        reward_mean = 0.0
        objective_mean = 0.0
        episode_length = 0
    print("---------------------------------")
    print(f"global_step        {global_step}/{target_global_step}")
    print(f"episodes           {episode_idx}")
    print(f"fps                {fps:.1f}")
    print(f"episode_len_last   {episode_length}")
    print(f"reward_mean        {reward_mean:.3f}")
    print(f"objective_mean     {objective_mean:.4f}")
    if update_metrics:
        print(f"actor_loss         {update_metrics['actor_loss']:.4f}")
        print(f"critic_loss        {update_metrics['critic_loss']:.4f}")
        print(f"alpha              {update_metrics['alpha']:.6f}")
        print(f"alpha_loss         {update_metrics['alpha_loss']:.4f}")
    print("---------------------------------")


def train() -> Path:
    args = parse_args()
    cfg, source_run_dir = build_config(args)
    run_dir = create_run_dir(cfg.artifacts_root, cfg.run_name)
    cfg.source_run_dir = str(source_run_dir) if source_run_dir else None
    save_config_snapshot(run_dir, asdict(cfg))

    history: list[dict] = []
    if source_run_dir is not None:
        history = list(load_json(source_run_dir / "train_history.json", default=[]))

    env = make_train_env(cfg)
    agent = CustomSACAgent(env.observation_space, env.action_space, cfg)
    replay_buffer = make_replay_buffer(env, cfg)

    global_step = max((int(item["global_step"]) for item in history), default=0)
    episode_idx = max((int(item["episode_idx"]) for item in history), default=0)
    update_step = 0
    best_reward = max((float(item["episode_reward"]) for item in history), default=float("-inf"))
    if source_run_dir is not None:
        state = load_training_state(agent, replay_buffer, source_run_dir, cfg.load_mode, cfg.save_replay_buffer)
        global_step = max(global_step, int(state["global_step"]))
        episode_idx = max(episode_idx, int(state["episode_idx"]))
        update_step = int(state["update_step"])
        best_reward = max(best_reward, float(state["best_reward"]))
        copy_checkpoint_if_exists(source_run_dir, run_dir, "best")

    target_global_step = global_step + cfg.total_timesteps
    start_time = time.time()
    last_log_step = global_step
    latest_update_metrics: dict[str, float] = {}

    print(f"Run directory: {run_dir}")
    print(f"Environment: {cfg.env_id}")
    print(f"Policy: {describe_policy(cfg)}")
    print("Algorithm: Custom SAC (PyTorch)")
    print(f"Episode horizon: {cfg.max_episode_steps}")
    if source_run_dir is not None:
        print(f"Resuming from: {source_run_dir} ({cfg.load_mode})")

    observation, _ = env.reset(seed=cfg.seed)
    episode_reward = 0.0
    episode_length = 0
    interrupted = False

    try:
        while global_step < target_global_step:
            if global_step < cfg.learning_starts:
                action = env.action_space.sample()
            else:
                action = agent.select_action(observation, deterministic=False)

            next_observation, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            replay_buffer.add(observation, action, reward, next_observation, done)

            observation = next_observation
            global_step += 1
            episode_reward += float(reward)
            episode_length += 1

            if (
                global_step >= cfg.learning_starts
                and replay_buffer.size >= cfg.batch_size
                and global_step % cfg.train_freq == 0
            ):
                for _ in range(cfg.gradient_steps):
                    latest_update_metrics = agent.update(replay_buffer.sample(cfg.batch_size))
                    update_step += 1

            if done:
                final_distance = compute_final_distance(next_observation)
                objective_metric = compute_objective_metric(next_observation)
                reward_series = [float(item["episode_reward"]) for item in history] + [episode_reward]
                objective_series = [float(item["objective_metric"]) for item in history] + [objective_metric]
                reward_mean, reward_std = rolling_mean_std(reward_series, cfg.rolling_window)
                objective_mean, objective_std = rolling_mean_std(objective_series, cfg.rolling_window)
                episode_idx += 1
                record = {
                    "episode_idx": episode_idx,
                    "global_step": global_step,
                    "episode_reward": float(episode_reward),
                    "episode_length": int(episode_length),
                    "objective_metric": float(objective_metric),
                    "final_distance": float(final_distance),
                    "success": bool(final_distance < cfg.success_threshold),
                    "reward_mean": reward_mean,
                    "reward_std": reward_std,
                    "objective_mean": objective_mean,
                    "objective_std": objective_std,
                    "reward_dist": float(info.get("reward_dist", objective_metric)),
                    "reward_near": float(info.get("reward_near", 0.0)),
                    "reward_ctrl": float(info.get("reward_ctrl", 0.0)),
                }
                history.append(record)
                save_json(run_dir / "train_history.json", history)

                if episode_reward > best_reward:
                    best_reward = float(episode_reward)
                    save_checkpoint(
                        agent=agent,
                        replay_buffer=replay_buffer,
                        run_dir=run_dir,
                        tag="best",
                        global_step=global_step,
                        episode_idx=episode_idx,
                        update_step=update_step,
                        best_reward=best_reward,
                        save_replay_buffer=cfg.save_replay_buffer,
                    )

                observation, _ = env.reset()
                episode_reward = 0.0
                episode_length = 0

            if global_step - last_log_step >= cfg.log_interval_steps:
                print_progress(
                    global_step=global_step,
                    target_global_step=target_global_step,
                    episode_idx=episode_idx,
                    start_time=start_time,
                    history=history,
                    update_metrics=latest_update_metrics,
                )
                last_log_step = global_step
    except KeyboardInterrupt:
        interrupted = True
        print("Training interrupted by user. Saving artifacts...")
    finally:
        save_checkpoint(
            agent=agent,
            replay_buffer=replay_buffer,
            run_dir=run_dir,
            tag="last",
            global_step=global_step,
            episode_idx=episode_idx,
            update_step=update_step,
            best_reward=best_reward,
            save_replay_buffer=cfg.save_replay_buffer,
        )
        ensure_best_checkpoint(run_dir)
        save_json(run_dir / "train_history.json", history)
        plot_training_curves(history, run_dir, cfg)
        env.close()
        try:
            evaluate_checkpoints(
                run_dir=run_dir,
                cfg=cfg,
                checkpoints=("best", "last"),
                record_video=cfg.record_video and len(history) >= cfg.min_video_episodes,
                save_path=run_dir / "eval.json",
            )
        except Exception as error:
            print(f"Evaluation skipped: {error}")
        if interrupted:
            print(f"Interrupted run saved to {run_dir}")
        else:
            print(f"Training finished. Artifacts saved to {run_dir}")
    return run_dir


if __name__ == "__main__":
    train()
