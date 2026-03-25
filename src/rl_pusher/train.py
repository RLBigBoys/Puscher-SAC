from __future__ import annotations

import argparse
import shutil
<<<<<<< HEAD
=======
import time
>>>>>>> d65a859ed45d7f3d5bdc83cb1279b522b7a16e68
from dataclasses import asdict
from pathlib import Path

import gymnasium as gym
<<<<<<< HEAD
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

try:
    from .config import Config
    from .eval import evaluate_checkpoints
    from .model import describe_policy, get_policy_kwargs
=======
import numpy as np

try:
    from .config import Config
    from .custom_sac import CustomSACAgent, ReplayBuffer
    from .eval_custom import evaluate_checkpoints
    from .model import describe_policy
>>>>>>> d65a859ed45d7f3d5bdc83cb1279b522b7a16e68
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
<<<<<<< HEAD
    from eval import evaluate_checkpoints
    from model import describe_policy, get_policy_kwargs
=======
    from custom_sac import CustomSACAgent, ReplayBuffer
    from eval_custom import evaluate_checkpoints
    from model import describe_policy
>>>>>>> d65a859ed45d7f3d5bdc83cb1279b522b7a16e68
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
<<<<<<< HEAD
    parser = argparse.ArgumentParser(description="Train SAC on Pusher-v5.")
=======
    parser = argparse.ArgumentParser(description="Train a custom PyTorch SAC agent on Pusher-v5.")
>>>>>>> d65a859ed45d7f3d5bdc83cb1279b522b7a16e68
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--load-mode", choices=["none", "best", "last"], default="none")
    parser.add_argument("--source-run-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
<<<<<<< HEAD
=======
    parser.add_argument("--max-episode-steps", type=int, default=None)
>>>>>>> d65a859ed45d7f3d5bdc83cb1279b522b7a16e68
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
<<<<<<< HEAD
    if args.timesteps is not None:
        cfg.total_timesteps = args.timesteps
    if args.run_name is not None:
        cfg.run_name = args.run_name
=======
    cfg.run_name = args.run_name or ("custom_sac_pusher" if args.load_mode == "none" else cfg.run_name)
    if args.timesteps is not None:
        cfg.total_timesteps = args.timesteps
>>>>>>> d65a859ed45d7f3d5bdc83cb1279b522b7a16e68
    if args.seed is not None:
        cfg.seed = args.seed
    if args.eval_episodes is not None:
        cfg.n_eval_episodes = args.eval_episodes
<<<<<<< HEAD
=======
    if args.max_episode_steps is not None:
        cfg.max_episode_steps = args.max_episode_steps
>>>>>>> d65a859ed45d7f3d5bdc83cb1279b522b7a16e68
    if args.no_video:
        cfg.record_video = False
    return cfg, source_run_dir


<<<<<<< HEAD
def make_train_env(cfg: Config) -> Monitor:
    env = gym.make(cfg.env_id)
    env.reset(seed=cfg.seed)
    env.action_space.seed(cfg.seed)
    return Monitor(env)


def build_new_model(cfg: Config, env: Monitor) -> SAC:
    return SAC(
        "MlpPolicy",
        env,
        learning_rate=cfg.learning_rate,
        buffer_size=cfg.buffer_size,
        batch_size=cfg.batch_size,
        gamma=cfg.gamma,
        tau=cfg.tau,
        train_freq=(cfg.train_freq, "step"),
        gradient_steps=cfg.gradient_steps,
        learning_starts=cfg.learning_starts,
        policy_kwargs=get_policy_kwargs(cfg),
        seed=cfg.seed,
        device=cfg.device,
        verbose=1,
    )


def load_model_for_resume(cfg: Config, env: Monitor, source_run_dir: Path) -> SAC:
    paths = checkpoint_paths(source_run_dir, cfg.load_mode)
    if not paths["model"].exists():
        raise FileNotFoundError(f"Checkpoint not found: {paths['model']}")
    model = SAC.load(str(paths["model"]), env=env, device=cfg.device, seed=cfg.seed)
    if cfg.save_replay_buffer and paths["replay_buffer"].exists():
        model.load_replay_buffer(str(paths["replay_buffer"]))
    return model


def save_checkpoint(model: SAC, run_dir: Path, tag: str, save_replay_buffer: bool) -> None:
    paths = checkpoint_paths(run_dir, tag)
    model.save(str(paths["model"]))
    torch.save(model.policy.state_dict(), paths["policy"])
    if save_replay_buffer:
        try:
            model.save_replay_buffer(str(paths["replay_buffer"]))
        except Exception:
            pass


class TrainingLoggerCallback(BaseCallback):
    def __init__(self, cfg: Config, run_dir: Path, history: list[dict], best_reward: float):
        super().__init__(verbose=0)
        self.cfg = cfg
        self.run_dir = run_dir
        self.history = history
        self.best_reward = float(best_reward)
        self.history_path = run_dir / "train_history.json"
        self.reward_series = [float(item["episode_reward"]) for item in history]
        self.objective_series = [float(item["objective_metric"]) for item in history]

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        dones = self.locals.get("dones")
        if infos is None or dones is None:
            return True
        for done, info in zip(dones, infos):
            if not bool(done):
                continue
            episode_info = info.get("episode")
            if episode_info is None:
                continue
            terminal_observation = info.get("terminal_observation")
            if terminal_observation is None:
                reward_dist = info.get("reward_dist")
                if reward_dist is None:
                    continue
                objective_metric = float(reward_dist)
                final_distance = -objective_metric
            else:
                final_distance = compute_final_distance(terminal_observation)
                objective_metric = compute_objective_metric(terminal_observation)
            episode_reward = float(episode_info["r"])
            self.reward_series.append(episode_reward)
            self.objective_series.append(objective_metric)
            reward_mean, reward_std = rolling_mean_std(self.reward_series, self.cfg.rolling_window)
            objective_mean, objective_std = rolling_mean_std(self.objective_series, self.cfg.rolling_window)
            record = {
                "episode_idx": len(self.history) + 1,
                "global_step": int(self.model.num_timesteps),
                "episode_reward": episode_reward,
                "objective_metric": float(objective_metric),
                "final_distance": float(final_distance),
                "success": bool(final_distance < self.cfg.success_threshold),
                "reward_mean": reward_mean,
                "reward_std": reward_std,
                "objective_mean": objective_mean,
                "objective_std": objective_std,
                "reward_dist": float(info.get("reward_dist", objective_metric)),
                "reward_near": float(info.get("reward_near", 0.0)),
                "reward_ctrl": float(info.get("reward_ctrl", 0.0)),
            }
            self.history.append(record)
            save_json(self.history_path, self.history)
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                save_checkpoint(self.model, self.run_dir, "best", self.cfg.save_replay_buffer)
        return True


def plot_training_curves(history: list[dict], run_dir: Path, cfg: Config) -> None:
    architecture = describe_policy(cfg)
    plot_training_curve(
        history=history,
        value_key="episode_reward",
        mean_key="reward_mean",
        std_key="reward_std",
        title=f"SAC on {cfg.env_id} | {architecture} | Episode Reward",
        ylabel="Reward",
        path=run_dir / "reward_curve.png",
        window=cfg.rolling_window,
    )
    plot_training_curve(
        history=history,
        value_key="objective_metric",
        mean_key="objective_mean",
        std_key="objective_std",
        title=f"SAC on {cfg.env_id} | {architecture} | Objective Metric",
        ylabel="Objective Metric",
        path=run_dir / "objective_curve.png",
        window=cfg.rolling_window,
    )
=======
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
>>>>>>> d65a859ed45d7f3d5bdc83cb1279b522b7a16e68


def ensure_best_checkpoint(run_dir: Path) -> None:
    best_paths = checkpoint_paths(run_dir, "best")
    if best_paths["model"].exists():
        return
    last_paths = checkpoint_paths(run_dir, "last")
    for key, path in last_paths.items():
        if path.exists():
            shutil.copy2(path, best_paths[key])


<<<<<<< HEAD
=======
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


>>>>>>> d65a859ed45d7f3d5bdc83cb1279b522b7a16e68
def train() -> Path:
    args = parse_args()
    cfg, source_run_dir = build_config(args)
    run_dir = create_run_dir(cfg.artifacts_root, cfg.run_name)
    cfg.source_run_dir = str(source_run_dir) if source_run_dir else None
    save_config_snapshot(run_dir, asdict(cfg))

    history: list[dict] = []
    if source_run_dir is not None:
        history = list(load_json(source_run_dir / "train_history.json", default=[]))
<<<<<<< HEAD
        copy_checkpoint_if_exists(source_run_dir, run_dir, "best")

    env = make_train_env(cfg)
    model = build_new_model(cfg, env) if source_run_dir is None else load_model_for_resume(cfg, env, source_run_dir)
    best_reward = max((float(item["episode_reward"]) for item in history), default=float("-inf"))
    callback = TrainingLoggerCallback(cfg, run_dir, history, best_reward)
=======

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
>>>>>>> d65a859ed45d7f3d5bdc83cb1279b522b7a16e68

    print(f"Run directory: {run_dir}")
    print(f"Environment: {cfg.env_id}")
    print(f"Policy: {describe_policy(cfg)}")
<<<<<<< HEAD
    if source_run_dir is not None:
        print(f"Resuming from: {source_run_dir} ({cfg.load_mode})")

    interrupted = False
    try:
        model.learn(total_timesteps=cfg.total_timesteps, callback=callback, reset_num_timesteps=source_run_dir is None)
=======
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
>>>>>>> d65a859ed45d7f3d5bdc83cb1279b522b7a16e68
    except KeyboardInterrupt:
        interrupted = True
        print("Training interrupted by user. Saving artifacts...")
    finally:
<<<<<<< HEAD
        save_checkpoint(model, run_dir, "last", cfg.save_replay_buffer)
=======
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
>>>>>>> d65a859ed45d7f3d5bdc83cb1279b522b7a16e68
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
