from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np

try:
    from .config import Config
    from .custom_sac import CustomSACAgent
    from .utils import (
        checkpoint_paths,
        compute_final_distance,
        load_config_snapshot,
        load_json,
        resolve_run_dir,
        save_json,
        video_path,
    )
except ImportError:
    from config import Config
    from custom_sac import CustomSACAgent
    from utils import (
        checkpoint_paths,
        compute_final_distance,
        load_config_snapshot,
        load_json,
        resolve_run_dir,
        save_json,
        video_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a custom PyTorch SAC checkpoint on Pusher-v5.")
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--checkpoint", choices=["best", "last", "both"], default="last")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--video-episodes", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    return parser.parse_args()


def load_config_for_run(run_dir: Path) -> Config:
    config_dict = load_config_snapshot(run_dir)
    if not config_dict:
        raise FileNotFoundError(f"config.json not found in {run_dir}")
    return Config(**config_dict)


def maybe_record_frame(writer: imageio.Writer | None, env: gym.Env) -> None:
    if writer is None:
        return
    frame = env.render()
    if frame is not None:
        writer.append_data(frame)


def evaluate_checkpoint(
    run_dir: Path,
    cfg: Config,
    checkpoint: str,
    n_episodes: int,
    record_video: bool,
) -> dict[str, float | int | list[float] | str]:
    paths = checkpoint_paths(run_dir, checkpoint)
    if not paths["model"].exists():
        raise FileNotFoundError(f"Checkpoint not found: {paths['model']}")

    render_mode = "rgb_array" if record_video else None
    env = gym.make(
        cfg.env_id,
        render_mode=render_mode,
        max_episode_steps=cfg.max_episode_steps,
    )
    agent = CustomSACAgent(env.observation_space, env.action_space, cfg)
    agent.load_checkpoint(paths["model"])

    writer = None
    if record_video:
        writer = imageio.get_writer(video_path(run_dir, checkpoint), fps=cfg.video_fps, macro_block_size=None)

    returns: list[float] = []
    final_distances: list[float] = []
    successes: list[float] = []

    try:
        for episode_idx in range(n_episodes):
            obs, _ = env.reset(seed=cfg.seed + episode_idx)
            record_this_episode = writer is not None and episode_idx < cfg.video_episodes
            if record_this_episode:
                maybe_record_frame(writer, env)
            episode_return = 0.0
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action = agent.select_action(obs, deterministic=cfg.eval_deterministic)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)
                if record_this_episode:
                    maybe_record_frame(writer, env)
            final_distance = compute_final_distance(obs)
            returns.append(float(episode_return))
            final_distances.append(float(final_distance))
            successes.append(float(final_distance < cfg.success_threshold))
    finally:
        if writer is not None:
            writer.close()
        env.close()

    return {
        "checkpoint": checkpoint,
        "n_episodes": int(n_episodes),
        "average_return": float(np.mean(returns)),
        "std_return": float(np.std(returns, ddof=0)),
        "mean_final_distance": float(np.mean(final_distances)),
        "std_final_distance": float(np.std(final_distances, ddof=0)),
        "success_rate": float(np.mean(successes)),
        "episode_returns": returns,
        "final_distances": final_distances,
    }


def evaluate_checkpoints(
    run_dir: str | Path,
    cfg: Config,
    checkpoints: Iterable[str],
    record_video: bool,
    save_path: str | Path | None = None,
) -> dict[str, dict]:
    run_path = Path(run_dir)
    results: dict[str, dict] = {}
    for checkpoint in checkpoints:
        results[checkpoint] = evaluate_checkpoint(
            run_dir=run_path,
            cfg=cfg,
            checkpoint=checkpoint,
            n_episodes=cfg.n_eval_episodes,
            record_video=record_video,
        )
    if save_path is not None:
        existing = load_json(save_path, default={}) or {}
        existing.update(results)
        save_json(save_path, existing)
    return results


def main() -> dict[str, dict]:
    args = parse_args()
    run_dir = resolve_run_dir(args.run_dir, Config().artifacts_root)
    cfg = load_config_for_run(run_dir)
    if args.episodes is not None:
        cfg.n_eval_episodes = args.episodes
    if args.max_episode_steps is not None:
        cfg.max_episode_steps = args.max_episode_steps
    if args.video_episodes is not None:
        cfg.video_episodes = args.video_episodes
    if args.fps is not None:
        cfg.video_fps = args.fps
    checkpoints = ("best", "last") if args.checkpoint == "both" else (args.checkpoint,)
    results = evaluate_checkpoints(
        run_dir=run_dir,
        cfg=cfg,
        checkpoints=checkpoints,
        record_video=args.video,
        save_path=run_dir / "eval.json",
    )
    print(f"Evaluation saved to {run_dir / 'eval.json'}")
    for name, metrics in results.items():
        print(
            f"{name}: avg_return={metrics['average_return']:.3f}, "
            f"std_return={metrics['std_return']:.3f}, "
            f"mean_final_distance={metrics['mean_final_distance']:.4f}, "
            f"success_rate={metrics['success_rate']:.3f}"
        )
    return results


if __name__ == "__main__":
    main()
