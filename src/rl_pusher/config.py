from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    env_id: str = "Pusher-v5"
    seed: int = 42
    total_timesteps: int = 300_000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    buffer_size: int = 200_000
    batch_size: int = 256
    tau: float = 0.005
    train_freq: int = 1
    gradient_steps: int = 1
    learning_starts: int = 1_000
    hidden_sizes: tuple[int, int] = (256, 256)
    load_mode: str = "none"
    source_run_dir: str | None = None
    n_eval_episodes: int = 20
    success_threshold: float = 0.05
    record_video: bool = True
    video_episodes: int = 3
    min_video_episodes: int = 10
    video_fps: int = 20
    rolling_window: int = 20
    artifacts_root: str = "artifacts"
    run_name: str = "sac_pusher"
    device: str = "cpu"
    eval_deterministic: bool = True
    save_replay_buffer: bool = True
