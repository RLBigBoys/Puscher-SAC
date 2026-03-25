from __future__ import annotations

import json
import platform
import shutil
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def make_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def runs_root(artifacts_root: str | Path) -> Path:
    return ensure_dir(Path(artifacts_root) / "runs")


def create_run_dir(artifacts_root: str | Path, run_name: str) -> Path:
    run_dir = ensure_dir(runs_root(artifacts_root) / f"{make_timestamp()}_{run_name}")
    write_latest_run(artifacts_root, run_dir)
    return run_dir


def latest_run_file(artifacts_root: str | Path) -> Path:
    return ensure_dir(artifacts_root) / "latest_run.txt"


def write_latest_run(artifacts_root: str | Path, run_dir: str | Path) -> None:
    latest_run_file(artifacts_root).write_text(str(Path(run_dir).resolve()), encoding="utf-8")


def read_latest_run(artifacts_root: str | Path) -> Path | None:
    file_path = latest_run_file(artifacts_root)
    if not file_path.exists():
        return None
    content = file_path.read_text(encoding="utf-8").strip()
    if not content:
        return None
    return Path(content)


def resolve_run_dir(run_dir: str | Path | None, artifacts_root: str | Path) -> Path:
    if run_dir:
        return Path(run_dir).expanduser().resolve()
    latest = read_latest_run(artifacts_root)
    if latest is None:
        raise FileNotFoundError("Run directory is not provided and artifacts/latest_run.txt is missing.")
    return latest


def to_serializable(value: Any) -> Any:
    if is_dataclass(value):
        return to_serializable(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def save_json(path: str | Path, data: Any) -> None:
    file_path = Path(path)
    ensure_dir(file_path.parent)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(to_serializable(data), handle, indent=2, ensure_ascii=False)


def load_json(path: str | Path, default: Any = None) -> Any:
    file_path = Path(path)
    if not file_path.exists():
        return default
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_runtime_metadata() -> dict[str, str]:
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import gymnasium

        metadata["gymnasium"] = gymnasium.__version__
    except Exception:
        metadata["gymnasium"] = "unknown"
    try:
        import mujoco

        metadata["mujoco"] = mujoco.__version__
    except Exception:
        metadata["mujoco"] = "unknown"
    try:
        import torch

        metadata["torch"] = torch.__version__
    except Exception:
        metadata["torch"] = "unknown"
    try:
        import stable_baselines3

        metadata["stable_baselines3"] = stable_baselines3.__version__
    except Exception:
        metadata["stable_baselines3"] = "unknown"
    return metadata


def save_config_snapshot(run_dir: str | Path, config_dict: dict[str, Any]) -> None:
    payload = {
        "config": to_serializable(config_dict),
        "runtime": collect_runtime_metadata(),
    }
    save_json(Path(run_dir) / "config.json", payload)


def load_config_snapshot(run_dir: str | Path) -> dict[str, Any]:
    payload = load_json(Path(run_dir) / "config.json", default={})
    if not payload:
        return {}
    if "config" in payload:
        return payload["config"]
    return payload


def rolling_mean_std(values: Sequence[float], window: int) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    window_values = np.asarray(values[-window:], dtype=float)
    return float(window_values.mean()), float(window_values.std(ddof=0))


def compute_final_distance(observation: Sequence[float]) -> float:
    obs = np.asarray(observation, dtype=float)
    object_position = obs[17:20]
    goal_position = obs[20:23]
    return float(np.linalg.norm(object_position - goal_position))


def compute_objective_metric(observation: Sequence[float]) -> float:
    return -compute_final_distance(observation)


def checkpoint_paths(run_dir: str | Path, tag: str) -> dict[str, Path]:
    directory = Path(run_dir)
    return {
        "model": directory / f"{tag}_model.zip",
        "replay_buffer": directory / f"{tag}_replay_buffer.pkl",
        "policy": directory / f"{tag}_policy.pt",
    }


def video_path(run_dir: str | Path, tag: str) -> Path:
    return Path(run_dir) / f"{tag}_policy.mp4"


def copy_checkpoint_if_exists(source_run_dir: str | Path, target_run_dir: str | Path, tag: str) -> None:
    source_paths = checkpoint_paths(source_run_dir, tag)
    target_paths = checkpoint_paths(target_run_dir, tag)
    for key, source_path in source_paths.items():
        if source_path.exists():
            shutil.copy2(source_path, target_paths[key])


def plot_training_curve(
    history: Sequence[dict[str, Any]],
    value_key: str,
    mean_key: str,
    std_key: str,
    title: str,
    ylabel: str,
    path: str | Path,
    window: int,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))
    if history:
        x_values = [int(item["episode_idx"]) for item in history]
        raw_values = np.asarray([float(item[value_key]) for item in history], dtype=float)
        mean_values = np.asarray([float(item[mean_key]) for item in history], dtype=float)
        std_values = np.asarray([float(item[std_key]) for item in history], dtype=float)
        axis.plot(x_values, raw_values, label=ylabel, linewidth=1.2, alpha=0.5)
        axis.plot(
            x_values,
            mean_values,
            label=f"Rolling mean (window={window})",
            linewidth=2.0,
        )
        axis.fill_between(
            x_values,
            mean_values - std_values,
            mean_values + std_values,
            alpha=0.2,
            label=f"Rolling std (window={window})",
        )
    else:
        axis.text(0.5, 0.5, "No episode data collected", ha="center", va="center", transform=axis.transAxes)
    axis.set_title(title)
    axis.set_xlabel("Episode")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
