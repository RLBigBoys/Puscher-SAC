from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


class ReplayBuffer:
    def __init__(self, obs_dim: int, action_dim: int, capacity: int):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.capacity = int(capacity)
        self.observations = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_observations = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        self.observations[self.ptr] = np.asarray(observation, dtype=np.float32).reshape(-1)
        self.actions[self.ptr] = np.asarray(action, dtype=np.float32).reshape(-1)
        self.rewards[self.ptr] = float(reward)
        self.next_observations[self.ptr] = np.asarray(next_observation, dtype=np.float32).reshape(-1)
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_observations": self.next_observations[indices],
            "dones": self.dones[indices],
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "capacity": self.capacity,
            "ptr": self.ptr,
            "size": self.size,
            "observations": self.observations[: self.size].copy(),
            "next_observations": self.next_observations[: self.size].copy(),
            "actions": self.actions[: self.size].copy(),
            "rewards": self.rewards[: self.size].copy(),
            "dones": self.dones[: self.size].copy(),
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        self.obs_dim = int(state_dict["obs_dim"])
        self.action_dim = int(state_dict["action_dim"])
        self.capacity = int(state_dict["capacity"])
        self.observations = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.next_observations = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.ptr = int(state_dict["ptr"])
        self.size = int(state_dict["size"])
        self.observations[: self.size] = np.asarray(state_dict["observations"], dtype=np.float32)
        self.next_observations[: self.size] = np.asarray(state_dict["next_observations"], dtype=np.float32)
        self.actions[: self.size] = np.asarray(state_dict["actions"], dtype=np.float32)
        self.rewards[: self.size] = np.asarray(state_dict["rewards"], dtype=np.float32)
        self.dones[: self.size] = np.asarray(state_dict["dones"], dtype=np.float32)

    def save(self, path: str | Path) -> None:
        file_path = Path(path)
        with file_path.open("wb") as handle:
            pickle.dump(self.state_dict(), handle)

    def load(self, path: str | Path) -> None:
        file_path = Path(path)
        with file_path.open("rb") as handle:
            state_dict = pickle.load(handle)
        self.load_state_dict(state_dict)
