from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    from ..config import Config
    from .networks import QNetwork, SquashedGaussianActor
except ImportError:
    from config import Config
    from custom_sac.networks import QNetwork, SquashedGaussianActor


CHECKPOINT_ALGORITHM = "custom_sac_pytorch"


class CustomSACAgent:
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, cfg: Config):
        if not isinstance(observation_space, gym.spaces.Box):
            raise TypeError("Custom SAC expects a continuous Box observation space.")
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError("Custom SAC expects a continuous Box action space.")

        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.obs_dim = int(np.prod(observation_space.shape))
        self.action_dim = int(np.prod(action_space.shape))
        action_low = torch.as_tensor(action_space.low, dtype=torch.float32, device=self.device).view(-1)
        action_high = torch.as_tensor(action_space.high, dtype=torch.float32, device=self.device).view(-1)

        self.actor = SquashedGaussianActor(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_sizes=cfg.hidden_sizes,
            action_low=action_low,
            action_high=action_high,
            log_std_min=cfg.log_std_min,
            log_std_max=cfg.log_std_max,
        ).to(self.device)
        self.q1 = QNetwork(self.obs_dim, self.action_dim, cfg.hidden_sizes).to(self.device)
        self.q2 = QNetwork(self.obs_dim, self.action_dim, cfg.hidden_sizes).to(self.device)
        self.q1_target = QNetwork(self.obs_dim, self.action_dim, cfg.hidden_sizes).to(self.device)
        self.q2_target = QNetwork(self.obs_dim, self.action_dim, cfg.hidden_sizes).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=cfg.critic_lr)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=cfg.critic_lr)

        initial_log_alpha = float(np.log(max(cfg.alpha_init, 1e-6)))
        self.log_alpha = nn.Parameter(torch.tensor([initial_log_alpha], dtype=torch.float32, device=self.device))
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
        self.target_entropy = -float(self.action_dim) * float(cfg.target_entropy_scale)

    @property
    def alpha(self) -> Tensor:
        return self.log_alpha.exp()

    def select_action(self, observation: np.ndarray, deterministic: bool) -> np.ndarray:
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).reshape(1, -1)
        with torch.no_grad():
            action_tensor = self.actor.act(observation_tensor, deterministic=deterministic)
        return action_tensor.squeeze(0).cpu().numpy()

    def update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        observations = torch.as_tensor(batch["observations"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_observations = torch.as_tensor(batch["next_observations"], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_actions, next_log_prob, _ = self.actor.sample(next_observations)
            target_q1 = self.q1_target(next_observations, next_actions)
            target_q2 = self.q2_target(next_observations, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha.detach() * next_log_prob
            td_target = self.cfg.reward_scale * rewards + self.cfg.gamma * (1.0 - dones) * target_q

        current_q1 = self.q1(observations, actions)
        current_q2 = self.q2(observations, actions)
        q1_loss = F.mse_loss(current_q1, td_target)
        q2_loss = F.mse_loss(current_q2, td_target)

        self.q1_optimizer.zero_grad(set_to_none=True)
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad(set_to_none=True)
        q2_loss.backward()
        self.q2_optimizer.step()

        sampled_actions, log_prob, _ = self.actor.sample(observations)
        q1_pi = self.q1(observations, sampled_actions)
        q2_pi = self.q2(observations, sampled_actions)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha.detach() * log_prob - min_q_pi).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.soft_update_targets(self.cfg.tau)

        return {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(0.5 * (q1_loss.item() + q2_loss.item())),
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "alpha": float(self.alpha.item()),
            "alpha_loss": float(alpha_loss.item()),
        }

    def soft_update_targets(self, tau: float) -> None:
        with torch.no_grad():
            for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters(), strict=True):
                target_param.data.mul_(1.0 - tau).add_(tau * param.data)
            for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters(), strict=True):
                target_param.data.mul_(1.0 - tau).add_(tau * param.data)

    def checkpoint_state(
        self,
        global_step: int,
        episode_idx: int,
        update_step: int,
        best_reward: float,
    ) -> dict[str, Any]:
        return {
            "algorithm": CHECKPOINT_ALGORITHM,
            "global_step": int(global_step),
            "episode_idx": int(episode_idx),
            "update_step": int(update_step),
            "best_reward": float(best_reward),
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "q1_optimizer": self.q1_optimizer.state_dict(),
            "q2_optimizer": self.q2_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
        }

    def save_checkpoint(
        self,
        path: str | Path,
        global_step: int,
        episode_idx: int,
        update_step: int,
        best_reward: float,
    ) -> None:
        torch.save(self.checkpoint_state(global_step, episode_idx, update_step, best_reward), Path(path))

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        checkpoint = torch.load(Path(path), map_location=self.device)
        if checkpoint.get("algorithm") != CHECKPOINT_ALGORITHM:
            raise ValueError(f"{path} is not a custom SAC checkpoint.")

        self.actor.load_state_dict(checkpoint["actor"])
        self.q1.load_state_dict(checkpoint["q1"])
        self.q2.load_state_dict(checkpoint["q2"])
        self.q1_target.load_state_dict(checkpoint["q1_target"])
        self.q2_target.load_state_dict(checkpoint["q2_target"])
        self.log_alpha.data.copy_(checkpoint["log_alpha"].to(self.device))
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.q1_optimizer.load_state_dict(checkpoint["q1_optimizer"])
        self.q2_optimizer.load_state_dict(checkpoint["q2_optimizer"])
        self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
        return {
            "global_step": int(checkpoint.get("global_step", 0)),
            "episode_idx": int(checkpoint.get("episode_idx", 0)),
            "update_step": int(checkpoint.get("update_step", 0)),
            "best_reward": float(checkpoint.get("best_reward", float("-inf"))),
        }

    def save_policy(self, path: str | Path) -> None:
        torch.save(self.actor.state_dict(), Path(path))
