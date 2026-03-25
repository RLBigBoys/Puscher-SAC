from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn
from torch.distributions import Normal


LOG_PROB_EPS = 1e-6


def build_mlp(
    input_dim: int,
    hidden_sizes: Sequence[int],
    output_dim: int,
    activation: type[nn.Module] = nn.ReLU,
    output_activation: type[nn.Module] | None = None,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = input_dim
    for hidden_dim in hidden_sizes:
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(activation())
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: Sequence[int]):
        super().__init__()
        self.network = build_mlp(obs_dim + action_dim, hidden_sizes, 1)

    def forward(self, obs: Tensor, action: Tensor) -> Tensor:
        return self.network(torch.cat([obs, action], dim=-1))


class SquashedGaussianActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int],
        action_low: Tensor,
        action_high: Tensor,
        log_std_min: float,
        log_std_max: float,
    ):
        super().__init__()
        last_hidden_size = hidden_sizes[-1] if hidden_sizes else obs_dim
        self.backbone = build_mlp(obs_dim, hidden_sizes, last_hidden_size)
        self.mean_head = nn.Linear(last_hidden_size, action_dim)
        self.log_std_head = nn.Linear(last_hidden_size, action_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.register_buffer("action_scale", (action_high - action_low) / 2.0)
        self.register_buffer("action_bias", (action_high + action_low) / 2.0)

    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.backbone(obs)
        mean = self.mean_head(hidden)
        log_std = self.log_std_head(hidden)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, obs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        distribution = Normal(mean, std)
        raw_action = distribution.rsample()
        squashed_action = torch.tanh(raw_action)
        action = squashed_action * self.action_scale + self.action_bias

        log_prob = distribution.log_prob(raw_action)
        correction = torch.log(self.action_scale.abs() * (1.0 - squashed_action.pow(2)) + LOG_PROB_EPS)
        log_prob = (log_prob - correction).sum(dim=-1, keepdim=True)

        deterministic_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, deterministic_action

    def act(self, obs: Tensor, deterministic: bool) -> Tensor:
        if deterministic:
            mean, _ = self.forward(obs)
            return torch.tanh(mean) * self.action_scale + self.action_bias
        action, _, _ = self.sample(obs)
        return action
