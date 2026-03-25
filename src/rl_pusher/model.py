from __future__ import annotations

import torch.nn as nn

try:
    from .config import Config
except ImportError:
    from config import Config


def get_policy_kwargs(cfg: Config) -> dict[str, object]:
    hidden_layers = list(cfg.hidden_sizes)
    return {
        "activation_fn": nn.ReLU,
        "net_arch": {"pi": hidden_layers, "qf": hidden_layers},
    }


def describe_policy(cfg: Config) -> str:
    hidden_layers = " x ".join(str(size) for size in cfg.hidden_sizes)
    return f"Actor/Critic MLP ({hidden_layers})"
