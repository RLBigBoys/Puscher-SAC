# Custom PyTorch SAC on Pusher-v5

## Overview

This project implements a custom PyTorch version of Soft Actor-Critic (SAC) for `Pusher-v5`. The agent controls a 7-DoF robotic arm and must push an object to a target location. Success is defined as a final object-goal distance below a configurable threshold.

## Environment

- State dimension: `23`
- Action dimension: `7`
- Action range: `[-2, 2]^7`
- Episode length: `100` steps
- Control type: continuous torques applied to robot joints

The observation contains joint positions, joint velocities, fingertip position, object position and goal position.

## Mathematics and Notation

We follow the standard notation

\[
S_t \in \mathcal{S}, \quad A_t \in \mathcal{A}, \quad R_t \in \mathbb{R}
\]

\[
a_t \sim \pi^\theta(\cdot \mid s_t)
\]

\[
J(\theta) = \mathbb{E}_{\tau \sim \zeta^\theta}\left[\sum_{t=0}^{\tau-1}\gamma^t R_t\right]
\]

\[
v^\pi(s) = \mathbb{E}_{\pi}\left[\sum_{t=0}^{\infty}\gamma^t R_t \mid S_0=s\right]
\]

Here `S_t, A_t, R_t` denote random variables, while `s_t, a_t, r_t` denote concrete tensor values observed at time step `t`. The policy is parameterized by `\theta`, and SAC optimizes an entropy-regularized actor-critic objective.

## Reward Function

For `Pusher-v5`, the reward is

\[
r_t = r_t^{dist} + r_t^{near} + r_t^{ctrl}
\]

with the following interpretation.

| Component | Meaning |
| --- | --- |
| `reward_dist` | Encourages the object to move toward the goal |
| `reward_near` | Encourages the fingertip to approach the object |
| `reward_ctrl` | Penalizes large torque commands |

The project also tracks an objective metric

\[
\text{objective} = - \lVert p_{object} - p_{goal} \rVert_2
\]

where `p_object` and `p_goal` are extracted from the final observation of the episode.

## Algorithm

SAC is an off-policy actor-critic algorithm. The actor learns a stochastic policy, the critics learn soft Q-functions, and a replay buffer stores past transitions for sample-efficient updates. In this repository, the full training pipeline, replay buffer, actor, and critics are implemented in custom PyTorch code.

## Repository Structure

```text
Pusher-SAC/
├── README.md
├── .gitignore
├── pyproject.toml
├── artifacts/
├── src/
│   └── rl_pusher/
│       ├── __init__.py
│       ├── config.py
│       ├── custom_sac/
│       ├── utils.py
│       ├── train.py
│       ├── eval.py
│       └── gui.py
└── tests/
```

## Configuration

The main settings live in `src/rl_pusher/config.py`.

- Environment id, random seed and number of timesteps
- SAC hyperparameters: learning rate, `\gamma`, replay buffer size, batch size, `\tau`, training frequency, gradient steps
- Network architecture via `hidden_sizes`
- Resume mode: `none`, `best`, `last`
- Evaluation and success-threshold settings
- Video recording settings
- Rolling-window size for mean and standard deviation on plots
- Artifact root and run name

## Training

Training creates a unique run directory inside `artifacts/runs/`. Each run stores:

- `config.json`
- `train_history.json`
- `best_model.zip`, `last_model.zip`
- `best_replay_buffer.pkl`, `last_replay_buffer.pkl`
- `best_policy.pt`, `last_policy.pt`
- `reward_curve.png`, `objective_curve.png`
- `eval.json`
- `best_policy.mp4`, `last_policy.mp4` when video recording is enabled

The training pipeline:

1. Create the environment, policy, critics, and replay buffer.
2. Optionally resume from a previous `best` or `last` checkpoint.
3. Log episode reward, objective metric, rolling mean and rolling standard deviation to JSON.
4. Save `best` and `last` checkpoints.
5. Handle `Ctrl+C` safely and still save artifacts.
6. Run evaluation automatically after training.

## Evaluation

Evaluation reports:

- average return
- return standard deviation
- mean final distance
- final-distance standard deviation
- success rate

Success is defined as

\[
\lVert p_{object} - p_{goal} \rVert_2 < \text{success\_threshold}
\]

## Results

After training, place the numerical results from `eval.json` here and reference the generated figures. A typical report should discuss whether the reward curve trends upward, whether the final distance decreases, and whether the success rate becomes non-zero.

## Artifact Gallery

The repository generates the following visual artifacts automatically:

- `reward_curve.png`
- `objective_curve.png`
- `best_policy.mp4`
- `last_policy.mp4`

Use scientific figure captions in the final report, for example:

`Figure 1: Episode reward during SAC training on Pusher-v5. The rolling mean over 20 episodes increases from ... to ...`

## Commands

Install dependencies:

```bash
uv sync
```

Quick smoke-train:

```bash
uv run python src/rl_pusher/train.py --timesteps 2000 --eval-episodes 3 --no-video
```

Full training:

```bash
uv run python src/rl_pusher/train.py --timesteps 300000 --run-name custom_sac_pusher
```

Resume from the latest run:

```bash
uv run python src/rl_pusher/train.py --load-mode last --timesteps 100000
```

Evaluate the latest run:

```bash
uv run python src/rl_pusher/eval.py --checkpoint both --episodes 20
```

Open the viewer:

```bash
uv run python src/rl_pusher/gui.py --checkpoint best
```

## References

- Gymnasium MuJoCo Pusher documentation: https://gymnasium.farama.org/environments/mujoco/pusher/
