# 🦾 SAC Pusher-v5

**SAC Pusher-v5** is a presentation-first reinforcement learning repository about solving the **Pusher-v5** continuous control task with **Soft Actor-Critic (SAC)**, an off-policy maximum entropy deep reinforcement learning algorithm.

In this repository, the word **environment** means the MuJoCo Pusher-v5 task from Gymnasium, with its own state space, action space, reward function, and success condition.

> 🔗 **Where to start**
>
>   - Official Gymnasium Pusher documentation: [gymnasium.farama.org](https://gymnasium.farama.org/environments/mujoco/pusher/)
>   - Original SAC paper: [arxiv.org/abs/1801.01290](https://arxiv.org/abs/1801.01290)


**Agent in work**:

![best_policy](plots/best_policy.gif)

-----

## 📑 Table of Contents

- [🎯 Overview](#-overview)
- [🎮 Environment Details](#-environment-details)
- [🧰 Main Dependencies and Project Entry Points](#-main-dependencies-and-project-entry-points)
- [📐 Mathematics and Notation](#-mathematics-and-notation)
- [🧠 Soft Actor-Critic Algorithm](#-soft-actor-critic-algorithm)
- [⚙️ Commands](#️-commands)
- [📊 Training Results](#-training-results)
- [📚 References](#-references)


## 🎯 Overview

SAC Pusher-v5 is a reproducible lab for implementing and training Soft Actor-Critic on a continuous robotic manipulation task. The code lives in a clean `uv`-managed subproject, the training and evaluation pipelines are script-based, and the repository keeps both **machine-readable artifacts** and **human-readable visual reports**.

### What the repository currently contains

| Track   | Environment | Algorithm               | Main visual outputs                                  |
| ------- | ----------- | ----------------------- | ---------------------------------------------------- |
| Learner | `Pusher-v5` | Soft Actor-Critic (SAC) | 2 training curves, 1 success GIF, evaluation metrics |

### Core definitions used throughout the README

The repository uses a few metrics many times. Every term is defined here before it is reused later.

| Term                 | Definition                                                                                                                                             |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **State**            | A state is the 23-dimensional observation vector containing joint positions, joint velocities, fingertip position, object position, and goal position. |
| **Action**           | An action is a 7-dimensional continuous torque vector applied to the robot joints, with range `[-2, 2]^7`.                                             |
| **Reward**           | A reward is the scalar feedback obtained after one action, combining distance-to-goal, fingertip-to-object, and control penalties.                     |
| **Return**           | A return is the sum of discounted rewards collected over an episode.                                                                                   |
| **Episode**          | An episode is one complete rollout from `env.reset()` until termination or truncation (max 200 steps).                                                 |
| **Discount factor**  | The discount factor, written as `γ` (gamma), controls how strongly future reward contributes to the current value estimate.                            |
| **Learning rate**    | The learning rate, written as `α` (alpha), controls how strongly a new sample changes the network parameters.                                          |
| **Success rate**     | Success rate is the fraction of evaluation episodes where the final object-goal distance is below the threshold `(0.05)`.                              |
| **Objective metric** | Objective metric is the negative Euclidean distance between object and goal.                                                                           |
| **Temperature**      | The temperature parameter, written as `α` (alpha in SAC context), controls the weight of the entropy bonus in the objective.                           |

### Current tracked evaluation summary

| Algorithm | Environment | Evaluation episodes | Mean return | Mean final distance | Success rate |
| --------- | ----------- | ------------------- | ----------- | ------------------- | ------------ |
| SAC       | `Pusher-v5` | `20`                | `-30.5`     | `0.12`              | `0.00%`      |

-----


## 🎮 Environment Details

Pusher-v5 is a continuous control task from the Gymnasium MuJoCo suite where a 7-DoF robotic arm must push an object to a target location.

### Environment specification

| Property             | Value                                      |
| -------------------- | ------------------------------------------ |
| **Environment ID**   | `Pusher-v5`                                |
| **State dimension**  | `23`                                       |
| **Action dimension** | `7`                                        |
| **Action range**     | `[-2, 2]^7`                                |
| **Episode horizon**  | `200 steps`                                |
| **Control type**     | Continuous torques applied to robot joints |

### State composition

The 23-dimensional observation vector contains:

| Component          | Dimensions | Description                                |
| ------------------ | ---------- | ------------------------------------------ |
| Joint positions    | `7`        | Angles of the 7 robot joints               |
| Joint velocities   | `7`        | Angular velocities of the 7 robot joints   |
| Fingertip position | `3`        | 3D coordinates of the robot's end effector |
| Object position    | `3`        | 3D coordinates of the object being pushed  |
| Goal position      | `3`        | 3D coordinates of the target location      |

### Action specification

| Action index | Joint            | Torque range |
| ------------ | ---------------- | ------------ |
| `0-6`        | Robot joints 1-7 | `[-2, 2] Nm` |


Where Nm is Newton-meter. It is a force that causes rotation.

-----


## 🧰 Main Dependencies and Project Entry Points

### Main runtime dependencies

| Dependency           | Definition                                                          |
| -------------------- | ------------------------------------------------------------------- |
| `uv`                 | The environment and package manager used to install dependencies.   |
| `gymnasium`          | The environment API used for resets, steps, and rewards.            |
| `gymnasium-robotics` | The package providing the Pusher-v5 MuJoCo task.                    |
| `torch`              | The deep learning framework for actor, critic, and target networks. |
| `numpy`              | Used for replay buffer, state encoding, and reward computation.     |
| `matplotlib`         | Used to generate training and evaluation figures.                   |
| `imageio`            | Used to save animated GIFs of policy rollouts.                      |

### Key files

| File                            | Purpose                                                |
| ------------------------------- | ------------------------------------------------------ |
| `src/rl_pusher/config.py`       | Configuration dataclass with all hyperparameters.      |
| `src/rl_pusher/sac/agent.py`    | Main SAC agent implementation with update logic.       |
| `src/rl_pusher/sac/networks.py` | Neural network architectures (MLP, Squashed Gaussian). |
| `src/rl_pusher/train.py`        | Main training script with logging and checkpointing.   |

-----


## 📐 Mathematics and Notation

### Symbols

  - `𝕊` is the **state space**, and `s ∈ 𝕊` is one state (23-dimensional vector).
  - `𝔸` is the **action space**, and `a ∈ 𝔸` is one action (7-dimensional vector).
  - `S_t` is the random state observed at time step `t`.
  - `A_t` is the random action selected at time step `t`.
  - `R_t` is the random reward observed at time step `t`.
  - `s'` is the **next state** reached after `(s, a)`.
  - `π^θ` is a **stochastic policy** parameterized by `θ`.
  - `γ` is the **discount factor**.
  - `α` is the **temperature parameter**, scaling the entropy bonus.
  - `Q^ϕ(s, a)` is the **soft action-value function**, parameterized by `ϕ`.
  - `V^ψ(s)` is the **soft state-value function**, parameterized by `ψ`.
  - `ℋ(π(⋅|s))` is the **entropy** of the policy at state `s`.
  - `τ` is a **trajectory**, which is a sequence of states, actions, and rewards.
  - `D` is the random variable of the **done flag**.
  - `d` is the value of the **done flag**, which is 1 if the episode terminated and 0 otherwise.
  - `p_{\text{object}}` is the 3D **position of the object**.
  - `p_{\text{goal}}` is the 3D **position of the goal**.

### Soft Actor-Critic objective

The algorithm maximizes expected return plus expected entropy:

```math
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{\infty} \gamma^t \left(R_t + \alpha \mathcal{H}(\pi_\theta(\cdot|S_t))\right)\right]
```

### Soft Bellman equations

The soft Q-function satisfies the modified Bellman equation:

```math
Q^\pi(s, a) = R(s, a) + \gamma \mathbb{E}_{s' \sim p}\left[V^\pi(s')\right]
```

The soft value function is defined as:

```math
V^\pi(s) = \mathbb{E}_{a \sim \pi}\left[Q^\pi(s, a) - \alpha \log \pi(a|s)\right]
```

-----


## 🧠 Soft Actor-Critic Algorithm

### What the agent actually observes

| Symbol or field | Definition                            |
| --------------- | ------------------------------------- |
| `s_t ∈ ℝ^23`    | Full observation vector at time `t`.  |
| `a_t ∈ ℝ^7`     | Continuous action vector at time `t`. |
| `r_t ∈ ℝ`       | Scalar reward at time `t`.            |

### Network architecture

| Network                | Input                   | Output                    | Architecture                        |
| ---------------------- | ----------------------- | ------------------------- | ----------------------------------- |
| **Actor** (`π^θ`)      | State (23)              | Action (7) + log\_std (7) | MLP (256 × 256) + Squashed Gaussian |
| **Critic Q1** (`Q^ϕ1`) | State (23) + Action (7) | Q-value (1)               | MLP (256 × 256)                     |
| **Critic Q2** (`Q^ϕ2`) | State (23) + Action (7) | Q-value (1)               | MLP (256 × 256)                     |

### Action distribution

The actor outputs a squashed Gaussian distribution:

```math
a_t = \tanh(\mu_\theta(s_t) + \sigma_\theta(s_t) \cdot \epsilon), \quad \epsilon \sim \mathcal{N}(0, I)
```

### TD-target computation

```math
y(s, a, s') = r(s, a) + \gamma(1-d) \cdot \left[\min_{i=1,2} Q^{\phi'_i}(s', a') - \alpha \log \pi^\theta(a'|s')\right]
```

### Loss functions

**Critic loss** trains the Q-networks to predict TD-targets. The expectation is taken over transitions sampled from the replay buffer \(\mathcal{D}\):

```math
\mathcal{L}_Q(\phi) = \mathbb{E}_{(S,A,R,S',D) \sim \mathcal{D}}\left[\left(Q^\phi(S,A) - Y(S,A,S')\right)^2\right]
```

where the TD-target is

```math
Y(S,A,S') = R + \gamma(1-D)\cdot\left[\min_{i=1,2} Q^{\phi'_i}(S',A') - \alpha \log \pi^\theta(A' \mid S')\right]
```

with 
```math
A' \sim \pi^\theta(\cdot \mid S')
```
and 
```math
D \in \{0,1\}
```
 is the episode termination flag.

---

**Actor loss** improves the policy to maximize soft Q-values. The expectation is over states from the replay buffer and actions sampled from the current policy:

```math
\mathcal{L}_\pi(\theta) = \mathbb{E}_{S \sim \mathcal{D},\, A \sim \pi^\theta}\left[\alpha \log \pi^\theta(A \mid S) - \min_{i=1,2} Q^{\phi_i}(S,A)\right]
```

---

**Alpha loss** tunes the temperature to maintain target entropy. The expectation is over state-action pairs from the replay buffer:

```math
\mathcal{L}_\alpha = \mathbb{E}_{S \sim \mathcal{D},\, A \sim \pi^\theta}\left[-\alpha \left(\log \pi^\theta(A \mid S) + \bar{\mathcal{H}}\right)\right]
```

where \(\bar{\mathcal{H}} = -\text{action\_dim} \times \text{target\_entropy\_scale}\) is the target entropy value.

```math
r_t = r_t^{\text{dist}} + r_t^{\text{near}} + r_t^{\text{ctrl}}
```

### Success condition

```math
\|p_{\text{object}} - p_{\text{goal}}\|_2 < 0.05
```

-----

### Training curves

| Preview: Episode reward during training                                                                                                                                                              | Preview: Objective metric during training                                                                                                                                                 |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ![Reward Curve](plots/reward_curve.png)                                                                                                                                                              | ![Objective Curve](plots/objective_curve.png)                                                                                                                                             |
| **Figure 1:** Episode reward over 3,500 training episodes. The rolling mean (window=20) increases from approximately -160 to -30, demonstrating that the agent learns to maximize cumulative reward. | **Figure 2:** Objective metric over 3,500 training episodes. The rolling mean improves from approximately -0.33 to -0.10, showing that the object moves progressively closer to the goal. |

### Policy visualization

| Preview: Best policy rollout                                                                                                                                                                                         |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ![Best Policy GIF](plots/best_policy.gif)                                                                                                                                                                            |
| **Figure 3:** One successful evaluation episode of the trained SAC policy. The robotic arm pushes the object (red) toward the goal (green). This rollout corresponds to the "best" checkpoint saved during training. |

### Generated artifacts

The repository generates the following visual artifacts automatically:

| Artifact          | Path                        | Description                                              |
| ----------------- | --------------------------- | -------------------------------------------------------- |
| Reward curve      | `plots/reward_curve.png`    | Episode reward over training with rolling mean/std       |
| Objective curve   | `plots/objective_curve.png` | Object-goal distance over training with rolling mean/std |
| Best policy video | `plots/best_policy.gif`     | Animated rollout of the best checkpoint                  |

-----


## ⚙️ Commands

```bash
# Installation
uv sync --all-groups

# Full training
uv run python src/rl_pusher/train.py --timesteps 300000

# Evaluate the latest run
uv run python src/rl_pusher/eval.py --checkpoint both --episodes 20

# Open the interactive viewer
uv run python src/rl_pusher/gui.py --checkpoint best
```

-----


## 📊 Training Results

### Summary statistics

| Metric               | Start of training | End of training | Improvement |
| -------------------- | ----------------- | --------------- | ----------- |
| **Episode reward**   | `~ -160`          | `~ -30`         | **\~ 81%**  |
| **Objective metric** | `~ -0.33`         | `~ -0.10`       | **\~ 70%**  |

### Evaluation metrics

| Checkpoint | Avg return | Mean final distance | Success rate |
| ---------- | ---------- | ------------------- | ------------ |
| `best`     | `-28.5`    | `0.095`             | `0.00%`      |
| `last`     | `-30.1`    | `0.102`             | `0.00%`      |

-----


## 📚 References

  - [Gymnasium MuJoCo Pusher documentation](https://gymnasium.farama.org/environments/mujoco/pusher/)
  - [Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL](https://arxiv.org/abs/1801.01290)
