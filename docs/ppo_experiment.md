# PPO Reinforcement Learning Experiment

## Objective

Train a neural-network policy to move a two-link MuJoCo robot arm from randomized initial joint states to randomized target joint angles.

Unlike behavioral cloning, the policy is not given expert torque labels. It learns directly from rewards obtained through interaction with MuJoCo.

## Environment

### Observation

The policy receives the current joint state and the target joint angles:

```text
[q1, q2, dq1, dq2, q1_goal, q2_goal]
```

- `q1`, `q2`: current joint angles
- `dq1`, `dq2`: current joint velocities
- `q1_goal`, `q2_goal`: target joint angles

### Action

The PPO policy outputs normalized continuous actions:

```text
[action1, action2] in [-1, 1]
```

The environment converts the normalized actions into joint torques:

```text
tau1 = action1 * 30 Nm
tau2 = action2 * 15 Nm
```

Normalizing the action space made PPO training more stable than directly using the physical torque ranges as the action space.

### Control Frequency

```text
MuJoCo physics frequency: 500 Hz
PPO action update frequency: 50 Hz
Physics steps per action: 10
Episode duration: 4 s
Maximum RL steps per episode: 200
```

### Initial State and Goal

At the beginning of each episode:

- Initial joint angles are randomized.
- Initial joint velocities are slightly randomized.
- Target joint angles are randomized.

This allows the policy to learn goal-conditioned control rather than memorizing one fixed target.

### Reward

The reward penalizes joint-position error, joint velocity, and torque use:

```text
reward =
    - position error squared
    - velocity penalty
    - torque penalty
    + success bonus
```

Conceptually:

```text
reward =
    - ||q_goal - q||^2
    - 0.02 * ||dq||^2
    - 0.0005 * ||tau||^2
    + success bonus
```

A success bonus is added when:

```text
joint-position error norm < 0.05 rad
and
joint-velocity norm < 0.15 rad/s
```

The current evaluation counts an episode as successful if this condition is satisfied at least once during the episode.

## PPO Training

The policy was trained using Stable-Baselines3 PPO with an MLP policy.

Main settings:

```text
Algorithm: PPO
Policy: MlpPolicy
Learning rate: 3e-4
Rollout length: 2048 steps
Batch size: 64
Epochs per update: 10
Discount factor gamma: 0.99
GAE lambda: 0.95
PPO clip range: 0.2
Device: CPU
```

Training was performed in stages:

1. Initial training to 100k environment steps
2. Continued training to 300k-equivalent steps
3. Fine-tuning to 500k-equivalent steps with a lower learning rate

## Training Progression

| Training stage | Success rate | Mean final error |
|---|---:|---:|
| 100k steps | 65% | 0.0755 rad |
| 300k-equivalent steps | 70% | 0.0525 rad |
| 500k-equivalent steps | 95% | 0.0380 rad |

The final policy was evaluated over 100 deterministic episodes.

## Final Evaluation

```text
Episodes:            100
Success rate:        95.0%
Mean final error:    0.0380 rad
Median final error:  0.0352 rad
Mean minimum error:  0.0289 rad
```

The mean final error of `0.0380 rad` corresponds to approximately `2.18 degrees`.

## Main Findings

1. Directly using large physical torque ranges as the PPO action space made optimization difficult.

2. Normalizing the policy action to `[-1, 1]` and scaling it into physical torque inside the environment substantially improved learning.

3. Continued training improved both task success and final positioning accuracy.

4. Fine-tuning the 300k policy with a lower learning rate improved the success rate from 70% to 95%.

5. The trained policy learned to control the robot without using expert torque labels from CTC.

6. Remaining failures were concentrated mainly around relatively large positive joint-angle targets.

## Comparison with Behavioral Cloning

Behavioral cloning learns by imitating expert actions:

```text
state + goal
    ↓
neural-network policy
    ↓
predicted torque

expert CTC torque
    ↓
supervised loss
```

PPO does not receive expert torque labels:

```text
state + goal
    ↓
PPO policy
    ↓
torque
    ↓
MuJoCo
    ↓
reward
    ↓
policy update
```

This experiment therefore demonstrates the difference between supervised imitation learning and reward-based reinforcement learning on the same robot-arm model.

## Limitations

- Success currently means that the goal condition was met at least once during the episode.
- Stable goal holding should be evaluated separately.
- The policy uses joint-state observations rather than camera images.
- The policy directly outputs torque commands.
- Robustness to model uncertainty, disturbances, and sensor noise has not yet been evaluated.
- Sim-to-real transfer has not yet been evaluated.
