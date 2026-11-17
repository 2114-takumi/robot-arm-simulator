import os

import gymnasium as gym
from gymnasium import spaces

import mujoco
import numpy as np


class RobotArmRLEnv(gym.Env):

    metadata = {
        "render_modes": []
    }

    def __init__(self):

        super().__init__()

        # ==========================================
        # MuJoCo model
        # ==========================================

        model_path = os.path.join(
            os.path.dirname(__file__),
            "models",
            "two_link_arm.xml",
        )

        self.model = mujoco.MjModel.from_xml_path(
            model_path
        )

        self.data = mujoco.MjData(
            self.model
        )

        # MuJoCo: 500 Hz
        # RL policy: 50 Hz
        self.physics_steps_per_action = 10

        # 200 RL steps × 0.02 s = 4 s
        self.max_steps = 200

        self.step_count = 0

        # ==========================================
        # Goal
        # ==========================================

        self.q_goal = np.zeros(
            2,
            dtype=np.float32,
        )

        # ==========================================
        # Observation
        #
        # [q1, q2, dq1, dq2, q1_goal, q2_goal]
        # ==========================================

        obs_high = np.array(
            [
                np.pi,
                np.pi,
                20.0,
                20.0,
                np.pi,
                np.pi,
            ],
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-obs_high,
            high=obs_high,
            dtype=np.float32,
        )

        # ==========================================
        # Action
        #
        # PPO outputs normalized action:
        # [-1, 1]
        #
        # Environment converts this to torque:
        # joint1 -> [-30, 30] Nm
        # joint2 -> [-15, 15] Nm
        # ==========================================

        self.torque_limit = np.array(
            [30.0, 15.0],
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

    # ==========================================
    # Observation
    # ==========================================

    def _get_obs(self):

        q = self.data.qpos[:2]
        dq = self.data.qvel[:2]

        obs = np.concatenate(
            [
                q,
                dq,
                self.q_goal,
            ]
        )

        return obs.astype(
            np.float32
        )

    # ==========================================
    # Reset
    # ==========================================

    def reset(
        self,
        seed=None,
        options=None,
    ):

        super().reset(
            seed=seed
        )

        mujoco.mj_resetData(
            self.model,
            self.data,
        )

        # Random initial joint angles
        q_start = self.np_random.uniform(
            low=np.array(
                [-0.5, -0.7]
            ),
            high=np.array(
                [0.5, 0.7]
            ),
        )

        # Small random initial velocity
        dq_start = self.np_random.uniform(
            low=-0.1,
            high=0.1,
            size=2,
        )

        # Random goal
        self.q_goal = self.np_random.uniform(
            low=np.array(
                [-0.8, -0.8]
            ),
            high=np.array(
                [0.8, 0.8]
            ),
        ).astype(
            np.float32
        )

        self.data.qpos[:2] = q_start
        self.data.qvel[:2] = dq_start

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        self.step_count = 0

        observation = self._get_obs()

        info = {
            "q_goal": self.q_goal.copy()
        }

        return observation, info

    # ==========================================
    # Step
    # ==========================================

    def step(
        self,
        action,
    ):

        self.step_count += 1

        # ------------------------------------------
        # PPO action [-1, 1]
        # ------------------------------------------

        action = np.clip(
            action,
            self.action_space.low,
            self.action_space.high,
        )

        # ------------------------------------------
        # Convert normalized action to torque
        # ------------------------------------------

        tau = (
            action
            * self.torque_limit
        )

        # ------------------------------------------
        # Apply torque for 10 MuJoCo physics steps
        # ------------------------------------------

        for _ in range(
            self.physics_steps_per_action
        ):

            self.data.qfrc_applied[:2] = tau

            mujoco.mj_step(
                self.model,
                self.data,
            )

        # ------------------------------------------
        # Current state
        # ------------------------------------------

        q = self.data.qpos[:2].copy()
        dq = self.data.qvel[:2].copy()

        error = (
            self.q_goal - q
        )

        # ==========================================
        # Reward
        # ==========================================

        position_cost = np.sum(
            error ** 2
        )

        velocity_cost = (
            0.02
            * np.sum(
                dq ** 2
            )
        )

        torque_cost = (
            0.0005
            * np.sum(
                tau ** 2
            )
        )

        reward = -(
            position_cost
            + velocity_cost
            + torque_cost
        )

        # ------------------------------------------
        # Success condition
        # ------------------------------------------

        position_error = np.linalg.norm(
            error
        )

        velocity_norm = np.linalg.norm(
            dq
        )

        success = (
            position_error < 0.05
            and velocity_norm < 0.15
        )

        if success:
            reward += 10.0

        # Do not terminate immediately when successful.
        # This encourages the policy to remain stable
        # around the goal.
        terminated = False

        truncated = (
            self.step_count
            >= self.max_steps
        )

        observation = self._get_obs()

        info = {
            "position_error": float(
                position_error
            ),
            "velocity_norm": float(
                velocity_norm
            ),
            "success": bool(
                success
            ),
            "q": q.copy(),
            "q_goal": self.q_goal.copy(),
            "tau": tau.copy(),
        }

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info,
        )


# ==========================================
# Simple environment test
# ==========================================

if __name__ == "__main__":

    env = RobotArmRLEnv()

    obs, info = env.reset(
        seed=0
    )

    print(
        "Initial observation:",
        obs,
    )

    print(
        "Goal:",
        info["q_goal"],
    )

    total_reward = 0.0

    for step in range(
        env.max_steps
    ):

        # Random normalized action [-1, 1]
        action = env.action_space.sample()

        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            action
        )

        total_reward += reward

        if (
            terminated
            or truncated
        ):
            break

    print(
        "Final q:",
        info["q"],
    )

    print(
        "Goal:",
        info["q_goal"],
    )

    print(
        "Position error:",
        info["position_error"],
    )

    print(
        "Total reward:",
        total_reward,
    )
