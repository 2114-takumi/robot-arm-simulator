import numpy as np

from stable_baselines3 import PPO

from robot_arm_rl_env import RobotArmRLEnv


def main():

    # ==========================================
    # Load trained policy
    # ==========================================

    model = PPO.load(
        "models_rl/ppo_robot_arm_500k"
    )

    env = RobotArmRLEnv()

    num_episodes = 100

    successes = 0

    final_errors = []
    minimum_errors = []

    # ==========================================
    # Evaluation
    # ==========================================

    for episode in range(
        num_episodes
    ):

        obs, info = env.reset(
            seed=1000 + episode
        )

        q_goal = info[
            "q_goal"
        ].copy()

        ever_success = False

        min_error = np.inf

        for step in range(
            env.max_steps
        ):

            action, _ = model.predict(
                obs,
                deterministic=True,
            )

            (
                obs,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                action
            )

            min_error = min(
                min_error,
                info[
                    "position_error"
                ],
            )

            if info["success"]:
                ever_success = True

            if (
                terminated
                or truncated
            ):
                break

        if ever_success:
            successes += 1

        final_error = info[
            "position_error"
        ]

        final_errors.append(
            final_error
        )

        minimum_errors.append(
            min_error
        )

        print(
            f"Episode {episode + 1:02d}: "
            f"goal={q_goal}, "
            f"final_q={info['q']}, "
            f"final_error={final_error:.4f}, "
            f"min_error={min_error:.4f}, "
            f"success={ever_success}"
        )

    # ==========================================
    # Summary
    # ==========================================

    success_rate = (
        100.0
        * successes
        / num_episodes
    )

    print()
    print(
        "=============================="
    )
    print(
        "PPO evaluation"
    )
    print(
        "=============================="
    )

    print(
        f"Success rate: "
        f"{success_rate:.1f}%"
    )

    print(
        f"Mean final error: "
        f"{np.mean(final_errors):.4f} rad"
    )

    print(
        f"Median final error: "
        f"{np.median(final_errors):.4f} rad"
    )

    print(
        f"Mean minimum error: "
        f"{np.mean(minimum_errors):.4f} rad"
    )

    env.close()


if __name__ == "__main__":
    main()

