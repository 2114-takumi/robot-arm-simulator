from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from robot_arm_rl_env import RobotArmRLEnv


def main():

    # ==========================================
    # 1. Create environment
    # ==========================================

    env = RobotArmRLEnv()
    env = Monitor(env)

    # ==========================================
    # 2. Load 100k trained model
    # ==========================================

    model = PPO.load(
        "models_rl/ppo_robot_arm_100k",
        env=env,
        device="cpu",
    )

    print()
    print("Loaded 100k PPO model.")
    print("Starting additional 200k training...")
    print()

    # ==========================================
    # 3. Continue training
    # ==========================================

    model.learn(
        total_timesteps=200_000,
        reset_num_timesteps=False,
        progress_bar=True,
    )

    # ==========================================
    # 4. Save 300k-equivalent model
    # ==========================================

    model.save(
        "models_rl/ppo_robot_arm_300k"
    )

    print()
    print(
        "Saved: models_rl/ppo_robot_arm_300k"
    )

    env.close()


if __name__ == "__main__":
    main()

