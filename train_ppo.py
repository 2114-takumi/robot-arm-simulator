import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from robot_arm_rl_env import RobotArmRLEnv


def main():

    # ==========================================
    # 1. Create environment
    # ==========================================

    env = RobotArmRLEnv()

    print("Checking environment...")

    check_env(
        env,
        warn=True,
    )

    print("Environment check passed.")

    env.close()

    # ==========================================
    # 2. Monitor wrapper
    # ==========================================

    env = RobotArmRLEnv()

    env = Monitor(
        env
    )

    # ==========================================
    # 3. PPO model
    # ==========================================

    model = PPO(
        policy="MlpPolicy",
        env=env,

        learning_rate=3e-4,

        n_steps=2048,
        batch_size=64,
        n_epochs=10,

        gamma=0.99,
        gae_lambda=0.95,

        clip_range=0.2,

        verbose=1,

        seed=0,

        device="cpu",
    )

    # ==========================================
    # 4. Train
    # ==========================================

    print()
    print("Starting PPO training...")
    print()

    model.learn(
        total_timesteps=100_000,
        progress_bar=True,
    )

    # ==========================================
    # 5. Save model
    # ==========================================

    os.makedirs(
        "models_rl",
        exist_ok=True,
    )

    model_path = (
        "models_rl/ppo_robot_arm"
    )

    model.save(
        model_path
    )

    print()
    print(
        f"Model saved to: {model_path}"
    )

    env.close()


if __name__ == "__main__":
    main()
