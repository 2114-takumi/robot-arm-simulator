from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from robot_arm_rl_env import RobotArmRLEnv


def main():

    env = RobotArmRLEnv()
    env = Monitor(env)

    # 300kモデルを読み込む
    model = PPO.load(
        "models_rl/ppo_robot_arm_300k",
        env=env,
        device="cpu",
    )

    # Fine-tuning用に学習率を下げる
    new_learning_rate = 1e-4

    model.learning_rate = new_learning_rate
    model.lr_schedule = lambda _: new_learning_rate

    for param_group in model.policy.optimizer.param_groups:
        param_group["lr"] = new_learning_rate

    print()
    print("Loaded 300k PPO model.")
    print("Learning rate: 3e-4 -> 1e-4")
    print("Starting additional 200k fine-tuning...")
    print()

    model.learn(
        total_timesteps=200_000,
        reset_num_timesteps=False,
        progress_bar=True,
    )

    model.save(
        "models_rl/ppo_robot_arm_500k"
    )

    print()
    print("Saved: models_rl/ppo_robot_arm_500k")

    env.close()


if __name__ == "__main__":
    main()
