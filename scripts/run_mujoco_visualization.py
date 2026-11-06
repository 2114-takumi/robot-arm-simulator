import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "two_link_arm.xml"

sys.path.insert(0, str(ROOT))

from robot_arm.dynamics import (
    mass_matrix,
    coriolis_centrifugal_vector,
    gravity_vector,
)
from scripts.run_coupled_dynamics_demo import physical_parameters


CTC_KP = np.array([25.0, 25.0])
CTC_KD = np.array([10.0, 10.0])

LOG_PERIOD = 0.1


MOVE_DURATION = 4.0
Q_START = np.array([0.0, 0.0])
Q_GOAL = np.array([0.3, -0.4])


def desired_trajectory(t):
    """5次多項式による滑らかな点-to-点軌道."""
    T = MOVE_DURATION

    if t <= 0.0:
        return (
            Q_START.copy(),
            np.zeros(2),
            np.zeros(2),
        )
    if t >= T:
        return (
            Q_GOAL.copy(),
            np.zeros(2),
            np.zeros(2),
        )

    # 正規化時間 0 <= s <= 1
    s = t / T

    # 位置の補間係数
    h = 10 * s**3 - 15 * s**4 + 6 * s**5

    # dh/dt
    dh = (
        30 * s**2
        - 60 * s**3
        + 30 * s**4
    ) / T

    # d²h/dt²
    ddh = (
        60 * s
        - 180 * s**2
        + 120 * s**3
    ) / T**2

    delta_q = Q_GOAL - Q_START
    q_des = Q_START + delta_q * h
    dq_des = delta_q * dh
    ddq_des = delta_q * ddh

    return q_des, dq_des, ddq_des


def trajectory_computed_torque(
    q,
    dq,
    q_des,
    dq_des,
    ddq_des,
    kp,
    kd,
    params,
):
    """軌道追従用Computed Torque Control."""

    position_error = q_des - q
    velocity_error = dq_des - dq

    # CTC内部で実現したい加速度
    v = (
        ddq_des
        + kd * velocity_error
        + kp * position_error
    )

    M = mass_matrix(
        q[0],
        q[1],
        L1=params["L1"],
        m1=params["m1"],
        m2=params["m2"],
        lc1=params["lc1"],
        lc2=params["lc2"],
        I1=params["I1"],
        I2=params["I2"],
    )

    c = coriolis_centrifugal_vector(
        q[0],
        q[1],
        dq[0],
        dq[1],
        L1=params["L1"],
        m2=params["m2"],
        lc2=params["lc2"],
    )

    g = gravity_vector(
        q[0],
        q[1],
        L1=params["L1"],
        m1=params["m1"],
        m2=params["m2"],
        lc1=params["lc1"],
        lc2=params["lc2"],
        g=9.81,
    )

    tau = M @ v + c + g

    return tau


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"MuJoCo model was not found: {MODEL_PATH}"
        )

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    if model.nq != 2 or model.nv != 2:
        raise ValueError(
            f"This script expects nq=2 and nv=2, "
            f"but nq={model.nq}, nv={model.nv}"
        )

    data.qpos[:] = np.array([0.0, 0.0])
    data.qvel[:] = np.array([0.0, 0.0])

    mujoco.mj_forward(model, data)

    params = physical_parameters()

    print("Model path:", MODEL_PATH)
    print("Simulation timestep:", model.opt.timestep)
    print("CTC Kp:", CTC_KP)
    print("CTC Kd:", CTC_KD)

    next_log_time = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.perf_counter()

            # 現在状態
            q = data.qpos.copy()
            dq = data.qvel.copy()

            # 現在時刻における目標軌道
            q_des, dq_des, ddq_des = desired_trajectory(
                data.time
            )

            # 軌道追従用CTC
            tau = trajectory_computed_torque(
                q,
                dq,
                q_des,
                dq_des,
                ddq_des,
                CTC_KP,
                CTC_KD,
                params,
            )

            # MuJoCoへトルク入力
            data.qfrc_applied[:] = tau

            # シミュレーションを1ステップ進める
            mujoco.mj_step(model, data)

            # ログ
            if data.time >= next_log_time:
                q_now = data.qpos.copy()
                dq_now = data.qvel.copy()

                q_des_now, dq_des_now, _ = desired_trajectory(
                    data.time
                )

                position_error = q_des_now - q_now
                velocity_error = dq_des_now - dq_now

                print(
                    f"t={data.time:5.2f} | "
                    f"q=[{q_now[0]: .3f}, {q_now[1]: .3f}] | "
                    f"q_des=[{q_des_now[0]: .3f}, {q_des_now[1]: .3f}] | "
                    f"e=[{position_error[0]: .3f}, {position_error[1]: .3f}] | "
                    f"de=[{velocity_error[0]: .3f}, {velocity_error[1]: .3f}] | "
                    f"tau=[{tau[0]: .3f}, {tau[1]: .3f}]"
                )

                next_log_time += LOG_PERIOD

            viewer.sync()

            elapsed = time.perf_counter() - step_start
            remaining = model.opt.timestep - elapsed

            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
