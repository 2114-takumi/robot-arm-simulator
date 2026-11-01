"""2関節のトルク制御器。既存シミュレーターには接続しない。"""

import numpy as np

from robot_arm.dynamics import gravity_vector


def pd_torque(q, omega, q_target, *, Kp, Kd):
    """静止目標へのPDトルク [N·m] をshape=(2,)の配列で返す。

    q, q_target は関節角度 [rad]、omega は角速度 [rad/s] の2成分。
    Kp [N·m/rad]、Kd [N·m·s/rad] は対角ゲインの2成分で、
    非負値を指定する。Kd=0でP制御、両ゲイン0で出力0となる。
    """
    q = np.asarray(q, dtype=float)
    omega = np.asarray(omega, dtype=float)
    q_target = np.asarray(q_target, dtype=float)
    Kp = np.asarray(Kp, dtype=float)
    Kd = np.asarray(Kd, dtype=float)
    for value in (q, omega, q_target, Kp, Kd):
        if value.shape != (2,) or not np.all(np.isfinite(value)):
            raise ValueError("States, target and gains must be finite two-component vectors")
    if np.any(Kp < 0) or np.any(Kd < 0):
        raise ValueError("Gains must be nonnegative")
    # 各関節の位置誤差を戻すP項と、現在の速度を抑えるD項。
    error = q_target - q
    return Kp * error - Kd * omega


def pd_gravity_torque(q, omega, q_target, *, Kp, Kd, L1, m1, m2, lc1, lc2, g=9.81):
    """PDトルクに現在姿勢の重力補償を加え、2成分のトルク [N·m] を返す。

    q, q_target [rad]、omega [rad/s]、Kp [N·m/rad]、Kd [N·m·s/rad]。
    L1, lc1, lc2 [m]、m1, m2 [kg]、g [m/s^2] は動力学と同じ値を使う。
    目標速度は0とし、重力項は既存のgravity_vector()で計算する。
    """
    pd = pd_torque(q, omega, q_target, Kp=Kp, Kd=Kd)
    gravity = gravity_vector(q[0], q[1], L1=L1, m1=m1, m2=m2, lc1=lc1, lc2=lc2, g=g)
    return pd + gravity
