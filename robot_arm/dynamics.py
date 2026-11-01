"""鉛直平面内で動く2リンクロボットの動力学計算。"""

import numpy as np


def mass_matrix(theta1, theta2, *, L1, m1, m2, lc1, lc2, I1, I2):
    """関節角度から2リンクロボットの慣性行列 M(q) を計算する。

    Args:
        theta1: 水平軸を基準とする第1関節角度 [rad]。
        theta2: 第2関節の相対角度 [rad]。
        L1: 第1リンクの長さ [m]。
        m1, m2: 各リンクの質量 [kg]。
        lc1, lc2: 各関節から対応するリンク重心までの距離 [m]。
        I1, I2: 各リンクの重心まわりの慣性モーメント [kg m^2]。

    Returns:
        shape=(2, 2) の numpy.ndarray。各要素の単位は [kg m^2]。

    物理パラメータはキーワード専用引数で指定する。
    このモデルの慣性行列は相対角度 theta2 のみに依存し、theta1 には
    依存しない。L2 は lc2 と I2 を通じて反映されるため引数に含めない。
    """
    cos_theta2 = np.cos(theta2)

    # 第1関節の慣性には、両リンクの回転と重心の並進が寄与する。
    M11 = I1 + I2 + m1 * lc1**2 + m2 * (
        L1**2 + lc2**2 + 2 * L1 * lc2 * cos_theta2
    )
    # 関節間の慣性結合。対称性により M21 = M12 となる。
    M12 = I2 + m2 * (lc2**2 + L1 * lc2 * cos_theta2)
    M21 = M12
    # 第2関節まわりの慣性（重心まわりの慣性に平行軸の項を加える）。
    M22 = I2 + m2 * lc2**2

    return np.array([[M11, M12], [M21, M22]], dtype=float)


def coriolis_centrifugal_vector(theta1, theta2, omega1, omega2, *, L1, m2, lc2):
    """コリオリ力・遠心力による速度依存トルクを計算する。

    Args:
        theta1: 水平軸を基準とする第1関節角度 [rad]。今回の式には現れない。
        theta2: 第2関節の相対角度 [rad]。
        omega1, omega2: 各関節角度の時間微分（角速度）[rad/s]。
        L1: 第1リンクの長さ [m]。
        m2: 第2リンクの質量 [kg]。
        lc2: 第2関節から第2リンク重心までの距離 [m]。

    Returns:
        C(q, qdot) @ qdot に相当する shape=(2,) の numpy.ndarray [N·m]。
        運動方程式の左辺に置く [c1, c2] であり、C行列自体は返さない。
    """
    # リンク間の姿勢と、第2リンク重心の位置で決まる結合係数。
    h = m2 * L1 * lc2 * np.sin(theta2)

    # 第1関節の項：角速度の積と、第2関節角速度の2乗に依存する。
    c1 = -h * (2 * omega1 * omega2 + omega2**2)
    # 第2関節の項：第1関節角速度の2乗に依存する。
    c2 = h * omega1**2

    return np.array([c1, c2])


def gravity_vector(theta1, theta2, *, L1, m1, m2, lc1, lc2, g=9.81):
    """鉛直面内の2リンクロボットの重力項を計算する。

    y軸は上向き、theta1 は水平右向きから測り、theta2 は相対角度とする。
    重力は下向きに作用し、g はその加速度の正の大きさで指定する。

    Args:
        theta1: 第1関節角度 [rad]。
        theta2: 第2関節の相対角度 [rad]。
        L1: 第1リンクの長さ [m]。
        m1, m2: 各リンクの質量 [kg]。
        lc1, lc2: 各関節から対応するリンク重心までの距離 [m]。
        g: 重力加速度の大きさ [m/s^2]。標準値は 9.81。

    Returns:
        運動方程式の左辺に置く重力項 [g1, g2]。
        shape=(2,) の numpy.ndarray で、単位は [N·m]。
        静止状態で重力を補償するモータトルクと同じ符号を持つ。
    """
    # 第1関節には両リンクの重さが寄与する。cos は水平の腕の長さに対応。
    g1 = (m1 * lc1 + m2 * L1) * g * np.cos(theta1) + m2 * lc2 * g * np.cos(
        theta1 + theta2
    )
    # 第2関節には第2リンクの重さが寄与する。
    g2 = m2 * lc2 * g * np.cos(theta1 + theta2)

    return np.array([g1, g2])
