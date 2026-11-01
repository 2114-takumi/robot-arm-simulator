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
