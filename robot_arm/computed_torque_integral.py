"""既存の2リンク動力学を用いる計算トルク制御（摩擦補償なし）。"""

import numpy as np

from robot_arm.dynamics import (
    mass_matrix, coriolis_centrifugal_vector, gravity_vector,
)


def computed_torque_integral(q, omega, q_target, integral_error, Kp, Kd, Ki, physical_params):
    """静止目標へのトルク [N·m] をshape=(2,)で返す。

    integral_error [rad*s] は外部で保持・制限する積分状態。
    Ki [1/s³] は非負の2成分。内部での積分更新・隠れた状態はない。
    q, q_target [rad]、omega [rad/s] は有限な2成分ベクトル。
    Kp [1/s²]、Kd [1/s] は正の対角ゲインの2成分であり、
    トルク出力型PDのゲインとは単位が異なる。
    physical_paramsは既存動力学のキーワード引数を格納した辞書。
    必須キー: L1, m1, m2, lc1, lc2, I1, I2。gの既定値は9.81。
    b1, b2を含んでいても摩擦補償には使用しない。
    """
    integral_error = np.asarray(integral_error, dtype=float)
    Ki = np.asarray(Ki, dtype=float)
    for value in (integral_error, Ki):
        if value.shape != (2,) or not np.all(np.isfinite(value)):
            raise ValueError("Integral state and Ki must be finite two-component vectors")
    if np.any(Ki < 0):
        raise ValueError("Ki must be nonnegative")
    q, omega, q_target, Kp, Kd = (
        np.asarray(value, dtype=float) for value in (q, omega, q_target, Kp, Kd)
    )
    for value in (q, omega, q_target, Kp, Kd):
        if value.shape != (2,) or not np.all(np.isfinite(value)):
            raise ValueError("States, target and gains must be finite two-component vectors")
    if np.any(Kp <= 0) or np.any(Kd <= 0):
        raise ValueError("Gains must be positive")

    error = q_target - q
    v = Kp * error + Ki * integral_error - Kd * omega
    M = mass_matrix(*q, **{key: physical_params[key] for key in (
        "L1", "m1", "m2", "lc1", "lc2", "I1", "I2",
    )})
    c = coriolis_centrifugal_vector(
        *q, *omega, **{key: physical_params[key] for key in ("L1", "m2", "lc2")}
    )
    g = gravity_vector(
        *q, **{key: physical_params[key] for key in ("L1", "m1", "m2", "lc1", "lc2")},
        g=physical_params.get("g", 9.81),
    )
    return M @ v + c + g
