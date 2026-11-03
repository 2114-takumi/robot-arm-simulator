"""制御器と動力学の間に適用するアクチュエータのトルク制限。"""

import numpy as np


def saturate_torque(tau_command, tau_max):
    """2関節の要求トルクを±tau_max [N·m]に制限した新規配列を返す。

    両入力は有限なshape=(2,)、tau_maxは各成分が正であること。
    入力配列は変更しない。制限なしは呼び出し側で処理する。
    """
    command = np.asarray(tau_command, dtype=float)
    limits = np.asarray(tau_max, dtype=float)
    for value in (command, limits):
        if value.shape != (2,) or not np.all(np.isfinite(value)):
            raise ValueError("Torques and limits must be finite two-component vectors")
    if np.any(limits <= 0):
        raise ValueError("Torque limits must be positive")
    return np.clip(command, -limits, limits)
