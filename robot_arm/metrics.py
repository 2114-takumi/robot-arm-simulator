import numpy as np


# 2%整定時間を計算する関数
def calculate_settling_time(times, angles, target):
    initial_error = abs(target - angles[0])
    threshold = 0.02 * initial_error

    for i in range(len(angles)):
        # この時刻以降、すべて許容範囲内にあるか
        remaining_errors = np.abs(
            np.array(angles[i:]) - target
        )

        if np.all(remaining_errors <= threshold):
            return times[i]

    return None
