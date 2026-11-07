import numpy as np


class JointEncoderSensor:
    """2関節のエンコーダ相当センサモデル."""

    def __init__(
        self,
        *,
        counts_per_rev=4096,
        angle_noise_std=0.0005,
        velocity_cutoff_hz=20.0,
        seed=0,
    ):
        if counts_per_rev <= 0:
            raise ValueError(
                "counts_per_rev must be positive"
            )

        if angle_noise_std < 0:
            raise ValueError(
                "angle_noise_std must be nonnegative"
            )

        if velocity_cutoff_hz <= 0:
            raise ValueError(
                "velocity_cutoff_hz must be positive"
            )

        self.counts_per_rev = counts_per_rev
        self.angle_noise_std = angle_noise_std
        self.velocity_cutoff_hz = velocity_cutoff_hz

        self.angle_resolution = (
            2.0 * np.pi / counts_per_rev
        )

        self.rng = np.random.default_rng(seed)

        self.previous_q = None
        self.filtered_dq = np.zeros(2)


    def reset(self):
        """センサ内部状態を初期化する."""

        self.previous_q = None
        self.filtered_dq[:] = 0.0


    def measure(self, q_true, dt):
        """真の関節角度から測定角度・推定角速度を返す."""

        q_true = np.asarray(
            q_true,
            dtype=float,
        )

        if q_true.shape != (2,):
            raise ValueError(
                "q_true must have shape (2,)"
            )

        if not np.isfinite(dt) or dt <= 0:
            raise ValueError(
                "dt must be positive and finite"
            )

        # 1. 角度測定ノイズ
        noise = self.rng.normal(
            loc=0.0,
            scale=self.angle_noise_std,
            size=2,
        )

        q_noisy = (
            q_true + noise
        )

        # 2. エンコーダ量子化
        q_measured = (
            np.round(
                q_noisy
                / self.angle_resolution
            )
            * self.angle_resolution
        )

        # 初回は速度を0とする
        if self.previous_q is None:
            self.previous_q = (
                q_measured.copy()
            )

            self.filtered_dq[:] = 0.0

            return (
                q_measured,
                self.filtered_dq.copy(),
            )

        # 3. 角度差分から速度を推定
        raw_dq = (
            q_measured
            - self.previous_q
        ) / dt

        # 4. 一次Low-pass filter
        tau = (
            1.0
            / (
                2.0
                * np.pi
                * self.velocity_cutoff_hz
            )
        )

        alpha = (
            dt / (tau + dt)
        )

        self.filtered_dq = (
            self.filtered_dq
            + alpha
            * (
                raw_dq
                - self.filtered_dq
            )
        )

        self.previous_q = (
            q_measured.copy()
        )

        return (
            q_measured,
            self.filtered_dq.copy(),
        )