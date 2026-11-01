"""重力補償による静止と、出力データの整合性を検証する。"""

from pathlib import Path
from uuid import uuid4
import unittest

import numpy as np

from robot_arm.dynamics import forward_dynamics
from scripts.run_gravity_compensation_demo import compensation_torque, save_results, simulate
from scripts.run_coupled_dynamics_demo import physical_parameters


class TestGravityCompensationDemo(unittest.TestCase):
    def test_static_balance_at_various_angles(self):
        parameters = physical_parameters()
        # 水平・鉛直・折り曲げ姿勢を含む、正負の角度の組み合わせ。
        for theta1 in (-2.1, 0.0, 0.7, np.pi / 2):
            for theta2 in (-1.3, 0.0, 0.9, np.pi):
                with self.subTest(theta1=theta1, theta2=theta2):
                    torque = compensation_torque(theta1, theta2, parameters)
                    alpha = forward_dynamics(theta1, theta2, 0.0, 0.0, *torque, **parameters)
                    np.testing.assert_allclose(alpha, [0, 0], rtol=0, atol=1e-12)

    def test_histories_and_static_compensation(self):
        for compensated in (False, True):
            with self.subTest(compensated=compensated):
                time, states, torques = simulate(compensated)
                np.testing.assert_allclose(time, np.arange(801) * 0.0025, rtol=0, atol=1e-14)
                self.assertEqual(states.shape, (801, 4))
                self.assertEqual(torques.shape, (801, 2))
                self.assertTrue(np.all(np.isfinite(np.column_stack((time, states, torques)))))
                if compensated:
                    np.testing.assert_allclose(states, 0, rtol=0, atol=1e-12)
                    np.testing.assert_allclose(torques[:, 0], 15.8922, rtol=0, atol=1e-12)
                    np.testing.assert_allclose(torques[:, 1], 3.1392, rtol=0, atol=1e-12)
                else:
                    self.assertGreater(np.max(np.abs(states[:, :2])), 1.0)
                    self.assertGreater(np.max(np.abs(states[:, 2:])), 1.0)
                    np.testing.assert_array_equal(torques, np.zeros((801, 2)))

    def test_csv_export_and_no_overwrite(self):
        results = {"uncompensated": simulate(False), "compensated": simulate(True)}
        # 一意な出力先を使い、生成したファイルだけを終了時に除去する。
        output = Path(__file__).resolve().parent / ("gravity_test_" + uuid4().hex)
        try:
            summary = save_results(results, output)
            for name, (time, states, torques) in results.items():
                data = np.genfromtxt(output / (name + ".csv"), delimiter=",", names=True)
                self.assertEqual(len(data), 801)
                self.assertEqual(data.dtype.names, (
                    "time_s", "theta1_rad", "theta2_rad", "omega1_rad_s",
                    "omega2_rad_s", "tau1_Nm", "tau2_Nm",
                ))
                actual = np.column_stack([data[column] for column in data.dtype.names])
                np.testing.assert_array_equal(actual, np.column_stack((time, states, torques)))
                self.assertEqual(summary[name]["samples"], 801)
            with self.assertRaises(FileExistsError):
                save_results(results, output)
        finally:
            if output.exists():
                for path in output.iterdir():
                    path.unlink()
                output.rmdir()


if __name__ == "__main__":
    unittest.main()
