"""PD制御器と3条件比較の物理的な挙動・保存結果を検証する。"""

from pathlib import Path
import unittest
from uuid import uuid4

import numpy as np

from robot_arm.controllers import pd_torque, pd_gravity_torque
from robot_arm.dynamics import gravity_vector
from scripts.run_pd_gravity_demo import CASES, DT, KP, KD, TARGET, save_results, settling_times, simulate


class TestPDTorque(unittest.TestCase):
    def test_zero_error_at_rest(self):
        np.testing.assert_array_equal(pd_torque(TARGET, [0, 0], TARGET, Kp=KP, Kd=KD), [0, 0])

    def test_gravity_compensation_at_target(self):
        parameters = dict(L1=1.0, m1=1.0, m2=0.8, lc1=0.5, lc2=0.4)
        torque = pd_gravity_torque(TARGET, [0, 0], TARGET, Kp=KP, Kd=KD, **parameters)
        np.testing.assert_array_equal(torque, gravity_vector(*TARGET, **parameters))

    def test_derivative_opposes_velocity(self):
        torque = pd_torque(TARGET, [0.5, -0.7], TARGET, Kp=KP, Kd=KD)
        np.testing.assert_allclose(torque, [-4.0, 3.5])
        self.assertTrue(np.all(torque * [0.5, -0.7] < 0))

    def test_zero_gains_and_proportional_term(self):
        np.testing.assert_array_equal(pd_torque([0, 0], [2, -3], TARGET, Kp=[0, 0], Kd=[0, 0]), [0, 0])
        np.testing.assert_allclose(pd_torque([0, 0], [2, -3], TARGET, Kp=KP, Kd=[0, 0]), [9, -8])
        np.testing.assert_allclose(pd_torque([0, 0], [2, -3], TARGET, Kp=[0, 0], Kd=KD), [-16, 15])
        parameters = dict(L1=1.0, m1=1.0, m2=0.8, lc1=0.5, lc2=0.4)
        np.testing.assert_array_equal(
            pd_gravity_torque([0, 0], [2, -3], TARGET, Kp=[0, 0], Kd=[0, 0], **parameters),
            gravity_vector(0, 0, **parameters),
        )


class TestPDGravityDemo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = {case: simulate(case) for case in CASES}

    def test_histories_and_compensated_convergence(self):
        for case, (time, states, torques, errors) in self.results.items():
            with self.subTest(case=case):
                self.assertEqual(states.shape, (1601, 4))
                self.assertEqual(torques.shape, (1601, 2))
                self.assertEqual(errors.shape, (1601, 2))
                np.testing.assert_array_equal(time, np.arange(1601) * DT)
                self.assertEqual(time[-1], 4.0)
                np.testing.assert_array_equal(states[0], [0, 0, 0, 0])
                np.testing.assert_array_equal(errors, TARGET - states[:, :2])
                self.assertTrue(np.all(np.isfinite(np.column_stack((time, states, torques, errors)))))
        time, states, _, errors = self.results["PD_gravity"]
        self.assertLess(np.max(np.abs(errors[-1])), 0.001)
        self.assertLess(np.max(np.abs(states[-1, 2:])), 0.01)
        times, _ = settling_times(time, errors)
        self.assertTrue(all(value is not None and value < 4 for value in times))

    def test_settling_definition_and_zero_target_scale(self):
        time = np.array([0., 1., 2., 3.])
        errors = np.array([[0., 1.], [0.0003, 0.01], [0.0001, 0.03], [0., 0.01]])
        times, thresholds = settling_times(time, errors)
        np.testing.assert_allclose(thresholds, [0.0002, 0.02])
        self.assertEqual(times, [2.0, 3.0])
        times, _ = settling_times(time, np.ones((4, 2)))
        self.assertEqual(times, [None, None])

    def test_csv_export_and_no_overwrite(self):
        output = Path(__file__).resolve().parent / ("pd_test_" + uuid4().hex)
        try:
            summary = save_results(self.results, output)
            for case, data in self.results.items():
                csv = np.genfromtxt(output / (case + ".csv"), delimiter=",", names=True)
                self.assertEqual(csv.shape, (1601,))
                self.assertEqual(csv.dtype.names, (
                    "time_s", "theta1_rad", "theta2_rad", "omega1_rad_s", "omega2_rad_s",
                    "tau1_Nm", "tau2_Nm", "error1_rad", "error2_rad",
                ))
                actual = np.column_stack([csv[name] for name in csv.dtype.names])
                np.testing.assert_array_equal(actual, np.column_stack(data))
                self.assertEqual(summary[case]["samples"], 1601)
            with self.assertRaises(FileExistsError):
                save_results(self.results, output)
        finally:
            if output.exists():
                for path in output.iterdir():
                    path.unlink()
                output.rmdir()


if __name__ == "__main__":
    unittest.main()
