"""計算トルクの加速度再現と、サンプルホールド実験・保存結果を検証。"""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from robot_arm.computed_torque import computed_torque
from robot_arm.dynamics import (
    mass_matrix, coriolis_centrifugal_vector, gravity_vector,
    forward_dynamics, integrate_dynamics,
)
from scripts.run_coupled_dynamics_demo import physical_parameters
from scripts.run_computed_torque_demo import (
    CASES, CTC_KP, CTC_KD, DT, TARGET, simulate, save_results, settling_times,
)
from scripts.run_pd_gravity_demo import simulate as simulate_pd


class TestComputedTorque(unittest.TestCase):
    def setUp(self):
        self.p = physical_parameters()
        self.gravity_params = {k: self.p[k] for k in ('L1', 'm1', 'm2', 'lc1', 'lc2')}

    def torque(self, q, omega, target=TARGET):
        return computed_torque(q, omega, target, CTC_KP, CTC_KD, self.p)

    def test_at_rest_at_target(self):
        np.testing.assert_array_equal(self.torque(TARGET, [0, 0]), gravity_vector(*TARGET, **self.gravity_params))

    def test_requested_acceleration(self):
        # 単位慣性・c=g=0に置換するとトルクからvを直接観測できる。
        with patch('robot_arm.computed_torque.mass_matrix', return_value=np.eye(2)), \
             patch('robot_arm.computed_torque.coriolis_centrifugal_vector', return_value=np.zeros(2)), \
             patch('robot_arm.computed_torque.gravity_vector', return_value=np.zeros(2)):
            np.testing.assert_allclose(self.torque([0.1, -0.2], [0.3, -0.1]), [2, -4])

    def test_full_torque_expression(self):
        q, omega = [0.1, -0.2], [0.3, -0.1]
        c = coriolis_centrifugal_vector(*q, *omega, L1=1, m2=0.8, lc2=0.4)
        expected = mass_matrix(*q, **self.p) @ np.array([2., -4.]) + c + gravity_vector(*q, **self.gravity_params)
        np.testing.assert_allclose(self.torque(q, omega), expected, rtol=1e-14, atol=1e-14)

    def test_zero_acceleration_with_nonzero_velocity(self):
        q, omega = np.array([0.2, -0.5]), np.array([0.5, -0.25])
        target = q + CTC_KD * omega / CTC_KP
        c = coriolis_centrifugal_vector(*q, *omega, L1=1, m2=0.8, lc2=0.4)
        np.testing.assert_allclose(self.torque(q, omega, target), c + gravity_vector(*q, **self.gravity_params), atol=1e-14)

    def test_forward_dynamics_recovers_requested_acceleration(self):
        rng = np.random.default_rng(83)
        for g in (0.0, 9.81):
            self.p['g'] = g
            for _ in range(50):
                q, omega, target = rng.uniform(-2, 2, (3, 2))
                v = CTC_KP * (target - q) - CTC_KD * omega
                actual = forward_dynamics(*q, *omega, *self.torque(q, omega, target), **self.p, b1=0, b2=0)
                np.testing.assert_allclose(actual, v, rtol=1e-12, atol=1e-12)

    def test_invalid_vectors_and_gains(self):
        for key, value in [('q', [0]), ('omega', [0, np.nan]), ('q_target', [np.inf, 0]), ('Kp', [0, 25]), ('Kd', [-1, 10])]:
            args = dict(q=[0, 0], omega=[0, 0], q_target=TARGET, Kp=CTC_KP, Kd=CTC_KD, physical_params=self.p)
            args[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                computed_torque(**args)


class TestComputedTorqueDemo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = {case: simulate(case) for case in CASES}

    def test_histories_and_convergence(self):
        for case, (time, states, torques, errors) in self.results.items():
            with self.subTest(case=case):
                self.assertEqual(states.shape, (1601, 4))
                self.assertEqual(torques.shape, (1601, 2))
                self.assertEqual(errors.shape, (1601, 2))
                np.testing.assert_array_equal(time, np.arange(1601) * DT)
                self.assertEqual(time[-1], 4.0)
                np.testing.assert_array_equal(states[0], np.zeros(4))
                np.testing.assert_array_equal(errors, TARGET - states[:, :2])
                self.assertTrue(np.all(np.isfinite(np.column_stack((time, states, torques, errors)))))
                self.assertTrue(np.all(np.abs(errors[-1]) < 0.001))
                self.assertTrue(np.all(np.abs(states[-1, 2:]) < 0.01))
                settled, _ = settling_times(time, errors)
                self.assertTrue(all(t is not None and t < 4 for t in settled))

    def test_pd_matches_step_8_2_exactly(self):
        for actual, expected in zip(self.results['PD_gravity'], simulate_pd('PD_gravity')):
            np.testing.assert_array_equal(actual, expected)

    def test_sample_hold_rk2_and_recorded_torque(self):
        _, states, torques, _ = self.results['computed_torque']
        p = physical_parameters()
        for i in (0, 80, 400, 1599):
            np.testing.assert_array_equal(torques[i], computed_torque(states[i, :2], states[i, 2:], TARGET, CTC_KP, CTC_KD, p))
            expected = integrate_dynamics(*states[i], *torques[i], DT, **p, b1=0, b2=0, g=9.81, method='rk2')
            np.testing.assert_array_equal(states[i + 1], expected)

    def test_export_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(prefix='ctc_', dir=Path(__file__).parent) as directory:
            output = Path(directory) / 'results'
            summary = save_results(self.results, output)
            self.assertEqual(json.loads((output / 'summary.json').read_text()), summary)
            for case, data in self.results.items():
                csv = np.genfromtxt(output / (case + '.csv'), delimiter=',', names=True)
                self.assertEqual(csv.dtype.names, ('time_s', 'theta1_rad', 'theta2_rad', 'omega1_rad_s', 'omega2_rad_s', 'tau1_Nm', 'tau2_Nm', 'error1_rad', 'error2_rad'))
                np.testing.assert_array_equal(np.column_stack([csv[n] for n in csv.dtype.names]), np.column_stack(data))
                self.assertTrue(summary[case]['converged'])
                self.assertEqual(summary[case]['samples'], 1601)
            for name in ('angles', 'angular_velocities', 'errors', 'torques'):
                self.assertEqual((output / (name + '.png')).read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
            before = {p.name: p.read_bytes() for p in output.iterdir()}
            with self.assertRaises(FileExistsError):
                save_results(self.results, output)
            self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})

    def test_unknown_case(self):
        with self.assertRaises(ValueError):
            simulate('unknown')


if __name__ == '__main__':
    unittest.main()
