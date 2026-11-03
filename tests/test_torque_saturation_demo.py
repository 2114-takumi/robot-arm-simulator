"""トルク制限、既存応答の維持、飽和の物理と出力を検証する。"""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from robot_arm.actuator import saturate_torque
from robot_arm.computed_torque import computed_torque
from robot_arm.dynamics import forward_dynamics, mass_matrix, integrate_dynamics
from scripts.run_coupled_dynamics_demo import physical_parameters
from scripts.run_computed_torque_demo import simulate as simulate_step83
from scripts.run_torque_saturation_demo import (
    CASES, CONTROLLERS, LIMITS, CSV_HEADER, TARGET, DT, CTC_KP, CTC_KD,
    simulate, save_results, saturation_statistics, target_equilibrium,
)


class TestSaturateTorque(unittest.TestCase):
    def test_within_limits_and_boundary(self):
        for command in ([1., -1.], [20., -4.], [-20., 4.]):
            np.testing.assert_array_equal(saturate_torque(command, [20, 4]), command)

    def test_both_signs_and_independent_limits(self):
        np.testing.assert_array_equal(saturate_torque([30, -6], [20, 4]), [20, -4])
        np.testing.assert_array_equal(saturate_torque([-30, 6], [20, 4]), [-20, 4])
        np.testing.assert_array_equal(saturate_torque([3, 3], [20, 2]), [3, 2])

    def test_invalid_limits(self):
        for limits in ([0, 1], [-1, 2], [np.inf, 2], [1, np.nan], 2, [1], [[1, 2]], [1, 2, 3]):
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                saturate_torque([1, 2], limits)

    def test_invalid_commands(self):
        for command in ([np.nan, 0], [0, np.inf], 2, [1], [[1, 2]], [1, 2, 3]):
            with self.subTest(command=command), self.assertRaises(ValueError):
                saturate_torque(command, [20, 4])

    def test_input_arrays_unchanged(self):
        command, limits = np.array([30., -6.]), np.array([20., 4.])
        result = saturate_torque(command, limits)
        np.testing.assert_array_equal(command, [30, -6])
        np.testing.assert_array_equal(limits, [20, 4])
        self.assertFalse(np.shares_memory(result, command))
        self.assertFalse(np.shares_memory(result, limits))

    def test_output_always_bounded(self):
        rng = np.random.default_rng(84)
        for _ in range(100):
            limits = rng.uniform(0.01, 30, 2)
            actual = saturate_torque(rng.uniform(-100, 100, 2), limits)
            self.assertEqual(actual.shape, (2,))
            self.assertTrue(np.all(np.abs(actual) <= limits))


class TestTorqueSaturationDemo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = {case: simulate(*case) for case in CASES}

    def test_unlimited_matches_step83_exactly(self):
        for controller in CONTROLLERS:
            time, states, commands, actual, errors = self.results[controller, 'unlimited']
            for observed, expected in zip((time, states, actual, errors), simulate_step83(controller)):
                np.testing.assert_array_equal(observed, expected)
            np.testing.assert_array_equal(commands, actual)

    def test_histories_and_applied_limits(self):
        for (controller, limit), data in self.results.items():
            with self.subTest(controller=controller, limit=limit):
                time, states, commands, actual, errors = data
                self.assertEqual(states.shape, (1601, 4))
                for array in (commands, actual, errors):
                    self.assertEqual(array.shape, (1601, 2))
                np.testing.assert_array_equal(time, np.arange(1601) * DT)
                self.assertEqual(time[-1], 4.)
                np.testing.assert_array_equal(states[0], [0, 0, 0, 0])
                np.testing.assert_array_equal(errors, TARGET - states[:, :2])
                self.assertTrue(np.all(np.isfinite(np.column_stack(data))))
                if LIMITS[limit] is not None:
                    self.assertTrue(np.all(np.abs(actual) <= LIMITS[limit]))
                    np.testing.assert_array_equal(actual, np.clip(commands, -LIMITS[limit], LIMITS[limit]))

    def test_saturated_acceleration_and_model_residual(self):
        p = physical_parameters()
        for controller in CONTROLLERS:
            _, states, commands, actual, _ = self.results[controller, 'moderate']
            for i in (0, 1, 10):
                state = states[i]
                requested = forward_dynamics(*state, *commands[i], **p)
                applied = forward_dynamics(*state, *actual[i], **p)
                mismatch = np.linalg.solve(mass_matrix(*state[:2], **p), actual[i] - commands[i])
                np.testing.assert_allclose(applied - requested, mismatch, atol=1e-12, rtol=1e-12)
                if controller == 'computed_torque':
                    v = CTC_KP * (TARGET - state[:2]) - CTC_KD * state[2:]
                    np.testing.assert_allclose(requested, v, atol=1e-12, rtol=1e-12)
            self.assertGreater(np.linalg.norm(actual[0] - commands[0]), 1.)
            self.assertGreater(np.linalg.norm(forward_dynamics(*states[0], *actual[0], **p) - forward_dynamics(*states[0], *commands[0], **p)), 1.)

    def test_actual_torque_held_during_rk2(self):
        p = physical_parameters()
        for case, (_, states, _, actual, _) in self.results.items():
            for i in (0, 100, 1599):
                expected = integrate_dynamics(*states[i], *actual[i], DT, **p, b1=0., b2=0., g=9.81, method='rk2')
                np.testing.assert_array_equal(states[i + 1], expected)

    def test_gravity_equilibrium_feasibility(self):
        equilibrium = target_equilibrium()
        gravity = np.array(equilibrium['required_gravity_torque_Nm'])
        self.assertTrue(np.all(gravity < LIMITS['moderate']))
        self.assertTrue(np.all(gravity > LIMITS['strict']))
        self.assertTrue(equilibrium['limits']['moderate']['static_equilibrium_possible'])
        self.assertFalse(equilibrium['limits']['strict']['static_equilibrium_possible'])
        np.testing.assert_allclose(equilibrium['limits']['moderate']['acceleration_at_target_rad_s2'], [0, 0], atol=1e-12)
        self.assertGreater(np.linalg.norm(equilibrium['limits']['strict']['acceleration_at_target_rad_s2']), 1.)
        p = physical_parameters()
        np.testing.assert_allclose(computed_torque(TARGET, [0, 0], TARGET, CTC_KP, CTC_KD, p), gravity)

    def test_saturation_counts_tolerance_and_final_row(self):
        commands = np.array([[2., 0.], [0., -2.], [1. + 1e-13, 0.], [2., 2.]])
        actual = np.clip(commands, -1., 1.)
        statistics = saturation_statistics(commands, actual)
        self.assertEqual(statistics['control_steps'], 3)
        self.assertEqual(statistics['per_joint_steps'], [1, 1])
        self.assertEqual(statistics['any_joint_steps'], 2)
        np.testing.assert_array_equal(statistics['per_joint_fraction'], [1/3, 1/3])
        self.assertEqual(statistics['any_joint_fraction'], 2/3)
        for controller in CONTROLLERS:
            _, _, command, actual, _ = self.results[controller, 'unlimited']
            self.assertEqual(saturation_statistics(command, actual)['any_joint_steps'], 0)

    def test_exports_metrics_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(prefix='saturation_', dir=Path(__file__).parent) as directory:
            output = Path(directory) / 'results'
            summary = save_results(self.results, output)
            self.assertEqual(summary, json.loads((output / 'summary.json').read_text()))
            self.assertEqual(len(list(output.iterdir())), 11)
            for (controller, limit), data in self.results.items():
                name = controller + '_' + limit
                csv = np.genfromtxt(output / (name + '.csv'), delimiter=',', names=True)
                self.assertEqual(csv.dtype.names, tuple(CSV_HEADER.split(',')))
                np.testing.assert_array_equal(np.column_stack([csv[n] for n in csv.dtype.names]), np.column_stack(data))
                metrics = summary['cases'][name]
                self.assertEqual(metrics['saturation']['control_steps'], 1600)
                self.assertEqual(metrics['saturation'], saturation_statistics(data[2], data[3]))
                np.testing.assert_array_equal(metrics['final_angle_error_rad'], data[4][-1])
                np.testing.assert_array_equal(metrics['max_abs_actual_torque_Nm'], np.max(np.abs(data[3][:-1]), axis=0))
                if limit == 'unlimited':
                    self.assertTrue(metrics['converged'])
            for filename in ('angles.png', 'angular_velocities.png', 'errors.png', 'torques.png'):
                self.assertEqual((output / filename).read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
            before = {p.name: p.read_bytes() for p in output.iterdir()}
            with self.assertRaises(FileExistsError):
                save_results(self.results, output)
            self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})

    def test_invalid_cases(self):
        for case in (('unknown', 'unlimited'), ('computed_torque', 'unknown')):
            with self.assertRaises(ValueError):
                simulate(*case)


if __name__ == '__main__':
    unittest.main()
