"""固定ゲイン42条件の再現性、モデル分離、負荷積算と出力を検証。"""

import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.run_robustness_demo import (
    GAINS, CASES, TARGETS, ERROR_PERCENTAGES, DT, model_parameters, simulate,
    metrics, failure_reasons, save_results, case_name, meets_joint2_goal,
)
from scripts.run_kp_ki_sweep_demo import simulate as step812_simulate, metrics as step812_metrics
from scripts.run_coupled_dynamics_demo import physical_parameters
from scripts.run_pd_gravity_demo import settling_times
from scripts.run_ki_sweep_demo import overshoot_percent
from robot_arm.computed_torque_integral import computed_torque_integral
from robot_arm.dynamics import integrate_dynamics


class TestRobustness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = {case: simulate(*case) for case in CASES}

    def test_step812_corresponding_cases_exact(self):
        for gain, kp, ki in (("A", 25., 7.), ("B", 40., 10.)):
            for percent, model in ((0, "baseline"), (20, "mismatch")):
                actual = self.results[gain, percent, TARGETS[0]]
                expected = step812_simulate(model, kp, ki)
                for a, e in zip(actual, expected):
                    np.testing.assert_array_equal(a, e)
                previous = step812_metrics(expected)
                for key, value in metrics(actual, TARGETS[0]).items():
                    if key in previous:
                        self.assertEqual(value, previous[key])

    def test_only_plant_m2_I2_change(self):
        nominal = physical_parameters()
        for percent in ERROR_PERCENTAGES:
            cp, pp = model_parameters(percent)
            self.assertIsNot(cp, pp)
            self.assertEqual(cp, nominal)
            for key in nominal:
                self.assertEqual(pp[key], nominal[key] * (1.+percent/100. if key in ("m2", "I2") else 1.))
            cp["m2"] = -1.
            self.assertEqual(model_parameters(percent)[0], nominal)
            self.assertGreater(pp["m2"], 0.)

    def test_inputs_and_globals_unchanged(self):
        params = model_parameters(30)
        before = copy.deepcopy(params)
        gains_before = copy.deepcopy(GAINS)
        target = np.array(TARGETS[2])
        target_before = target.copy()
        with patch("scripts.run_robustness_demo.model_parameters", return_value=params):
            simulate("B", 30, target)
        self.assertEqual(params, before)
        self.assertEqual(GAINS, gains_before)
        np.testing.assert_array_equal(target, target_before)

    def test_all_cases_finite_shapes_and_control_timing(self):
        self.assertEqual(len(self.results), 42)
        for (gain, percent, target), data in self.results.items():
            time, states, torques, errors, integrals = data
            self.assertEqual(states.shape, (8001, 4))
            for values in (torques, errors, integrals):
                self.assertEqual(values.shape, (8001, 2))
            self.assertTrue(np.all(np.isfinite(np.column_stack(data))))
            np.testing.assert_array_equal(time, np.arange(8001)*DT)
            np.testing.assert_array_equal(states[0], np.zeros(4))
            np.testing.assert_array_equal(integrals[0], np.zeros(2))
            np.testing.assert_array_equal(errors, np.array(target)-states[:, :2])
            np.testing.assert_array_equal(integrals[1:], np.clip(integrals[:-1]+DT*errors[:-1], -.5, .5))
            self.assertTrue(np.all(np.abs(integrals) <= .5))
            cp, pp = model_parameters(percent)
            g = GAINS[gain]
            for i in (0, 1000, 7999):
                expected = computed_torque_integral(states[i, :2], states[i, 2:], target,
                    integrals[i], [g["Kp"]]*2, [g["Kd"]]*2, [g["Ki"]]*2, cp)
                np.testing.assert_array_equal(torques[i], expected)
                np.testing.assert_array_equal(states[i+1], integrate_dynamics(
                    *states[i], *torques[i], DT, **pp, method="rk2"))
            for key, values in metrics(data, target).items():
                self.assertTrue(all(v is None or np.isfinite(v) for v in values), key)

    def test_signed_overshoot_and_settling_reentry(self):
        for target in TARGETS:
            angles = np.array([[0., 0.], np.array(target)*[1.1, 1.2]])
            np.testing.assert_allclose(overshoot_percent(angles, target), [10., 20.])
            errors = np.array([target, np.array(target)*.01, np.array(target)*.03,
                               np.array(target)*.01, np.array(target)*[.01, .03]])
            settled, thresholds = settling_times(np.arange(5)*5., errors)
            self.assertEqual(settled, [15., None])
            np.testing.assert_array_equal(thresholds, .02*np.abs(target))

    def test_torque_squared_integral_uses_held_intervals(self):
        # Irregular interval lengths make left-hold integration distinguishable
        # from trapezoidal integration; the large final torque is not applied.
        time = np.array([0., .25, 1.])
        states = np.zeros((3, 4))
        torques = np.array([[2., -3.], [-4., 5.], [1e6, 1e6]])
        errors = np.tile(TARGETS[0], (3, 1))
        integrals = np.array([[0., 0.], [.5, -.4], [.2, -.5]])
        values = metrics((time, states, torques, errors, integrals), TARGETS[0])
        np.testing.assert_array_equal(values["torque_squared_integral_Nm2_s"], [13., 21.])
        np.testing.assert_array_equal(values["max_abs_torque_Nm"], [4., 5.])
        np.testing.assert_array_equal(values["max_abs_integral_rad_s"], [.5, .5])
        np.testing.assert_array_equal(values["final_angle_error_rad"], TARGETS[0])

    def test_goal_boundaries_and_failure_reasons(self):
        for over, settled, passed, reasons in ((19.99, 10., True, 0), (20., 9., False, 1),
                (19., 10.01, False, 1), (25., None, False, 2)):
            v = dict(overshoot_percent=[0., over], settling_time_s=[0., settled])
            self.assertEqual(meets_joint2_goal(v), passed)
            self.assertEqual(len(failure_reasons(v)), reasons)

    def test_exports_counts_and_overwrite_rejection(self):
        with tempfile.TemporaryDirectory(prefix="robustness_") as directory:
            output = Path(directory)/"results"
            summary = save_results(self.results, output)
            self.assertEqual(summary, json.loads((output/"summary.json").read_text(encoding="utf-8")))
            self.assertEqual(len(summary["cases"]), 42)
            with (output/"metrics.csv").open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 84)
            for row in rows:
                target = (float(row["target1_rad"]), float(row["target2_rad"]))
                case = row["gain"], int(row["error_percent"]), target
                j = int(row["joint"])-1
                expected = metrics(self.results[case], target)
                for key, values in expected.items():
                    self.assertEqual(row[key], "" if values[j] is None else str(values[j]))
            with (output/"comparison.csv").open(newline="", encoding="utf-8") as file:
                self.assertEqual(len(list(csv.DictReader(file))), 42)
            for gain in GAINS:
                entries = [summary["cases"][case_name(*c)] for c in CASES if c[0] == gain]
                count = sum(v["overshoot_percent"][1] < 20. and v["settling_time_s"][1] is not None
                            and v["settling_time_s"][1] <= 10. for v in entries)
                self.assertEqual(summary["gain_summary"][gain]["goal_met"], count)
                self.assertEqual(summary["gain_summary"][gain]["total"], 21)
                self.assertEqual(len(summary["gain_summary"][gain]["failures"]), 21-count)
            images = list(output.glob("*.png"))
            self.assertEqual(len(images), 7)
            for path in images:
                self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            before = {p.name: p.read_bytes() for p in output.iterdir()}
            with self.assertRaises(FileExistsError):
                save_results(self.results, output)
            self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})

    def test_invalid_inputs(self):
        for percent in (-100, -101, np.nan, np.inf, [10, 20]):
            with self.assertRaises(ValueError):
                model_parameters(percent)
        for target in ([0., .5], [.3], [np.nan, .5], [[.3, .5]]):
            with self.assertRaises(ValueError):
                simulate("A", 0, target)
        with self.assertRaises(ValueError):
            simulate("C", 0, TARGETS[0])


if __name__ == "__main__":
    unittest.main()
