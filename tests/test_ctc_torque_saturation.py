"""CTC+Iの外部トルク飽和、保持可能性、積分蓄積の検証。"""

import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.run_ctc_torque_saturation_demo import (
    CASES, DT, TARGET, GAINS, LIMITS, MODELS, simulate, model_parameters,
    saturation_mask, saturation_statistics, longest_run_steps, integral_statistics,
    metrics, target_equilibrium, save_results, case_name,
)
from scripts.run_robustness_demo import simulate as step813_simulate
from robot_arm.computed_torque_integral import computed_torque_integral
from robot_arm.actuator import saturate_torque
from robot_arm.dynamics import integrate_dynamics, forward_dynamics, mass_matrix, gravity_vector


class TestCtcTorqueSaturation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = {case: simulate(*case) for case in CASES}

    def test_unlimited_matches_step813_all_times_exactly(self):
        for gain in GAINS:
            for model, percent in (("baseline", 0), ("mismatch", 20)):
                time, states, commands, actual, errors, integrals = self.results[gain, model, "unlimited"]
                expected = step813_simulate(gain, percent, TARGET)
                for a, b in zip((time, states, commands, errors, integrals), expected):
                    np.testing.assert_array_equal(a, b)
                np.testing.assert_array_equal(actual, commands)

    def test_finite_shapes_limits_and_control_timing(self):
        self.assertEqual(len(self.results), 12)
        for (gain, model, limit), data in self.results.items():
            time, states, commands, actual, errors, integrals = data
            np.testing.assert_array_equal(time, np.arange(8001)*DT)
            self.assertEqual(states.shape, (8001, 4))
            for values in (commands, actual, errors, integrals):
                self.assertEqual(values.shape, (8001, 2))
            self.assertTrue(np.all(np.isfinite(np.column_stack(data))))
            np.testing.assert_array_equal(states[0], np.zeros(4))
            np.testing.assert_array_equal(integrals[0], np.zeros(2))
            np.testing.assert_array_equal(errors, TARGET-states[:, :2])
            np.testing.assert_array_equal(integrals[1:], np.clip(integrals[:-1]+DT*errors[:-1], -.5, .5))
            self.assertTrue(np.all(np.abs(integrals) <= .5))
            if LIMITS[limit] is not None:
                self.assertTrue(np.all(np.abs(actual) <= LIMITS[limit]))
                np.testing.assert_array_equal(actual, np.clip(commands, -LIMITS[limit], LIMITS[limit]))
            cp, pp = model_parameters(model)
            g = GAINS[gain]
            # Start, middle and final applied step: torque uses z_k, not z_(k+1).
            for i in (0, 1500, 7999):
                expected = computed_torque_integral(states[i, :2], states[i, 2:], TARGET,
                    integrals[i], [g["Kp"]]*2, [g["Kd"]]*2, [g["Ki"]]*2, cp)
                np.testing.assert_array_equal(commands[i], expected)
                np.testing.assert_array_equal(states[i+1], integrate_dynamics(
                    *states[i], *actual[i], DT, **pp, method="rk2"))

    def test_model_separation_and_no_mutation(self):
        nominal, _ = model_parameters("baseline")
        gains_before, limits_before = copy.deepcopy(GAINS), copy.deepcopy(LIMITS)
        target_before = TARGET.copy()
        for model in MODELS:
            params = model_parameters(model)
            before = copy.deepcopy(params)
            self.assertIsNot(params[0], params[1])
            with patch("scripts.run_ctc_torque_saturation_demo.model_parameters", return_value=params):
                simulate("B", model, "moderate")
            self.assertEqual(params, before)
            self.assertEqual(params[0], nominal)
            for key in nominal:
                factor = 1.2 if model == "mismatch" and key in ("m2", "I2") else 1.
                self.assertEqual(params[1][key], nominal[key]*factor)
        self.assertEqual(GAINS, gains_before)
        for key in LIMITS:
            np.testing.assert_array_equal(LIMITS[key], limits_before[key])
        np.testing.assert_array_equal(TARGET, target_before)

    def test_saturation_tolerance_denominator_and_runs(self):
        commands = np.array([[10., 5.], [10., 5.], [10., 5.], [10., 5.], [100., 100.]])
        actual = commands.copy()
        actual[0, 0] -= 5e-13  # Below the absolute + relative tolerance.
        actual[1:3, 0] -= 1e-8
        actual[2:4, 1] -= 1.
        actual[-1] = 0.  # Diagnostic row must not be counted.
        mask = saturation_mask(commands, actual)
        np.testing.assert_array_equal(mask, [[False, False], [True, False], [True, True], [False, True]])
        stats = saturation_statistics(commands, actual)
        self.assertEqual(stats["control_steps"], 4)
        self.assertEqual(stats["per_joint_steps"], [2, 2])
        self.assertEqual(stats["per_joint_fraction"], [.5, .5])
        self.assertEqual(stats["any_joint_fraction"], .75)
        for values, expected in (([], 0), ([False]*4, 0), ([True]*4, 4),
                                 ([True, True, False, True, True, True], 3)):
            self.assertEqual(longest_run_steps(values), expected)

    def test_integral_boundary_entries_are_not_dwell_counts(self):
        z = np.array([[0., 0.], [.5, -.5], [.5, -.5], [.4, -.4], [.5, -.5], [.5, -.5]])
        errors = np.zeros_like(z)
        errors[:-1] = np.diff(z, axis=0)/DT
        errors[1] = errors[4] = [1., -1.]
        saturated = np.array([[1, 0], [1, 1], [0, 0], [1, 1], [0, 1]], dtype=bool)
        stats = integral_statistics(z, errors, saturated)
        self.assertEqual(stats["limit_entry_count"], [2, 2])
        self.assertEqual(stats["at_limit_update_steps"], [4, 4])
        self.assertEqual(stats["clipping_update_steps"], [2, 2])
        self.assertEqual(stats["abs_increase_during_joint_saturation_steps"], [2, 1])
        self.assertEqual(stats["abs_increase_during_any_saturation_steps"], [2, 2])

    def test_plant_gravity_and_static_feasibility(self):
        required = []
        for model in MODELS:
            eq = target_equilibrium(model)
            _, pp = model_parameters(model)
            gravity = gravity_vector(*TARGET, **{k: pp[k] for k in ("L1", "m1", "m2", "lc1", "lc2")})
            np.testing.assert_array_equal(eq["required_gravity_torque_Nm"], gravity)
            required.append(gravity)
            for limit in LIMITS:
                v = eq["limits"][limit]
                self.assertEqual(v["static_hold_possible"], limit != "strict")
                self.assertEqual(v["feasible_per_joint"], [limit != "strict"]*2)
                acceleration = np.array(v["acceleration_with_clipped_gravity_rad_s2"])
                if limit == "strict":
                    self.assertGreater(np.linalg.norm(acceleration), 1.)
                else:
                    np.testing.assert_allclose(acceleration, 0., atol=1e-12)
        self.assertTrue(np.all(required[1] > required[0]))

    def test_saturation_changes_forward_acceleration(self):
        cp, pp = model_parameters("baseline")
        for gain in GAINS:
            _, states, commands, actual, _, _ = self.results[gain, "baseline", "moderate"]
            command_accel = forward_dynamics(*states[0], *commands[0], **pp)
            actual_accel = forward_dynamics(*states[0], *actual[0], **pp)
            np.testing.assert_allclose(command_accel, GAINS[gain]["Kp"]*TARGET, atol=1e-12)
            self.assertFalse(np.allclose(command_accel, actual_accel))
            np.testing.assert_allclose(actual_accel-command_accel,
                np.linalg.solve(mass_matrix(*states[0, :2], **cp), actual[0]-commands[0]), atol=1e-12)
            np.testing.assert_array_equal(actual[0], saturate_torque(commands[0], LIMITS["moderate"]))

    def test_metrics_match_applied_steps(self):
        for data in self.results.values():
            _, _, commands, actual, errors, integrals = data
            v = metrics(data)
            mask = saturation_mask(commands, actual)
            np.testing.assert_array_equal(v["max_abs_command_torque_Nm"], np.max(np.abs(commands[:-1]), axis=0))
            np.testing.assert_array_equal(v["max_abs_actual_torque_Nm"], np.max(np.abs(actual[:-1]), axis=0))
            np.testing.assert_array_equal(v["final_angle_error_rad"], errors[-1])
            np.testing.assert_array_equal(v["max_abs_integral_rad_s"], np.max(np.abs(integrals), axis=0))
            np.testing.assert_array_equal(v["saturation"]["per_joint_steps"], mask.sum(axis=0))
            self.assertEqual(v["saturation"]["control_steps"], 8000)
            for j in (0, 1):
                self.assertEqual(v["saturation"]["max_continuous_time_s"][j], longest_run_steps(mask[:, j])*DT)
            for values in v.values():
                if isinstance(values, list):
                    self.assertTrue(all(x is None or np.isfinite(x) for x in values))

    def test_csv_json_png_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="ctc_saturation_") as directory:
            output = Path(directory)/"results"
            summary = save_results(self.results, output)
            self.assertEqual(summary, json.loads((output/"summary.json").read_text(encoding="utf-8")))
            self.assertEqual(len(summary["cases"]), 12)
            for case, data in self.results.items():
                csv_data = np.loadtxt(output/(case_name(*case)+".csv"), delimiter=",", skiprows=1)
                self.assertEqual(csv_data.shape, (8001, 13))
                np.testing.assert_array_equal(csv_data, np.column_stack(data))
            with (output/"metrics.csv").open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 24)
            for row in rows:
                data = self.results[row["gain"], row["model"], row["limit"]]
                j = int(row["joint"])-1
                self.assertEqual(float(row["final_angle_error_rad"]), data[4][-1, j])
            images = list(output.glob("*.png"))
            self.assertEqual(len(images), 7)
            for path in images:
                self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}
            with self.assertRaises(FileExistsError):
                save_results(self.results, output)
            self.assertEqual(before, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()})

    def test_invalid_cases(self):
        for case in (("C", "baseline", "strict"), ("A", "unknown", "strict"), ("A", "baseline", "unknown")):
            with self.assertRaises(ValueError):
                simulate(*case)


if __name__ == "__main__":
    unittest.main()
