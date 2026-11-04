"""Step 8-17: 逆算の符号・連成、全軌道一致、更新時刻と失敗の保存。"""

import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import run_back_calculation_demo as demo
from scripts.run_per_joint_anti_windup_demo import simulate as previous_simulate, metrics as previous_metrics
from robot_arm.computed_torque_integral import computed_torque_integral
from robot_arm.dynamics import mass_matrix, integrate_dynamics


class TestBackCalculation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = {case: demo.run_case(case) for case in demo.CASES}

    def test_no_torque_difference_means_zero_correction(self):
        cp, _ = demo.model_parameters("baseline")
        z, e = np.array([.1, -.2]), np.array([.3, -.4])
        for kaw in demo.KAW_VALUES:
            updated, delta, correction = demo.back_calculation_update(
                [0., 0.], z, e, [20., -4.], [20., -4.], [7., 10.], kaw, cp)
            np.testing.assert_array_equal(delta, [0., 0.])
            np.testing.assert_array_equal(correction, [0., 0.])
            np.testing.assert_array_equal(updated, z+demo.DT*e)

    def test_positive_negative_saturation_and_coupling_use_solve(self):
        cp, _ = demo.model_parameters("baseline")
        ki, kaw = np.array([7., 10.]), 2.
        for command, actual, signs in (
            ([21., 0.], [20., 0.], [-1., 1.]),
            ([-21., 0.], [-20., 0.], [1., -1.]),
            ([0., 5.], [0., 4.], [1., -1.]),
            ([0., -5.], [0., -4.], [-1., 1.]),
        ):
            with patch.object(demo.np.linalg, "inv", side_effect=AssertionError("Do not form inverse")), \
                 patch.object(demo.np.linalg, "solve", wraps=np.linalg.solve) as solve:
                updated, delta, correction = demo.back_calculation_update(
                    [0., 0.], [0., 0.], [0., 0.], command, actual, ki, kaw, cp)
                solve.assert_called_once()
            matrix = mass_matrix(0., 0., **cp)
            np.testing.assert_allclose(matrix @ (correction*ki/kaw), delta, atol=1e-14, rtol=1e-14)
            np.testing.assert_array_equal(np.sign(correction), signs)
            np.testing.assert_array_equal(updated, np.clip(demo.DT*correction, -.5, .5))

    def test_clip_and_inputs_unchanged(self):
        cp, _ = demo.model_parameters("baseline")
        arrays = [np.array(v) for v in ([.2, -.3], [.499, -.499], [1000., -1000.],
                                       [1., 2.], [1., 2.], [7., 10.])]
        before, params_before = [v.copy() for v in arrays], copy.deepcopy(cp)
        updated, _, _ = demo.back_calculation_update(*arrays, 1., cp)
        np.testing.assert_array_equal(updated, [.5, -.5])
        for a, b in zip(arrays, before):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(cp, params_before)
        self.assertFalse(np.shares_memory(updated, arrays[1]))

    def test_unlimited_all_kaw_match_standard_at_all_times(self):
        for variant in demo.BACK_CALCULATION:
            for gain, model in demo.GROUPS:
                data = self.results[variant, gain, model, "unlimited"]["data"]
                expected = self.results["standard", gain, model, "unlimited"]["data"]
                for a, b in zip(data, expected):
                    np.testing.assert_array_equal(a, b)

    def test_kaw_zero_all_twelve_match_standard_even_saturated(self):
        for gain, model in demo.GROUPS:
            for limit in demo.LIMITS:
                data = demo.simulate(0., gain, model, limit)
                expected = self.results["standard", gain, model, limit]["data"]
                for a, b in zip(data, expected):
                    np.testing.assert_array_equal(a, b)

    def test_all_36_references_match_step816_at_all_times(self):
        for method in demo.METHODS:
            for gain, model in demo.GROUPS:
                for limit in demo.LIMITS:
                    case = method, gain, model, limit
                    data = self.results[case]["data"]
                    expected = previous_simulate(*case)
                    for a, b in zip(data[:7], expected):
                        np.testing.assert_array_equal(a, b)
                    for key, value in previous_metrics(expected).items():
                        self.assertEqual(demo.metrics(data)[key], value)
                    # 保存済みStep 8-16がある環境では、ファイルの全時刻とも照合する。
                    path = demo.ROOT/"outputs"/"step_8_16"/(demo.case_name(*case)+".csv")
                    if path.exists():
                        np.testing.assert_array_equal(np.column_stack(data[:7]),
                            np.loadtxt(path, delimiter=",", skiprows=1))

    def test_all_96_finite_shapes_and_integral_update_timing(self):
        self.assertEqual(len(self.results), 96)
        for case, record in self.results.items():
            with self.subTest(case=case):
                self.assertEqual(record["status"], "ok")
                time, states, command, actual, error, integral, stopped, delta, correction = record["data"]
                np.testing.assert_array_equal(time, np.arange(8001)*demo.DT)
                self.assertEqual(states.shape, (8001, 4))
                for a in (command, actual, error, integral, stopped, delta, correction):
                    self.assertEqual(a.shape, (8001, 2))
                self.assertTrue(np.all(np.isfinite(np.column_stack(record["data"]))))
                np.testing.assert_array_equal(error, demo.TARGET-states[:, :2])
                np.testing.assert_array_equal(delta, actual-command)
                np.testing.assert_array_equal(states[0], np.zeros(4))
                np.testing.assert_array_equal(integral[0], np.zeros(2))
                self.assertTrue(np.all(np.abs(integral) <= .5))
                np.testing.assert_array_equal(integral[1:], np.where(stopped[:-1], integral[:-1],
                    np.clip(integral[:-1]+demo.DT*(error[:-1]+correction[:-1]), -.5, .5)))
                if case[0] in demo.BACK_CALCULATION:
                    self.assertFalse(np.any(stopped))
                if demo.LIMITS[case[3]] is None:
                    np.testing.assert_array_equal(actual, command)
                else:
                    bound = demo.LIMITS[case[3]]
                    np.testing.assert_array_equal(actual, np.clip(command, -bound, bound))

    def test_current_integral_torque_hold_and_nominal_correction_model(self):
        for case, record in self.results.items():
            variant, gain, model, _ = case
            time, states, command, actual, error, integral, _, delta, correction = record["data"]
            cp, pp = demo.model_parameters(model)
            g = demo.GAINS[gain]
            for i in (0, 1, 100, 7999):
                np.testing.assert_array_equal(command[i], computed_torque_integral(
                    states[i, :2], states[i, 2:], demo.TARGET, integral[i],
                    [g["Kp"]]*2, [g["Kd"]]*2, [g["Ki"]]*2, cp))
                np.testing.assert_array_equal(states[i+1], integrate_dynamics(
                    *states[i], *actual[i], demo.DT, **pp, b1=0., b2=0., g=9.81, method="rk2"))
                if variant in demo.BACK_CALCULATION:
                    matrix = mass_matrix(*states[i, :2], **cp)
                    expected = demo.BACK_CALCULATION[variant]*np.linalg.solve(matrix, delta[i])/g["Ki"]
                    np.testing.assert_array_equal(correction[i], expected)

    def test_model_dictionaries_and_constants_unchanged(self):
        before = copy.deepcopy((demo.GAINS, demo.LIMITS, demo.TARGET))
        nominal, _ = demo.model_parameters("baseline")
        for model in demo.MODELS:
            params = demo.model_parameters(model)
            original = copy.deepcopy(params)
            self.assertIsNot(params[0], params[1])
            with patch.object(demo, "model_parameters", return_value=params), patch.object(demo, "DURATION", .01):
                demo.simulate(2., "B", model, "moderate")
            self.assertEqual(params, original)
            self.assertEqual(params[0], nominal)
            for key in nominal:
                self.assertEqual(params[1][key], nominal[key]*(1.2 if model == "mismatch" and key in ("m2", "I2") else 1.))
        self.assertEqual(demo.GAINS, before[0])
        for key in demo.LIMITS:
            np.testing.assert_array_equal(demo.LIMITS[key], before[1][key])
        np.testing.assert_array_equal(demo.TARGET, before[2])

    def test_metrics_for_corrected_clipping_delta_and_terminal_row(self):
        for case, record in self.results.items():
            data = record["data"]
            values = demo.metrics(data)
            delta, correction = data[7:]
            np.testing.assert_array_equal(values["max_abs_correction_rad"], np.max(np.abs(correction[:-1]), axis=0))
            np.testing.assert_array_equal(values["max_abs_delta_tau_Nm"], np.max(np.abs(delta[:-1]), axis=0))
            np.testing.assert_array_equal(values["delta_tau_squared_integral_Nm2_s"],
                np.sum(delta[:-1]**2*np.diff(data[0])[:, None], axis=0))
            raw = data[5][:-1]+demo.DT*(data[4][:-1]+correction[:-1])
            expected = ((np.abs(raw) > .5+demo.INTEGRAL_ATOL) & ~data[6][:-1]).sum(axis=0)
            np.testing.assert_array_equal(values["integral_limits"]["clipping_update_steps"], expected)
        data = tuple(a[:3].copy() for a in self.results["back_calculation_1", "B", "baseline", "unlimited"]["data"])
        data[7][:] = [[1., -2.], [3., 4.], [999., 999.]]
        data[8][:] = [[-2., 1.], [1., -3.], [999., 999.]]
        v = demo.metrics(data)
        self.assertEqual(v["max_abs_delta_tau_Nm"], [3., 4.])
        self.assertEqual(v["max_abs_correction_rad"], [2., 3.])
        np.testing.assert_allclose(v["delta_tau_squared_integral_Nm2_s"], [10*demo.DT, 20*demo.DT])

    def test_failed_candidates_are_recorded_and_exported(self):
        case = "back_calculation_1", "B", "mismatch", "moderate"
        with patch.object(demo, "DURATION", .005), \
             patch.object(demo, "computed_torque_integral", return_value=np.array([np.inf, 0.])):
            failed = demo.run_case(case)
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["failed_step"], 0)
        self.assertIn("ValueError", failed["reason"])
        with patch.object(demo, "DURATION", .005), \
             patch.object(demo, "integrate_dynamics", return_value=np.array([2*demo.STATE_GUARD, 0., 0., 0.])):
            guarded = demo.run_case(case)
        self.assertEqual(guarded["status"], "failed")
        self.assertEqual(guarded["failed_step"], 1)
        self.assertIn("guard", guarded["reason"])
        with patch.object(demo, "DURATION", .005):
            self.assertEqual(demo.run_case(case)["status"], "ok")
        small = {k: dict(status="ok", data=tuple(a[:3].copy() for a in v["data"])) for k, v in self.results.items()}
        small[case] = failed
        with tempfile.TemporaryDirectory(prefix="bc_failed_") as directory, patch.object(demo, "save_plots"):
            summary = demo.save_results(small, Path(directory)/"result")
            entry = summary["cases"][demo.case_name(*case)]
            self.assertEqual(entry["status"], "failed")
            self.assertFalse(entry["joint2_goal_met"])
            self.assertNotIn("settling_time_s", entry)
            self.assertEqual(summary["goal_counts"][case[0]]["failed"], 1)
            self.assertEqual(len(summary["cases"]), 96)

    def test_exports_all_cases_plots_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="back_calculation_") as directory:
            output = Path(directory)/"results"
            summary = demo.save_results(self.results, output)
            self.assertEqual(summary, json.loads((output/"summary.json").read_text(encoding="utf-8")))
            self.assertEqual(len(summary["cases"]), 96)
            self.assertEqual(len(summary["comparisons"]), 180)
            for case, record in self.results.items():
                csv_data = np.loadtxt(output/(demo.case_name(*case)+".csv"), delimiter=",", skiprows=1)
                self.assertEqual(csv_data.shape, (8001, 19))
                np.testing.assert_array_equal(csv_data, np.column_stack(record["data"]))
                self.assertEqual(summary["cases"][demo.case_name(*case)]["static_hold_possible"], case[3] != "strict")
            for filename, expected in (("metrics.csv", 192), ("comparison.csv", 360), ("focus_comparison.csv", 16)):
                with (output/filename).open(newline="", encoding="utf-8") as file:
                    self.assertEqual(len(list(csv.DictReader(file))), expected)
            images = list(output.glob("*.png"))
            self.assertEqual(len(images), 23)
            for path in images:
                self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}
            with self.assertRaises(FileExistsError):
                demo.save_results(self.results, output)
            self.assertEqual(before, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()})
            empty = Path(directory)/"empty"
            empty.mkdir()
            with self.assertRaises(FileExistsError):
                demo.save_results(self.results, empty)
            self.assertEqual(list(empty.iterdir()), [])

    def test_invalid_gains_inputs_and_cases(self):
        cp, _ = demo.model_parameters("baseline")
        for ki, kaw in (([0., 1.], 1.), ([-1., 1.], 1.), ([1.], 1.), ([1., np.nan], 1.),
                        ([1., 1.], -1.), ([1., 1.], np.inf), ([1., 1.], [1., 2.])):
            with self.assertRaises(ValueError):
                demo.back_calculation_update([0., 0.], [0., 0.], [1., 1.], [2., 2.], [1., 1.], ki, kaw, cp)
        for q, dt in (([np.nan, 0.], demo.DT), ([0.], demo.DT), ([0., 0.], 0.)):
            with self.assertRaises(ValueError):
                demo.back_calculation_update(q, [0., 0.], [1., 1.], [2., 2.], [1., 1.], [1., 1.], 1., cp, dt)
        for case in ((1., "X", "baseline", "strict"), (1., "A", "bad", "strict"), (1., "A", "baseline", "bad")):
            with self.assertRaises(ValueError):
                demo.simulate(*case)


if __name__ == "__main__":
    unittest.main()
