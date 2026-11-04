"""両関節同時の条件付き積分、更新順序、既存軌道との一致を検証。"""

import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.run_anti_windup_demo import (
    CASES, METHODS, GAINS, MODELS, LIMITS, DT, TARGET, update_with_saturation,
    simulate, metrics, model_parameters, compare_pair, save_results, case_name,
)
from scripts.run_ctc_torque_saturation_demo import (
    simulate as previous_simulate, metrics as previous_metrics, saturation_mask,
    CASES as PREVIOUS_CASES,
)
from robot_arm.computed_torque_integral import computed_torque_integral
from robot_arm.dynamics import integrate_dynamics


class TestAntiWindup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = {case: simulate(*case) for case in CASES}
        cls.previous = {case: previous_simulate(*case) for case in PREVIOUS_CASES}

    def test_either_joint_stops_both_with_tolerance(self):
        z = np.array([.1, -.2])
        error = np.array([.3, -.4])
        before = z.copy()
        for command, actual, expected in (
            ([21., 0.], [20., 0.], True),
            ([0., -5.], [0., -4.], True),
            ([20., 4.], [20., 4.], False),
            ([20., 4.], [20.-1e-12, 4.], False),
            ([20., 4.], [20.-1e-8, 4.], True),
        ):
            new, stopped = update_with_saturation(z, error, command, actual, "conditional")
            self.assertEqual(stopped, expected)
            np.testing.assert_array_equal(new, z if expected else z+DT*error)
            self.assertFalse(np.shares_memory(new, z))
            normal, stopped = update_with_saturation(z, error, command, actual, "standard")
            self.assertFalse(stopped)
            np.testing.assert_array_equal(normal, z+DT*error)
        np.testing.assert_array_equal(z, before)

    def test_resume_and_clipping_are_current_step_decisions(self):
        z = np.array([.499, -.499])
        e = np.array([2., -2.])
        frozen, stopped = update_with_saturation(z, e, [21., 0.], [20., 0.], "conditional")
        self.assertTrue(stopped)
        np.testing.assert_array_equal(frozen, z)
        resumed, stopped = update_with_saturation(frozen, e, [19., 0.], [19., 0.], "conditional")
        self.assertFalse(stopped)
        np.testing.assert_array_equal(resumed, [.5, -.5])
        unchanged, stopped = update_with_saturation([0., 0.], [0., 0.], [0., 0.], [0., 0.], "conditional")
        self.assertFalse(stopped)  # Zero error is not an intentional pause.
        np.testing.assert_array_equal(unchanged, [0., 0.])

    def test_standard_all_twelve_match_step814_exactly(self):
        for case, expected in self.previous.items():
            actual = self.results[("standard",)+case]
            for a, b in zip(actual[:6], expected):
                np.testing.assert_array_equal(a, b)
            self.assertFalse(np.any(actual[6]))
            values = metrics(actual)
            for key, value in previous_metrics(expected).items():
                self.assertEqual(values[key], value)

    def test_unlimited_both_methods_match_step814_exactly(self):
        for gain in GAINS:
            for model in MODELS:
                for method in METHODS:
                    actual = self.results[method, gain, model, "unlimited"]
                    for a, b in zip(actual[:6], self.previous[gain, model, "unlimited"]):
                        np.testing.assert_array_equal(a, b)
                    self.assertFalse(np.any(actual[6]))

    def test_all_values_finite_and_update_order(self):
        self.assertEqual(len(self.results), 24)
        for (method, gain, model, limit), data in self.results.items():
            time, states, commands, actual, errors, z, stopped = data
            np.testing.assert_array_equal(time, np.arange(8001)*DT)
            self.assertEqual(states.shape, (8001, 4))
            for values in (commands, actual, errors, z):
                self.assertEqual(values.shape, (8001, 2))
            self.assertEqual(stopped.shape, (8001,))
            self.assertEqual(stopped.dtype, np.dtype(bool))
            self.assertFalse(stopped[-1])
            self.assertTrue(np.all(np.isfinite(np.column_stack(data))))
            np.testing.assert_array_equal(states[0], np.zeros(4))
            np.testing.assert_array_equal(z[0], np.zeros(2))
            np.testing.assert_array_equal(errors, TARGET-states[:, :2])
            mask = np.any(saturation_mask(commands, actual), axis=1)
            np.testing.assert_array_equal(stopped[:-1], mask if method == "conditional" else np.zeros(8000, bool))
            np.testing.assert_array_equal(z[1:], np.where(stopped[:-1, None], z[:-1], np.clip(z[:-1]+DT*errors[:-1], -.5, .5)))
            self.assertTrue(np.all(np.abs(z) <= .5))
            if LIMITS[limit] is not None:
                np.testing.assert_array_equal(actual, np.clip(commands, -LIMITS[limit], LIMITS[limit]))
            cp, pp = model_parameters(model)
            g = GAINS[gain]
            indices = {0, 100, 1000, 7999}
            transitions = np.flatnonzero(stopped[:-2] != stopped[1:-1])+1
            indices.update(transitions[:2].tolist())
            for i in indices:
                np.testing.assert_array_equal(commands[i], computed_torque_integral(states[i, :2], states[i, 2:],
                    TARGET, z[i], [g["Kp"]]*2, [g["Kd"]]*2, [g["Ki"]]*2, cp))
                np.testing.assert_array_equal(states[i+1], integrate_dynamics(*states[i], *actual[i], DT, **pp, method="rk2"))

    def test_parameters_and_existing_constants_unchanged(self):
        gains_before, limits_before = copy.deepcopy(GAINS), copy.deepcopy(LIMITS)
        target_before = TARGET.copy()
        nominal, _ = model_parameters("baseline")
        for model in MODELS:
            params = model_parameters(model)
            before = copy.deepcopy(params)
            self.assertIsNot(params[0], params[1])
            with patch("scripts.run_anti_windup_demo.model_parameters", return_value=params):
                simulate("conditional", "B", model, "moderate")
            self.assertEqual(params, before)
            self.assertEqual(params[0], nominal)
            for key in nominal:
                self.assertEqual(params[1][key], nominal[key]*(1.2 if model == "mismatch" and key in ("m2", "I2") else 1.))
        self.assertEqual(GAINS, gains_before)
        for key in LIMITS:
            np.testing.assert_array_equal(LIMITS[key], limits_before[key])
        np.testing.assert_array_equal(TARGET, target_before)

    def test_stop_metrics_and_frozen_steps_not_counted_as_clipped(self):
        for case, data in self.results.items():
            method, _, _, _ = case
            v = metrics(data)
            expected = int(np.count_nonzero(data[6][:-1]))
            self.assertEqual(v["integral_stopped_steps"], expected)
            self.assertEqual(v["integral_stopped_steps_per_joint"], [expected]*2)
            self.assertEqual(v["integral_stopped_fraction"], expected/8000)
            self.assertEqual(expected, v["saturation"]["any_joint_steps"] if method == "conditional" else 0)
            raw = data[5][:-1]+DT*data[4][:-1]
            clipped = (np.abs(raw) > .5+1e-12) & ~data[6][:-1, None]
            np.testing.assert_array_equal(v["integral_limits"]["clipping_update_steps"], clipped.sum(axis=0))
        strict = self.results["conditional", "B", "mismatch", "strict"]
        self.assertEqual(metrics(strict)["integral_stopped_steps"], 8000)
        np.testing.assert_array_equal(strict[5], np.zeros((8001, 2)))

    def test_regression_classification_includes_unsettled(self):
        a = metrics(self.results["standard", "A", "baseline", "unlimited"])
        b = copy.deepcopy(a)
        b["overshoot_percent"] = [a["overshoot_percent"][0]+1., a["overshoot_percent"][1]-1.]
        a["settling_time_s"], b["settling_time_s"] = [5., None], [None, 4.]
        worse = compare_pair(a, b)["conditional_worse"]
        self.assertEqual(worse["overshoot_percent"], [True, False])
        self.assertEqual(worse["settling_time_s"], [True, False])
        self.assertEqual(worse["final_abs_angle_error_rad"], [False, False])

    def test_csv_json_png_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="anti_windup_") as directory:
            output = Path(directory)/"results"
            summary = save_results(self.results, output)
            self.assertEqual(summary, json.loads((output/"summary.json").read_text(encoding="utf-8")))
            self.assertEqual(len(summary["cases"]), 24)
            self.assertEqual(len(summary["comparisons"]), 12)
            for case, data in self.results.items():
                name = case_name(*case)
                csv_data = np.loadtxt(output/(name+".csv"), delimiter=",", skiprows=1)
                self.assertEqual(csv_data.shape, (8001, 14))
                np.testing.assert_array_equal(csv_data, np.column_stack(data))
                self.assertEqual(summary["cases"][name]["static_hold_possible"], case[3] != "strict")
            for filename, expected in (("metrics.csv", 48), ("comparison.csv", 24)):
                with (output/filename).open(newline="", encoding="utf-8") as file:
                    self.assertEqual(len(list(csv.DictReader(file))), expected)
            images = list(output.glob("*.png"))
            self.assertEqual(len(images), 13)
            for path in images:
                self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}
            with self.assertRaises(FileExistsError):
                save_results(self.results, output)
            self.assertEqual(before, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()})

    def test_invalid_cases_and_update_inputs(self):
        for case in (("wrong", "A", "baseline", "strict"), ("standard", "C", "baseline", "strict"),
                     ("standard", "A", "wrong", "strict"), ("standard", "A", "baseline", "wrong")):
            with self.assertRaises(ValueError):
                simulate(*case)
        for command, actual in (([np.nan, 0], [0, 0]), ([1], [1, 0]), ([1, 0], [np.inf, 0])):
            with self.assertRaises(ValueError):
                update_with_saturation([0, 0], [1, 1], command, actual, "conditional")
        with self.assertRaises(ValueError):
            update_with_saturation([0, 0], [1, 1], [1, 1], [0, 0], "wrong")
        with self.assertRaises(ValueError):
            update_with_saturation([0, 0], [1, 1], [1, 1], [0, 0], "conditional", dt=0)


if __name__ == "__main__":
    unittest.main()
