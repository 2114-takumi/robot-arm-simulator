"""Step 8-16: 関節別停止、更新時刻、Step 8-15の全軌道との完全一致。"""

import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.run_per_joint_anti_windup_demo import (
    CASES, METHODS, GAINS, MODELS, LIMITS, DT, TARGET, SAT_ATOL, SAT_RTOL,
    update_with_saturation, simulate, metrics, model_parameters, compare_methods,
    save_results, case_name, target_equilibrium,
)
from scripts.run_anti_windup_demo import simulate as previous_simulate, CASES as PREVIOUS_CASES
from scripts.run_ctc_torque_saturation_demo import saturation_mask
from robot_arm.computed_torque_integral import computed_torque_integral
from robot_arm.dynamics import integrate_dynamics


class TestPerJointAntiWindup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = {case: simulate(*case) for case in CASES}

    def test_single_joint_both_and_no_saturation(self):
        z, error = np.array([.1, -.2]), np.array([.3, -.4])
        before = z.copy(), error.copy()
        for command, actual, expected in (
            ([21., 0.], [20., 0.], [True, False]),
            ([-21., 0.], [-20., 0.], [True, False]),
            ([0., -5.], [0., -4.], [False, True]),
            ([0., 5.], [0., 4.], [False, True]),
            ([21., -5.], [20., -4.], [True, True]),
            ([20., -4.], [20., -4.], [False, False]),
        ):
            with self.subTest(command=command):
                command, actual = np.array(command), np.array(actual)
                torques_before = command.copy(), actual.copy()
                new, stopped = update_with_saturation(z, error, command, actual, "per_joint")
                np.testing.assert_array_equal(stopped, expected)
                np.testing.assert_array_equal(new, np.where(expected, z, z+DT*error))
                self.assertFalse(np.shares_memory(new, z))
                np.testing.assert_array_equal(command, torques_before[0])
                np.testing.assert_array_equal(actual, torques_before[1])
        np.testing.assert_array_equal(z, before[0])
        np.testing.assert_array_equal(error, before[1])

    def test_same_tolerance_as_global_and_resume_clips(self):
        for difference, expected in ((1e-12, False), (1e-8, True)):
            for method in ("conditional", "per_joint"):
                _, mask = update_with_saturation([0., 0.], [1., 1.], [20., 4.],
                                                  [20.-difference, 4.], method)
                np.testing.assert_array_equal(mask, [expected, expected if method == "conditional" else False])
        z, error = np.array([.499, -.499]), np.array([2., -2.])
        frozen, mask = update_with_saturation(z, error, [21., 0.], [20., 0.], "per_joint")
        np.testing.assert_array_equal(frozen, [.499, -.5])
        np.testing.assert_array_equal(mask, [True, False])
        resumed, mask = update_with_saturation(frozen, error, [19., 0.], [19., 0.], "per_joint")
        np.testing.assert_array_equal(resumed, [.5, -.5])
        self.assertFalse(np.any(mask))
        _, mask = update_with_saturation([0., 0.], [0., 0.], [21., 0.], [20., 0.], "per_joint")
        np.testing.assert_array_equal(mask, [True, False])  # Count intentional freeze even for zero error.

    def test_standard_and_global_all_24_match_step815_exactly(self):
        for case in PREVIOUS_CASES:
            with self.subTest(case=case):
                expected = previous_simulate(*case)
                actual = self.results[case]
                for a, b in zip(actual[:6], expected[:6]):
                    np.testing.assert_array_equal(a, b)
                for j in (0, 1):
                    np.testing.assert_array_equal(actual[6][:, j], expected[6])

    def test_unlimited_all_three_methods_exactly_match(self):
        for gain in GAINS:
            for model in MODELS:
                expected = self.results["standard", gain, model, "unlimited"]
                for method in METHODS:
                    actual = self.results[method, gain, model, "unlimited"]
                    for a, b in zip(actual, expected):
                        np.testing.assert_array_equal(a, b)
                    self.assertFalse(np.any(actual[6]))

    def test_all_values_finite_shapes_limits_and_integral_timing(self):
        self.assertEqual(len(self.results), 36)
        for (method, gain, model, limit), data in self.results.items():
            with self.subTest(method=method, gain=gain, model=model, limit=limit):
                time, states, command, actual, error, z, stopped = data
                np.testing.assert_array_equal(time, np.arange(8001)*DT)
                self.assertEqual(states.shape, (8001, 4))
                for values in (command, actual, error, z, stopped):
                    self.assertEqual(values.shape, (8001, 2))
                self.assertEqual(stopped.dtype, np.dtype(bool))
                self.assertFalse(np.any(stopped[-1]))
                self.assertTrue(np.all(np.isfinite(np.column_stack(data))))
                np.testing.assert_array_equal(states[0], np.zeros(4))
                np.testing.assert_array_equal(z[0], np.zeros(2))
                np.testing.assert_array_equal(error, TARGET-states[:, :2])
                saturated = saturation_mask(command, actual)
                expected = saturated if method == "per_joint" else (
                    np.repeat(np.any(saturated, axis=1)[:, None], 2, axis=1)
                    if method == "conditional" else np.zeros_like(saturated))
                np.testing.assert_array_equal(stopped[:-1], expected)
                np.testing.assert_array_equal(z[1:], np.where(stopped[:-1], z[:-1],
                                                            np.clip(z[:-1]+DT*error[:-1], -.5, .5)))
                self.assertTrue(np.all(np.abs(z) <= .5))
                if LIMITS[limit] is None:
                    np.testing.assert_array_equal(actual, command)
                else:
                    np.testing.assert_array_equal(actual, np.clip(command, -LIMITS[limit], LIMITS[limit]))

    def test_torque_uses_current_integral_and_actual_held_through_rk2(self):
        for case, data in self.results.items():
            _, gain, model, _ = case
            time, states, command, actual, _, z, stopped = data
            cp, pp = model_parameters(model)
            g = GAINS[gain]
            indices = {0, 100, 1000, len(time)-2}
            transitions = np.flatnonzero(np.any(stopped[:-2] != stopped[1:-1], axis=1))+1
            indices.update(transitions[:4].tolist())
            for i in indices:
                np.testing.assert_array_equal(command[i], computed_torque_integral(
                    states[i, :2], states[i, 2:], TARGET, z[i],
                    [g["Kp"]]*2, [g["Kd"]]*2, [g["Ki"]]*2, cp))
                np.testing.assert_array_equal(states[i+1], integrate_dynamics(
                    *states[i], *actual[i], DT, **pp, b1=0., b2=0., g=9.81, method="rk2"))

    def test_parameters_and_shared_constants_unchanged(self):
        gains_before, limits_before, target_before = copy.deepcopy(GAINS), copy.deepcopy(LIMITS), TARGET.copy()
        nominal, _ = model_parameters("baseline")
        for model in MODELS:
            params = model_parameters(model)
            before = copy.deepcopy(params)
            self.assertIsNot(params[0], params[1])
            with patch("scripts.run_per_joint_anti_windup_demo.model_parameters", return_value=params):
                simulate("per_joint", "B", model, "moderate")
            self.assertEqual(params, before)
            self.assertEqual(params[0], nominal)
            for key in nominal:
                self.assertEqual(params[1][key], nominal[key]*(1.2 if model == "mismatch" and key in ("m2", "I2") else 1.))
        self.assertEqual(GAINS, gains_before)
        for key in LIMITS:
            np.testing.assert_array_equal(LIMITS[key], limits_before[key])
        np.testing.assert_array_equal(TARGET, target_before)

    def test_metrics_count_joints_separately_and_exclude_terminal_row(self):
        for case, data in self.results.items():
            values = metrics(data)
            mask = data[6][:-1]
            np.testing.assert_array_equal(values["integral_stopped_steps_per_joint"], mask.sum(axis=0))
            np.testing.assert_array_equal(values["integral_stopped_fraction_per_joint"], mask.mean(axis=0))
            self.assertEqual(values["integral_stopped_any_joint_steps"], np.any(mask, axis=1).sum())
            self.assertEqual(values["integral_stopped_both_joints_steps"], np.all(mask, axis=1).sum())
            clipped = (np.abs(data[5][:-1]+DT*data[4][:-1]) > .5+1e-12) & ~mask
            np.testing.assert_array_equal(values["integral_limits"]["clipping_update_steps"], clipped.sum(axis=0))
            if case[0] == "per_joint":
                self.assertEqual(values["integral_stopped_steps_per_joint"], values["saturation"]["per_joint_steps"])
        # A hand-built 4-step record checks per-joint denominators and longest runs.
        source = self.results["per_joint", "B", "baseline", "unlimited"]
        data = tuple(a[:5].copy() for a in source)
        data[2][:] = 0.
        data[3][:] = 0.
        data[2][[0, 1, 4], 0] = 1.
        data[2][[1, 3], 1] = 1.
        data[6][:] = False
        data[6][0:2, 0] = True
        data[6][[1, 3], 1] = True
        v = metrics(data)
        self.assertEqual(v["saturation"]["per_joint_steps"], [2, 2])
        self.assertEqual(v["saturation"]["per_joint_fraction"], [.5, .5])
        self.assertEqual(v["saturation"]["any_joint_fraction"], .75)
        self.assertEqual(v["saturation"]["max_continuous_time_s"], [2*DT, DT])
        self.assertEqual(v["integral_stopped_steps_per_joint"], [2, 2])

    def test_static_holding_feasibility_uses_plant(self):
        for model in MODELS:
            equilibrium = target_equilibrium(model)
            gravity = np.array(equilibrium["required_gravity_torque_Nm"])
            self.assertTrue(np.all(np.abs(gravity) < LIMITS["moderate"]))
            self.assertTrue(np.all(np.abs(gravity) > LIMITS["strict"]))
            self.assertTrue(equilibrium["limits"]["moderate"]["static_hold_possible"])
            self.assertFalse(equilibrium["limits"]["strict"]["static_hold_possible"])

    def test_comparison_reports_regressions_and_unsettled(self):
        a = metrics(self.results["standard", "A", "baseline", "unlimited"])
        b = copy.deepcopy(a)
        a["settling_time_s"], b["settling_time_s"] = [5., None], [None, 4.]
        b["overshoot_percent"][0] += 1.
        increased = compare_methods(a, b)["candidate_increased"]
        self.assertEqual(increased["settling_time_s"], [True, False])
        self.assertEqual(increased["overshoot_percent"], [True, False])
        self.assertEqual(increased["final_abs_angle_error_rad"], [False, False])

    def test_csv_json_png_and_overwrite_refusal(self):
        with tempfile.TemporaryDirectory(prefix="per_joint_aw_") as directory:
            output = Path(directory)/"results"
            summary = save_results(self.results, output)
            self.assertEqual(summary, json.loads((output/"summary.json").read_text(encoding="utf-8")))
            self.assertEqual(len(summary["cases"]), 36)
            self.assertEqual(len(summary["comparisons"]), 24)
            for case, data in self.results.items():
                csv_data = np.loadtxt(output/(case_name(*case)+".csv"), delimiter=",", skiprows=1)
                self.assertEqual(csv_data.shape, (8001, 15))
                np.testing.assert_array_equal(csv_data, np.column_stack(data))
                self.assertEqual(summary["cases"][case_name(*case)]["static_hold_possible"], case[3] != "strict")
            for filename, expected in (("metrics.csv", 72), ("comparison.csv", 48)):
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
            empty = Path(directory)/"empty"
            empty.mkdir()
            with self.assertRaises(FileExistsError):
                save_results(self.results, empty)
            self.assertEqual(list(empty.iterdir()), [])

    def test_invalid_inputs(self):
        for case in (("wrong", "A", "baseline", "strict"), ("per_joint", "C", "baseline", "strict"),
                     ("per_joint", "A", "wrong", "strict"), ("per_joint", "A", "baseline", "wrong")):
            with self.assertRaises(ValueError):
                simulate(*case)
        for command, actual in (([np.nan, 0.], [0., 0.]), ([1.], [1., 0.]), ([1., 0.], [np.inf, 0.])):
            with self.assertRaises(ValueError):
                update_with_saturation([0., 0.], [1., 1.], command, actual, "per_joint")
        for method, dt in (("wrong", DT), ("per_joint", 0.)):
            with self.assertRaises(ValueError):
                update_with_saturation([0., 0.], [1., 1.], [1., 1.], [0., 0.], method, dt)
        self.assertEqual(SAT_ATOL, 1e-12)
        self.assertEqual(SAT_RTOL, 1e-12)


if __name__ == "__main__":
    unittest.main()
