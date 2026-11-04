"""関節別Kawの指定演算順序、全時刻一致、選定・出力・既存モデルを検証。"""

import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import run_per_joint_back_calculation_demo as demo
from robot_arm.computed_torque_integral import computed_torque_integral
from robot_arm.dynamics import mass_matrix, integrate_dynamics


class TestPerJointBackCalculation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results, cls.selection, cls.diagonal = demo.run_experiment()

    def test_required_division_first_and_solve_no_inverse(self):
        cp, _ = demo.model_parameters("baseline")
        dv, ki, kaw = np.array([.1, .7]), np.array([7., 10.]), np.array([5., 10.])
        with patch.object(demo.np.linalg, "solve", return_value=dv) as solve, \
             patch.object(demo.np.linalg, "inv", side_effect=AssertionError("No inverse")):
            _, _, correction = demo.back_calculation_update([.2, -.3], [0., 0.], [0., 0.],
                                                            [2., -3.], [1., -2.], ki, kaw, cp)
            solve.assert_called_once()
            np.testing.assert_array_equal(solve.call_args.args[0], mass_matrix(.2, -.3, **cp))
            np.testing.assert_array_equal(solve.call_args.args[1], [-1., 1.])
        np.testing.assert_array_equal(correction, kaw*(dv/ki))
        self.assertFalse(np.array_equal(correction, (kaw*dv)/ki))

    def test_positive_negative_saturation_couples_both_components(self):
        cp, _ = demo.model_parameters("baseline")
        ki, kaw = np.array([7., 10.]), np.array([1., 5.])
        for command, actual, signs in (
            ([21., 0.], [20., 0.], [-1., 1.]), ([-21., 0.], [-20., 0.], [1., -1.]),
            ([0., 5.], [0., 4.], [1., -1.]), ([0., -5.], [0., -4.], [-1., 1.]),
        ):
            z, delta, correction = demo.back_calculation_update([0., 0.], [0., 0.], [0., 0.], command, actual, ki, kaw, cp)
            np.testing.assert_array_equal(np.sign(correction), signs)
            np.testing.assert_allclose(mass_matrix(0., 0., **cp) @ (correction/kaw*ki), delta, atol=1e-14)
            np.testing.assert_array_equal(z, np.clip(demo.DT*correction, -.5, .5))

    def test_diagonal_matches_step817_all_60_conditions_with_explicit_roundoff(self):
        maximum = dict(state=0., command=0., integral=0., correction=0.)
        bit_equal = 0
        for kaw in demo.KAW_VALUES:
            for gain, model in demo.GROUPS:
                for limit in demo.LIMITS:
                    case = f"per_joint_{kaw:g}_{kaw:g}", gain, model, limit
                    data = self.results[case]["data"] if case in self.results else demo.simulate([kaw, kaw], gain, model, limit)
                    # Fresh old implementation, independent of optional untracked output artifacts.
                    expected = demo.previous.simulate(kaw, gain, model, limit)
                    compared = demo.trajectory_comparison(data, expected)
                    bit_equal += compared["bitwise_equal"]
                    for key in maximum:
                        maximum[key] = max(maximum[key], compared["max_abs_difference"][key])
                    with self.subTest(case=case, differences=compared["max_abs_difference"]):
                        self.assertTrue(compared["within_tolerance"])
        print("\nStep 8-18 diagonal comparison (60 cases):", maximum, "bit-equal:", bit_equal, flush=True)

    def test_zero_vector_matches_standard_all_twelve_exactly(self):
        for gain, model in demo.GROUPS:
            for limit in demo.LIMITS:
                data = demo.simulate([0., 0.], gain, model, limit)
                expected = self.results["standard", gain, model, limit]["data"]
                for a, b in zip(data, expected):
                    np.testing.assert_array_equal(a, b)

    def test_unlimited_all_25_vectors_zero_correction_and_same_trajectory(self):
        expected = self.results["standard", "B", "mismatch", "unlimited"]["data"]
        for variant, kaw in demo.VECTORS.items():
            case = variant, "B", "mismatch", "unlimited"
            data = self.results[case]["data"] if case in self.results else demo.simulate(kaw, "B", "mismatch", "unlimited")
            for a, b in zip(data, expected):
                np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(data[8], np.zeros((8001, 2)))

    def test_all_reference_trajectories_match_saved_step817_if_present(self):
        for variant in demo.REFERENCES:
            for gain, model in demo.GROUPS:
                for limit in demo.LIMITS:
                    case = variant, gain, model, limit
                    path = demo.ROOT/"outputs"/"step_8_17"/(demo.case_name(case)+".csv")
                    if path.exists():
                        np.testing.assert_array_equal(np.column_stack(self.results[case]["data"]),
                                                      np.loadtxt(path, delimiter=",", skiprows=1))
                    else:
                        expected = demo.previous.run_case(case)["data"]
                        for a, b in zip(self.results[case]["data"], expected):
                            np.testing.assert_array_equal(a, b)

    def test_all_finite_shapes_limits_update_time_and_held_actual_torque(self):
        self.assertEqual(len(self.results), 25+11*len(self.selection["selected"])+60)
        for case, record in self.results.items():
            self.assertEqual(record["status"], "ok")
            time, state, command, actual, error, z, stopped, delta, correction = record["data"]
            np.testing.assert_array_equal(time, np.arange(8001)*demo.DT)
            self.assertEqual(state.shape, (8001, 4))
            for a in (command, actual, error, z, stopped, delta, correction):
                self.assertEqual(a.shape, (8001, 2))
            self.assertTrue(np.all(np.isfinite(np.column_stack(record["data"]))))
            np.testing.assert_array_equal(z[0], [0., 0.])
            np.testing.assert_array_equal(state[0], np.zeros(4))
            self.assertTrue(np.all(np.abs(z) <= .5))
            np.testing.assert_array_equal(delta, actual-command)
            np.testing.assert_array_equal(error, demo.TARGET-state[:, :2])
            np.testing.assert_array_equal(z[1:], np.where(stopped[:-1], z[:-1],
                np.clip(z[:-1]+demo.DT*(error[:-1]+correction[:-1]), -.5, .5)))
            if demo.LIMITS[case[3]] is not None:
                limit = demo.LIMITS[case[3]]
                np.testing.assert_array_equal(actual, np.clip(command, -limit, limit))
            cp, pp = demo.model_parameters(case[2])
            g = demo.GAINS[case[1]]
            for i in (0, 1, 100, 7999):
                np.testing.assert_array_equal(command[i], computed_torque_integral(state[i, :2], state[i, 2:],
                    demo.TARGET, z[i], [g["Kp"]]*2, [g["Kd"]]*2, [g["Ki"]]*2, cp))
                np.testing.assert_array_equal(state[i+1], integrate_dynamics(*state[i], *actual[i], demo.DT,
                    **pp, b1=0., b2=0., g=9.81, method="rk2"))
                if case[0] in demo.VECTORS:
                    expected = demo.VECTORS[case[0]]*(np.linalg.solve(mass_matrix(*state[i, :2], **cp), delta[i])/g["Ki"])
                    np.testing.assert_array_equal(correction[i], expected)

    def test_clipping_and_parameter_nonmutation(self):
        cp, _ = demo.model_parameters("baseline")
        arrays = [np.array(a) for a in ([.2, -.3], [.499, -.499], [1000., -1000.], [1., 2.], [1., 2.], [7., 10.], [.5, 10.])]
        before = [a.copy() for a in arrays]
        z, _, _ = demo.back_calculation_update(*arrays, cp)
        np.testing.assert_array_equal(z, [.5, -.5])
        for a, b in zip(arrays, before):
            np.testing.assert_array_equal(a, b)
        constants = copy.deepcopy((demo.GAINS, demo.LIMITS, demo.VECTORS, demo.TARGET))
        for model in demo.MODELS:
            params = demo.model_parameters(model)
            original = copy.deepcopy(params)
            self.assertIsNot(params[0], params[1])
            with patch.object(demo, "model_parameters", return_value=params), patch.object(demo, "DURATION", .01):
                demo.simulate([.5, 10.], "B", model, "moderate")
            self.assertEqual(params, original)
            self.assertEqual(params[0], cp)
            for key in cp:
                self.assertEqual(params[1][key], cp[key]*(1.2 if model == "mismatch" and key in ("m2", "I2") else 1.))
        self.assertEqual(demo.GAINS, constants[0])
        for mapping, expected in ((demo.LIMITS, constants[1]), (demo.VECTORS, constants[2])):
            for key in mapping:
                np.testing.assert_array_equal(mapping[key], expected[key])
        np.testing.assert_array_equal(demo.TARGET, constants[3])

    def test_goal_boundaries_pareto_and_candidate_selection(self):
        v = dict(settling_time_s=[4., 10.], overshoot_percent=[100., 19.999])
        self.assertTrue(demo.meets_goal(v))
        for t1, t2, os in ((4.0025, 10., 19.), (4., 10.0025, 19.), (4., 10., 20.), (None, 2., 1.)):
            self.assertFalse(demo.meets_goal(dict(settling_time_s=[t1, t2], overshoot_percent=[0., os])))
        values = {"a": dict(settling_time_s=[1., 4.], overshoot_percent=[0., 10.]),
                  "b": dict(settling_time_s=[2., 3.], overshoot_percent=[0., 8.]),
                  "c": dict(settling_time_s=[3., 5.], overshoot_percent=[0., 11.])}
        self.assertEqual(demo.pareto_candidates(values), ["a", "b"])
        self.assertEqual(len(set(self.selection["selected"])), 4)
        achieved = [v for v in demo.VECTORS if demo.meets_goal(demo.metrics(self.results[v, *demo.FOCUS]["data"]))]
        self.assertEqual(self.selection["goal_achieving"], achieved)
        again = demo.select_candidates({v: self.results[v, *demo.FOCUS] for v in demo.VECTORS})
        self.assertEqual(again, self.selection)

    def test_failure_is_visible_and_invalid_input_rejected(self):
        with patch.object(demo, "DURATION", .005), patch.object(demo, "computed_torque_integral", return_value=np.array([np.inf, 0.])):
            result = demo.run_case(("per_joint_1_5", *demo.FOCUS))
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failed_step"], 0)
        self.assertIn("ValueError", result["reason"])
        cp, _ = demo.model_parameters("baseline")
        for ki, kaw in (([0., 1.], [1., 1.]), ([1., 1.], [-1., 1.]), ([1., 1.], 1.),
                        ([1., 1.], [1., np.nan]), ([1.], [1., 1.])):
            with self.assertRaises(ValueError):
                demo.back_calculation_update([0., 0.], [0., 0.], [1., 1.], [2., 2.], [1., 1.], ki, kaw, cp)

    def test_csv_summary_plots_and_overwrite_refusal(self):
        with tempfile.TemporaryDirectory(prefix="per_joint_bc_") as directory:
            output = Path(directory)/"results"
            summary = demo.save_results(self.results, self.selection, self.diagonal, output)
            self.assertEqual(summary, json.loads((output/"summary.json").read_text(encoding="utf-8")))
            self.assertEqual(len(summary["cases"]), len(self.results))
            for case, record in self.results.items():
                data = np.loadtxt(output/(demo.case_name(case)+".csv"), delimiter=",", skiprows=1)
                self.assertEqual(data.shape, (8001, 19))
                np.testing.assert_array_equal(data, np.column_stack(record["data"]))
                self.assertEqual(summary["cases"][demo.case_name(case)]["static_hold_possible"], case[3] != "strict")
            for filename, count in (("metrics.csv", 2*len(self.results)), ("focus_comparison.csv", 60),
                                    ("comparison.csv", 2*12*5*len(self.selection["selected"]))):
                with (output/filename).open(newline="", encoding="utf-8") as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), count)
            images = list(output.glob("*.png"))
            self.assertEqual(len(images), 22)
            for path in images:
                self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}
            with self.assertRaises(FileExistsError):
                demo.save_results(self.results, self.selection, self.diagonal, output)
            self.assertEqual(before, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()})
            empty = Path(directory)/"empty"
            empty.mkdir()
            with self.assertRaises(FileExistsError):
                demo.save_results(self.results, self.selection, self.diagonal, empty)


if __name__ == "__main__":
    unittest.main()
