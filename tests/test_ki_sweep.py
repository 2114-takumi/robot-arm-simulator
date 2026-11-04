"""Ki掃引の再現性、制御タイミング、評価指標、出力保護のテスト。"""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.run_ctc_integral_demo import simulate as step89
from scripts.run_ki_sweep_demo import (
    CASES, DT, TARGET, model_parameters, simulate, metrics, overshoot_percent,
    settling_times, save_results, CSV_HEADER,
)


class TestKiSweep(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = {case: simulate(*case) for case in CASES}

    def test_zero_ki_matches_ctc(self):
        for model in ("baseline", "mismatch"):
            for actual, expected in zip(self.results[model, 0.][:4], step89("CTC", model)[:4]):
                np.testing.assert_array_equal(actual[:len(expected)], expected)

    def test_ki10_matches_step89(self):
        for model in ("baseline", "mismatch"):
            for actual, expected in zip(self.results[model, 10.], step89("CTC_I", model)):
                np.testing.assert_array_equal(actual[:len(expected)], expected)

    def test_parameters_unchanged(self):
        for model in ("baseline", "mismatch"):
            params = model_parameters(model)
            before = tuple(p.copy() for p in params)
            self.assertIsNot(params[0], params[1])
            with patch("scripts.run_ki_sweep_demo.model_parameters", return_value=params):
                simulate(model, 3.)
            self.assertEqual(params, before)
            nominal, _ = model_parameters("baseline")
            self.assertEqual(params[0], nominal)
            changed = {k for k in nominal if params[1][k] != nominal[k]}
            self.assertEqual(changed, {"m2", "I2"} if model == "mismatch" else set())

    def test_all_histories_finite_and_integral_timing(self):
        for case, data in self.results.items():
            with self.subTest(case=case):
                time, states, torques, errors, integrals = data
                self.assertEqual(states.shape, (8001, 4))
                for values in (torques, errors, integrals):
                    self.assertEqual(values.shape, (8001, 2))
                np.testing.assert_array_equal(time, np.arange(8001)*DT)
                np.testing.assert_array_equal(states[0], np.zeros(4))
                np.testing.assert_array_equal(integrals[0], [0, 0])
                np.testing.assert_array_equal(errors, TARGET-states[:, :2])
                self.assertTrue(np.all(np.isfinite(np.column_stack(data))))
                self.assertTrue(np.all(np.abs(integrals) <= 0.5))
                np.testing.assert_array_equal(integrals[1:], np.clip(integrals[:-1]+DT*errors[:-1], -0.5, 0.5))

    def test_settling_requires_remaining_samples_in_band(self):
        time = np.array([0., 5., 10., 15., 20.])
        errors = np.array([[0.3, -0.4], [0.005, 0.007], [0.01, 0.009], [0.006, 0.008], [0., 0.009]])
        settled, thresholds = settling_times(time, errors)
        self.assertEqual(settled, [15., None])
        np.testing.assert_array_equal(thresholds, [0.006, 0.008])

    def test_negative_direction_overshoot(self):
        angles = np.array([[0, 0], [0.33, -0.48], [0.3, -0.4]])
        np.testing.assert_allclose(overshoot_percent(angles), [10, 20])
        np.testing.assert_array_equal(overshoot_percent(np.array([[0, 0], [0.2, -0.3]])), [0, 0])

    def test_tail_metrics(self):
        data = self.results["mismatch", 3.]
        values = metrics(data)
        tail = data[3][7200:]
        self.assertEqual(len(tail), 801)
        np.testing.assert_array_equal(values["last_2s_error_change_rad"], tail[-1]-tail[0])
        np.testing.assert_array_equal(values["last_2s_mean_abs_error_rad"], np.mean(np.abs(tail), axis=0))

    def test_export_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="ki_sweep_") as directory:
            output = Path(directory) / "results"
            summary = save_results(self.results, output)
            self.assertEqual(summary, json.loads((output / "summary.json").read_text()))
            self.assertEqual(len(list(output.iterdir())), 24)
            for (model, ki), data in self.results.items():
                csv = np.genfromtxt(output / f"{model}_Ki_{ki:g}.csv", delimiter=",", names=True)
                self.assertEqual(csv.dtype.names, tuple(CSV_HEADER.split(",")))
                np.testing.assert_array_equal(np.column_stack([csv[n] for n in csv.dtype.names]), np.column_stack(data))
            for name in ("angles", "final_errors", "settling_times", "overshoot"):
                self.assertEqual((output / (name+".png")).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            before = {p.name: p.read_bytes() for p in output.iterdir()}
            with self.assertRaises(FileExistsError):
                save_results(self.results, output)
            self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})

    def test_invalid_ki(self):
        for ki in (-1, np.inf, np.nan, [1, 1]):
            with self.assertRaises(ValueError):
                simulate("baseline", ki)


if __name__ == "__main__":
    unittest.main()
