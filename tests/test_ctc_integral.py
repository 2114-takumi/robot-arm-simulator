"""CTC+Iの状態分離、積分制限、既存CTCとの一致と保存処理を検証。"""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from robot_arm.computed_torque import computed_torque
from robot_arm.computed_torque_integral import computed_torque_integral
from robot_arm.dynamics import forward_dynamics, integrate_dynamics
from scripts.run_computed_torque_demo import simulate as original_simulate
from scripts.run_model_mismatch_demo import model_parameters, simulate as mismatch_simulate
from scripts.run_ctc_integral_demo import (
    CASES, TARGET, DT, CTC_KP, CTC_KD, KI, CSV_HEADER,
    update_integral, simulate, response_metrics, save_results,
)


class TestIntegralController(unittest.TestCase):
    def test_zero_ki_matches_existing_torque(self):
        p, _ = model_parameters("baseline")
        rng = np.random.default_rng(89)
        for _ in range(50):
            q, omega, target, integral = rng.uniform(-1, 1, (4, 2))
            actual = computed_torque_integral(q, omega, target, integral, CTC_KP, CTC_KD, [0, 0], p)
            np.testing.assert_array_equal(actual, computed_torque(q, omega, target, CTC_KP, CTC_KD, p))

    def test_integral_acceleration_and_no_mutation(self):
        p, _ = model_parameters("baseline")
        before = p.copy()
        q, omega, integral = np.array([0.1, -0.2]), np.array([0.2, -0.3]), np.array([0.04, -0.05])
        copies = [v.copy() for v in (q, omega, integral)]
        tau = computed_torque_integral(q, omega, TARGET, integral, CTC_KP, CTC_KD, KI, p)
        np.testing.assert_allclose(forward_dynamics(*q, *omega, *tau, **p), CTC_KP * (TARGET-q) + KI*integral - CTC_KD*omega, atol=1e-12)
        self.assertEqual(p, before)
        for actual, expected in zip((q, omega, integral), copies):
            np.testing.assert_array_equal(actual, expected)

    def test_update_clips_both_signs_and_uses_left_error(self):
        z, e = np.array([0.49, -0.49]), np.array([10., -10.])
        np.testing.assert_array_equal(update_integral(z, e, DT), [0.5, -0.5])
        np.testing.assert_array_equal(z, [0.49, -0.49])
        np.testing.assert_allclose(update_integral([0.5, -0.5], [-1, 1], DT), [0.4975, -0.4975])

    def test_invalid_inputs(self):
        p, _ = model_parameters("baseline")
        for ki in ([-1, 1], [np.nan, 1], [1]):
            with self.assertRaises(ValueError):
                computed_torque_integral([0, 0], [0, 0], TARGET, [0, 0], CTC_KP, CTC_KD, ki, p)
        for dt in (0, -1, np.inf):
            with self.assertRaises(ValueError):
                update_integral([0, 0], [1, 1], dt)


class TestIntegralDemo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = {case: simulate(*case) for case in CASES}

    def test_existing_baseline_and_mismatch_prefix_match(self):
        for model in ("baseline", "mismatch"):
            previous = original_simulate("computed_torque") if model == "baseline" else mismatch_simulate("computed_torque", model)
            for actual, expected in zip(self.results["CTC", model][:4], previous):
                np.testing.assert_array_equal(actual[:1601], expected)

    def test_zero_ki_entire_trajectory_matches(self):
        with patch("scripts.run_ctc_integral_demo.KI", np.zeros(2)):
            data = simulate("CTC_I", "baseline")
        for actual, expected in zip(data[:4], self.results["CTC", "baseline"][:4]):
            np.testing.assert_array_equal(actual, expected)

    def test_parameter_dictionaries_unchanged(self):
        for controller, model in CASES:
            params = model_parameters(model)
            before = tuple(p.copy() for p in params)
            with patch("scripts.run_ctc_integral_demo.model_parameters", return_value=params):
                simulate(controller, model)
            self.assertEqual(params, before)
            self.assertIsNot(params[0], params[1])

    def test_histories_and_integral_timing(self):
        for (controller, model), data in self.results.items():
            time, states, torques, errors, integral = data
            self.assertEqual(states.shape, (3201, 4))
            for values in (torques, errors, integral):
                self.assertEqual(values.shape, (3201, 2))
            np.testing.assert_array_equal(time, np.arange(3201)*DT)
            np.testing.assert_array_equal(states[0], np.zeros(4))
            np.testing.assert_array_equal(integral[0], [0, 0])
            np.testing.assert_array_equal(errors, TARGET - states[:, :2])
            self.assertTrue(np.all(np.isfinite(np.column_stack(data))))
            self.assertTrue(np.all(np.abs(integral) <= 0.5))
            if controller == "CTC_I":
                np.testing.assert_array_equal(integral[1:], np.clip(integral[:-1] + DT*errors[:-1], -0.5, 0.5))
            else:
                np.testing.assert_array_equal(integral, np.zeros_like(integral))
            cp, pp = model_parameters(model)
            for i in (0, 100, 3199):
                if controller == "CTC_I":
                    np.testing.assert_array_equal(torques[i], computed_torque_integral(states[i,:2], states[i,2:], TARGET, integral[i], CTC_KP, CTC_KD, KI, cp))
                np.testing.assert_array_equal(states[i+1], integrate_dynamics(*states[i], *torques[i], DT, **pp, b1=0., b2=0., g=9.81, method="rk2"))

    def test_metrics_and_mismatch_improvement(self):
        baseline = response_metrics(self.results["CTC", "baseline"])
        augmented = response_metrics(self.results["CTC_I", "baseline"])
        self.assertTrue(np.all(np.array(augmented["overshoot_rad"]) > baseline["overshoot_rad"]))
        original = self.results["CTC", "mismatch"][3][-1]
        improved = self.results["CTC_I", "mismatch"][3][-1]
        self.assertTrue(np.all(np.abs(improved) < np.abs(original)))

    def test_exports_no_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="ctc_integral_") as directory:
            output = Path(directory) / "results"
            summary = save_results(self.results, output)
            self.assertEqual(summary, json.loads((output / "summary.json").read_text()))
            self.assertEqual(len(list(output.iterdir())), 9)
            for case, data in self.results.items():
                csv = np.genfromtxt(output / ("_".join(case)+".csv"), delimiter=",", names=True)
                self.assertEqual(csv.dtype.names, tuple(CSV_HEADER.split(",")))
                np.testing.assert_array_equal(np.column_stack([csv[n] for n in csv.dtype.names]), np.column_stack(data))
            for name in ("angles", "errors", "torques", "integrals"):
                self.assertEqual((output / (name+".png")).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            before = {p.name: p.read_bytes() for p in output.iterdir()}
            with self.assertRaises(FileExistsError):
                save_results(self.results, output)
            self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})


if __name__ == "__main__":
    unittest.main()
