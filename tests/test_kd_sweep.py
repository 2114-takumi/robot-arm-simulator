"""Kd掃引の既存応答再現、指標と出力保護を検証する。"""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.run_ki_sweep_demo import simulate as ki_simulate, overshoot_percent
from scripts.run_kd_sweep_demo import (
    CASES, DT, TARGET, KI, CTC_KP, model_parameters, simulate, metrics,
    settling_times, save_results, CSV_HEADER, meets_joint2_goal,
)
from robot_arm.computed_torque_integral import computed_torque_integral
from robot_arm.dynamics import integrate_dynamics


class TestKdSweep(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = {case: simulate(*case) for case in CASES}

    def test_kd10_matches_step810_exactly(self):
        for model in ("baseline", "mismatch"):
            for actual, expected in zip(self.results[model, 10.], ki_simulate(model, 7.)):
                np.testing.assert_array_equal(actual, expected)

    def test_parameters_unchanged(self):
        for model in ("baseline", "mismatch"):
            params = model_parameters(model)
            before = tuple(p.copy() for p in params)
            self.assertIsNot(params[0], params[1])
            with patch("scripts.run_kd_sweep_demo.model_parameters", return_value=params):
                simulate(model, 12.)
            self.assertEqual(params, before)
            nominal, _ = model_parameters("baseline")
            self.assertEqual(params[0], nominal)
            self.assertEqual({k for k in nominal if params[1][k] != nominal[k]}, {"m2", "I2"} if model == "mismatch" else set())

    def test_finite_histories_and_control_timing(self):
        for (model, kd), data in self.results.items():
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
            cp, pp = model_parameters(model)
            for i in (0, 400, 7999):
                np.testing.assert_array_equal(torques[i], computed_torque_integral(states[i,:2], states[i,2:], TARGET, integrals[i], CTC_KP, [kd,kd], KI, cp))
                np.testing.assert_array_equal(states[i+1], integrate_dynamics(*states[i], *torques[i], DT, **pp, method="rk2"))

    def test_settling_reentry_and_negative_overshoot(self):
        time = np.array([0., 5., 10., 15., 20.])
        errors = np.array([[0.3, -0.4], [0.005, 0.007], [0.01, 0.009], [0.006, 0.008], [0., 0.009]])
        self.assertEqual(settling_times(time, errors)[0], [15., None])
        np.testing.assert_allclose(overshoot_percent(np.array([[0,0], [0.33,-0.48]])), [10,20])

    def test_performance_goal_boundaries(self):
        for overshoot, settled, expected in ((19.999,10.,True), (20.,9.,False), (19.,10.001,False), (19.,None,False)):
            self.assertEqual(meets_joint2_goal(dict(overshoot_percent=[0,overshoot], settling_time_s=[0,settled])), expected)

    def test_tail_and_absolute_error_metrics(self):
        data = self.results["mismatch", 12.]
        values = metrics(data)
        np.testing.assert_array_equal(values["final_abs_angle_error_rad"], np.abs(data[3][-1]))
        tail = data[3][7200:]
        self.assertEqual(len(tail), 801)
        np.testing.assert_array_equal(values["last_2s_error_change_rad"], tail[-1]-tail[0])
        np.testing.assert_array_equal(values["last_2s_mean_abs_error_rad"], np.mean(np.abs(tail),axis=0))

    def test_exports_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="kd_sweep_") as directory:
            output = Path(directory) / "results"
            summary = save_results(self.results, output)
            self.assertEqual(summary, json.loads((output / "summary.json").read_text()))
            self.assertEqual(len(list(output.iterdir())), 20)
            for (model, kd), data in self.results.items():
                csv = np.genfromtxt(output / f"{model}_Kd_{kd:g}.csv", delimiter=",", names=True)
                self.assertEqual(csv.dtype.names, tuple(CSV_HEADER.split(",")))
                np.testing.assert_array_equal(np.column_stack([csv[n] for n in csv.dtype.names]), np.column_stack(data))
            for name in ("angles", "final_errors", "settling_times", "overshoot"):
                self.assertEqual((output / (name+".png")).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            before = {p.name:p.read_bytes() for p in output.iterdir()}
            with self.assertRaises(FileExistsError):
                save_results(self.results, output)
            self.assertEqual(before, {p.name:p.read_bytes() for p in output.iterdir()})

    def test_invalid_kd(self):
        for kd in (0, -1, np.inf, np.nan, [10, 10]):
            with self.assertRaises(ValueError):
                simulate("baseline", kd)


if __name__ == "__main__":
    unittest.main()
