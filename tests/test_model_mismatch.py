"""Step 8-8のモデル分離、既存応答との一致、出力を検証する。"""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.run_coupled_dynamics_demo import physical_parameters
from scripts.run_computed_torque_demo import simulate as simulate_existing
from scripts.run_model_mismatch_demo import (
    CASES, CONTROLLERS, DT, TARGET, CSV_HEADER, model_parameters, simulate, save_results,
)


class TestModelMismatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = {case: simulate(*case) for case in CASES}

    def test_baseline_matches_existing_exactly(self):
        for controller in CONTROLLERS:
            for actual, expected in zip(self.results[controller, "baseline"], simulate_existing(controller)):
                np.testing.assert_array_equal(actual, expected)

    def test_only_plant_mass_and_inertia_change(self):
        nominal = physical_parameters()
        for model in ("baseline", "mismatch"):
            controller, plant = model_parameters(model)
            self.assertIsNot(controller, plant)
            self.assertEqual(controller, nominal)
            self.assertEqual(set(plant), set(nominal))
            for key in nominal:
                factor = 1.2 if model == "mismatch" and key in ("m2", "I2") else 1.
                self.assertEqual(plant[key], nominal[key] * factor)
            plant["m2"] = 999.
            self.assertEqual(controller, nominal)
            self.assertEqual(physical_parameters(), nominal)

    def test_simulation_does_not_mutate_parameters(self):
        for controller in CONTROLLERS:
            params = model_parameters("mismatch")
            before = tuple(p.copy() for p in params)
            with patch("scripts.run_model_mismatch_demo.model_parameters", return_value=params):
                simulate(controller, "mismatch")
            self.assertEqual(params, before)

    def test_finite_histories_shapes_times_and_errors(self):
        for case, data in self.results.items():
            with self.subTest(case=case):
                time, states, torques, errors = data
                self.assertEqual(states.shape, (1601, 4))
                self.assertEqual(torques.shape, (1601, 2))
                self.assertEqual(errors.shape, (1601, 2))
                np.testing.assert_array_equal(time, np.arange(1601) * DT)
                self.assertEqual(time[-1], 4.)
                np.testing.assert_array_equal(states[0], [0., 0., 0., 0.])
                np.testing.assert_array_equal(errors, TARGET - states[:, :2])
                self.assertTrue(np.all(np.isfinite(np.column_stack(data))))

    def test_same_initial_command_but_different_plant_response(self):
        for controller in CONTROLLERS:
            base = self.results[controller, "baseline"]
            mismatch = self.results[controller, "mismatch"]
            np.testing.assert_array_equal(base[2][0], mismatch[2][0])
            self.assertGreater(np.max(np.abs(base[1][1] - mismatch[1][1])), 1e-6)

    def test_exports_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="mismatch_") as directory:
            output = Path(directory) / "results"
            summary = save_results(self.results, output)
            self.assertEqual(summary, json.loads((output / "summary.json").read_text()))
            self.assertEqual(len(list(output.iterdir())), 8)
            for (controller, model), data in self.results.items():
                name = controller + "_" + model
                csv = np.genfromtxt(output / (name + ".csv"), delimiter=",", names=True)
                self.assertEqual(csv.dtype.names, tuple(CSV_HEADER.split(",")))
                np.testing.assert_array_equal(np.column_stack([csv[n] for n in csv.dtype.names]), np.column_stack(data))
                np.testing.assert_array_equal(summary["cases"][name]["final_angle_error_rad"], data[3][-1])
            for controller in CONTROLLERS:
                delta = self.results[controller, "mismatch"][1] - self.results[controller, "baseline"][1]
                np.testing.assert_array_equal(summary["trajectory_differences"][controller]["max_abs_angle_difference_rad"], np.max(np.abs(delta[:, :2]), axis=0))
            for filename in ("angles.png", "errors.png", "torques.png"):
                self.assertEqual((output / filename).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            before = {p.name: p.read_bytes() for p in output.iterdir()}
            with self.assertRaises(FileExistsError):
                save_results(self.results, output)
            self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})

    def test_invalid_cases(self):
        with self.assertRaises(ValueError):
            model_parameters("unknown")
        with self.assertRaises(ValueError):
            simulate("unknown", "baseline")


if __name__ == "__main__":
    unittest.main()
