"""Kp・Ki掃引の再現性、モデル分離、評価、Pareto判定、出力を検証。"""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.run_ki_sweep_demo import simulate as previous_simulate, overshoot_percent, settling_times
from scripts.run_kp_ki_sweep_demo import (
    CASES, DT, TARGET, KD, OBJECTIVES, model_parameters, simulate, metrics,
    meets_joint2_goal, pareto_indices, save_results,
)
from robot_arm.computed_torque_integral import computed_torque_integral
from robot_arm.dynamics import integrate_dynamics


class TestKpKiSweep(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = {case: simulate(*case) for case in CASES}

    def test_reference_matches_step810_exactly(self):
        for model in ("baseline","mismatch"):
            for actual, expected in zip(self.results[model,25.,7.],previous_simulate(model,7.)):
                np.testing.assert_array_equal(actual,expected)

    def test_parameters_unchanged(self):
        for model in ("baseline","mismatch"):
            params = model_parameters(model)
            before = tuple(p.copy() for p in params)
            self.assertIsNot(params[0],params[1])
            with patch("scripts.run_kp_ki_sweep_demo.model_parameters",return_value=params):
                simulate(model,30.,5.)
            self.assertEqual(params,before)
            nominal,_ = model_parameters("baseline")
            self.assertEqual(params[0],nominal)
            for key in nominal:
                self.assertEqual(params[1][key], nominal[key]*(1.2 if model=="mismatch" and key in ("m2","I2") else 1.))

    def test_all_finite_and_control_timing(self):
        for (model,kp,ki),data in self.results.items():
            time,states,torques,errors,integrals = data
            self.assertEqual(states.shape,(8001,4))
            for values in (torques,errors,integrals):
                self.assertEqual(values.shape,(8001,2))
            self.assertTrue(np.all(np.isfinite(np.column_stack(data))))
            np.testing.assert_array_equal(time,np.arange(8001)*DT)
            np.testing.assert_array_equal(states[0],np.zeros(4))
            np.testing.assert_array_equal(integrals[0],[0,0])
            np.testing.assert_array_equal(errors,TARGET-states[:,:2])
            np.testing.assert_array_equal(integrals[1:],np.clip(integrals[:-1]+DT*errors[:-1],-.5,.5))
            self.assertTrue(np.all(np.abs(integrals)<=.5))
            cp,pp = model_parameters(model)
            for i in (0,500,7999):
                np.testing.assert_array_equal(torques[i],computed_torque_integral(states[i,:2],states[i,2:],TARGET,integrals[i],[kp,kp],KD,[ki,ki],cp))
                np.testing.assert_array_equal(states[i+1],integrate_dynamics(*states[i],*torques[i],DT,**pp,method="rk2"))

    def test_metrics_and_boundaries(self):
        errors = np.array([[.3,-.4],[.005,.007],[.01,.009],[.006,.008],[0,.009]])
        self.assertEqual(settling_times(np.arange(5)*5.,errors)[0],[15.,None])
        np.testing.assert_allclose(overshoot_percent(np.array([[0,0],[.33,-.48]])),[10,20])
        for over,time,expected in ((19.99,10.,True),(20.,9.,False),(19.,10.01,False),(19.,None,False)):
            self.assertEqual(meets_joint2_goal(dict(overshoot_percent=[0,over],settling_time_s=[0,time])),expected)
        data = self.results["mismatch",25.,7.]
        values = metrics(data)
        np.testing.assert_array_equal(values["final_abs_angle_error_rad"],np.abs(data[3][-1]))
        np.testing.assert_array_equal(values["last_2s_mean_abs_error_rad"],np.mean(np.abs(data[3][7200:]),axis=0))
        np.testing.assert_array_equal(values["last_2s_error_change_rad"],data[3][-1]-data[3][7200])

    def test_pareto_ties_tradeoffs_and_unsettled(self):
        self.assertEqual(pareto_indices([[1,3],[2,2],[3,3],[1,3],[1,np.inf]]),[0,1,3])
        self.assertEqual(pareto_indices([]),[])

    def test_export_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="kp_ki_") as directory:
            output = Path(directory)/"results"
            summary = save_results(self.results,output)
            self.assertEqual(summary,json.loads((output/"summary.json").read_text()))
            self.assertEqual(len(summary["cases"]),40)
            self.assertEqual(len(summary["candidates"]),20)
            with (output/"metrics.csv").open(newline="",encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows),80)
            for row in rows:
                data = self.results[row["model"],float(row["Kp"]),float(row["Ki"])]
                self.assertEqual(float(row["final_abs_angle_error_rad"]),abs(data[3][-1,int(row["joint"])-1]))
            for name in ("overshoot","settling_times","final_errors"):
                self.assertEqual((output/(name+".png")).read_bytes()[:8],b"\x89PNG\r\n\x1a\n")
            for candidate in summary["candidates"]:
                kp,ki = candidate["Kp"],candidate["Ki"]
                for key in OBJECTIVES:
                    values = [v for m in ("baseline","mismatch") for v in metrics(self.results[m,kp,ki])[key]]
                    self.assertEqual(candidate["worst_case_objectives"][key],None if None in values else max(values))
            before = {p.name:p.read_bytes() for p in output.iterdir()}
            with self.assertRaises(FileExistsError):
                save_results(self.results,output)
            self.assertEqual(before,{p.name:p.read_bytes() for p in output.iterdir()})

    def test_invalid_gains(self):
        for kp,ki in ((0,7),(25,-1),(np.nan,7),(25,np.inf),([25,25],7)):
            with self.assertRaises(ValueError):
                simulate("baseline",kp,ki)


if __name__ == "__main__":
    unittest.main()
