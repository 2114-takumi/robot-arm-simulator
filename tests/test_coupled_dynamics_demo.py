"""独立した自由運動デモの物理計算を最小限検証する。"""

import unittest
import numpy as np

from scripts.run_coupled_dynamics_demo import calculate_energy, physical_parameters, simulate
from robot_arm.dynamics import forward_dynamics, gravity_vector


class TestCoupledDynamicsDemo(unittest.TestCase):
    def test_histories_and_initial_acceleration(self):
        parameters = physical_parameters()
        alpha = forward_dynamics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, **parameters)
        for method in ("euler", "rk2"):
            with self.subTest(method=method):
                time, states, energies = simulate(method)
                self.assertEqual(states.shape, (len(time), 4))
                self.assertEqual(energies.shape, (len(time), 3))
                self.assertEqual(len(time), 801)
                self.assertEqual(time[-1], 2.0)
                np.testing.assert_array_equal(states[0], np.zeros(4))
                self.assertTrue(np.all(np.isfinite(states)))
                self.assertTrue(np.all(np.isfinite(energies)))
                np.testing.assert_allclose(energies[:, 2], energies[:, 0] + energies[:, 1])
                # 差分加速度が直接計算に一致することを小さいdtでも確認する。
                fine_time, fine_states, _ = simulate(method, dt=1e-6, duration=1e-6)
                np.testing.assert_allclose(fine_states[1, 2:] / fine_time[1], alpha, rtol=0, atol=1e-9)

    def test_potential_gradient_matches_gravity(self):
        parameters = physical_parameters()
        # 水平静止ではT=V=0。鉛直上向きの位置エネルギーは1.62*9.81。
        np.testing.assert_array_equal(calculate_energy(np.zeros(4), parameters), np.zeros(3))
        self.assertAlmostEqual(calculate_energy([np.pi / 2, 0, 0, 0], parameters)[1], 15.8922)
        for angles in ((0.0, 0.0), (0.3, -0.7)):
            state = np.array([*angles, 0.0, 0.0])
            gradient = []
            for joint in (0, 1):
                plus, minus = state.copy(), state.copy()
                plus[joint] += 1e-6
                minus[joint] -= 1e-6
                gradient.append((calculate_energy(plus, parameters)[1] - calculate_energy(minus, parameters)[1]) / 2e-6)
            gravity = gravity_vector(*angles, L1=1.0, m1=1.0, m2=0.8, lc1=0.5, lc2=0.4)
            np.testing.assert_allclose(gradient, gravity, rtol=0, atol=1e-8)

    def test_short_time_rk2_energy_error_decreases(self):
        errors = []
        for dt in (0.005, 0.0025):
            _, _, energies = simulate("rk2", dt=dt, duration=0.2)
            errors.append(np.max(np.abs(energies[:, 2] - energies[0, 2])))
        self.assertLess(errors[1], errors[0])
        self.assertLess(errors[1], 0.001)  # 今回の0.2秒間の条件に対する絶対誤差 [J]。


if __name__ == "__main__":
    unittest.main()
