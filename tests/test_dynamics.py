"""2リンクロボットの動力学関数の単体テスト。"""

import unittest

import numpy as np

from robot_arm.dynamics import (
    coriolis_centrifugal_vector,
    forward_dynamics,
    gravity_vector,
    integrate_dynamics,
    mass_matrix,
)


class TestMassMatrix(unittest.TestCase):
    def setUp(self):
        # 一様な細い棒を仮定したテスト用の値であり、実機の仕様ではない。
        L1, L2 = 1.0, 0.8  # m
        m1, m2 = 1.0, 0.8  # kg
        self.parameters = dict(
            L1=L1, m1=m1, m2=m2, lc1=0.5, lc2=0.4,
            I1=m1 * L1**2 / 12, I2=m2 * L2**2 / 12,
        )

    def test_returns_two_by_two_array(self):
        matrix = mass_matrix(0.3, 0.7, **self.parameters)
        self.assertIsInstance(matrix, np.ndarray)
        self.assertEqual(matrix.shape, (2, 2))

    def test_is_symmetric(self):
        for theta2 in (-np.pi, -0.7, 0.0, np.pi / 2, np.pi):
            with self.subTest(theta2=theta2):
                matrix = mass_matrix(0.3, theta2, **self.parameters)
                np.testing.assert_array_equal(matrix, matrix.T)

    def test_is_positive_definite(self):
        # 対称行列の固有値がすべて正なら、運動エネルギーは正となる。
        for theta2 in np.linspace(-np.pi, np.pi, 101):
            with self.subTest(theta2=theta2):
                matrix = mass_matrix(0.3, theta2, **self.parameters)
                self.assertTrue(np.all(np.linalg.eigvalsh(matrix) > 0))

    def test_theta2_changes_coupled_elements(self):
        straight = mass_matrix(0.3, 0.0, **self.parameters)
        bent = mass_matrix(0.3, np.pi / 2, **self.parameters)
        self.assertNotAlmostEqual(straight[0, 0], bent[0, 0])
        self.assertNotAlmostEqual(straight[0, 1], bent[0, 1])
        self.assertEqual(straight[1, 1], bent[1, 1])

    def test_matches_hand_calculation(self):
        # 手計算: M11 = 1.304 + 0.64*cos(theta2),
        # M12 = 64/375 + 0.32*cos(theta2), M22 = 64/375。
        cases = (
            (0.0, [[1.944, 184 / 375], [184 / 375, 64 / 375]]),
            (np.pi / 2, [[1.304, 64 / 375], [64 / 375, 64 / 375]]),
            (np.pi, [[0.664, -56 / 375], [-56 / 375, 64 / 375]]),
        )
        for theta2, expected in cases:
            with self.subTest(theta2=theta2):
                matrix = mass_matrix(0.3, theta2, **self.parameters)
                np.testing.assert_allclose(matrix, expected, rtol=0, atol=1e-14)

    def test_same_inputs_return_same_result(self):
        first = mass_matrix(0.3, 0.7, **self.parameters)
        second = mass_matrix(0.3, 0.7, **self.parameters)
        np.testing.assert_array_equal(first, second)

    def test_theta1_does_not_change_matrix(self):
        first = mass_matrix(0.0, 0.7, **self.parameters)
        second = mass_matrix(np.pi / 2, 0.7, **self.parameters)
        np.testing.assert_array_equal(first, second)


class TestCoriolisCentrifugalVector(unittest.TestCase):
    def setUp(self):
        # 一様な細い棒モデルのテスト用の値であり、実機の仕様ではない。
        self.parameters = dict(L1=1.0, m2=0.8, lc2=0.4)

    def test_returns_two_component_array(self):
        vector = coriolis_centrifugal_vector(0.3, 0.7, 1.0, 2.0, **self.parameters)
        self.assertIsInstance(vector, np.ndarray)
        self.assertEqual(vector.shape, (2,))

    def test_zero_velocities_give_zero_torque(self):
        vector = coriolis_centrifugal_vector(0.3, 0.7, 0.0, 0.0, **self.parameters)
        np.testing.assert_array_equal(vector, [0.0, 0.0])

    def test_aligned_links_give_nearly_zero_torque(self):
        for theta2 in (0.0, np.pi):
            with self.subTest(theta2=theta2):
                vector = coriolis_centrifugal_vector(
                    0.3, theta2, 1.0, 2.0, **self.parameters
                )
                # sin(pi) の浮動小数点誤差を許容する。
                np.testing.assert_allclose(vector, [0.0, 0.0], rtol=0, atol=1e-14)

    def test_matches_hand_calculation(self):
        # theta2=pi/6 より h=0.16。
        # omega1=2, omega2=3: c1=-0.16*(12+9)=-3.36, c2=0.64。
        # omega2=-3: c1=-0.16*(-12+9)=0.48, c2=0.64。
        for omega2, expected in ((3.0, [-3.36, 0.64]), (-3.0, [0.48, 0.64])):
            with self.subTest(omega2=omega2):
                vector = coriolis_centrifugal_vector(
                    0.3, np.pi / 6, 2.0, omega2, **self.parameters
                )
                np.testing.assert_allclose(vector, expected, rtol=0, atol=1e-14)

    def test_theta1_does_not_change_vector(self):
        first = coriolis_centrifugal_vector(0.0, 0.7, 1.0, 2.0, **self.parameters)
        second = coriolis_centrifugal_vector(np.pi / 2, 0.7, 1.0, 2.0, **self.parameters)
        np.testing.assert_array_equal(first, second)

    def test_doubling_velocities_quadruples_torque(self):
        first = coriolis_centrifugal_vector(0.3, 0.7, 1.0, -3.0, **self.parameters)
        second = coriolis_centrifugal_vector(0.3, 0.7, 2.0, -6.0, **self.parameters)
        np.testing.assert_allclose(second, 4 * first, rtol=0, atol=1e-14)

    def test_only_first_joint_rotating_at_right_angle(self):
        vector = coriolis_centrifugal_vector(
            0.3, np.pi / 2, 1.0, 0.0, **self.parameters
        )
        np.testing.assert_allclose(vector, [0.0, 0.32], rtol=0, atol=1e-14)


class TestGravityVector(unittest.TestCase):
    def setUp(self):
        # 一様な細い棒モデルのテスト用の値であり、実機の仕様ではない。
        self.parameters = dict(L1=1.0, m1=1.0, m2=0.8, lc1=0.5, lc2=0.4)

    def test_returns_two_component_array(self):
        vector = gravity_vector(0.3, 0.7, **self.parameters)
        self.assertIsInstance(vector, np.ndarray)
        self.assertEqual(vector.shape, (2,))

    def test_horizontal_pose_matches_hand_calculation(self):
        # g1=(0.5+0.8+0.32)*9.81=15.8922, g2=0.32*9.81=3.1392。
        vector = gravity_vector(0.0, 0.0, **self.parameters)
        np.testing.assert_allclose(vector, [15.8922, 3.1392], rtol=0, atol=1e-14)

    def test_vertical_upward_pose_gives_nearly_zero_torque(self):
        vector = gravity_vector(np.pi / 2, 0.0, **self.parameters)
        # cos(pi/2) の浮動小数点誤差を許容する。
        np.testing.assert_allclose(vector, [0.0, 0.0], rtol=0, atol=1e-14)

    def test_zero_gravity_gives_zero_torque(self):
        vector = gravity_vector(0.3, 0.7, g=0.0, **self.parameters)
        np.testing.assert_array_equal(vector, [0.0, 0.0])

    def test_doubling_gravity_doubles_torque(self):
        first = gravity_vector(0.3, 0.7, g=9.81, **self.parameters)
        second = gravity_vector(0.3, 0.7, g=19.62, **self.parameters)
        np.testing.assert_allclose(second, 2 * first, rtol=0, atol=1e-14)

    def test_bent_pose_matches_hand_calculation(self):
        # theta1=pi/3, theta2=-pi/3: cos(theta1)=0.5, cos(theta1+theta2)=1。
        # g1=1.3*9.81*0.5+3.1392=9.5157, g2=3.1392。
        # theta1=pi/3, theta2=2*pi/3: cos(theta1+theta2)=-1。
        # g1=1.3*9.81*0.5-3.1392=3.2373, g2=-3.1392。
        cases = (
            (-np.pi / 3, [9.5157, 3.1392]),
            (2 * np.pi / 3, [3.2373, -3.1392]),
        )
        for theta2, expected in cases:
            with self.subTest(theta2=theta2):
                vector = gravity_vector(np.pi / 3, theta2, **self.parameters)
                np.testing.assert_allclose(vector, expected, rtol=0, atol=1e-14)

    def test_same_inputs_return_same_result(self):
        first = gravity_vector(0.3, 0.7, **self.parameters)
        second = gravity_vector(0.3, 0.7, **self.parameters)
        np.testing.assert_array_equal(first, second)


class TestForwardDynamics(unittest.TestCase):
    def setUp(self):
        # 一様な細い棒モデルのテスト用の値であり、実機の仕様ではない。
        self.parameters = dict(
            L1=1.0, m1=1.0, m2=0.8, lc1=0.5, lc2=0.4,
            I1=1 / 12, I2=0.8 * 0.8**2 / 12,
        )
        self.gravity_parameters = dict(L1=1.0, m1=1.0, m2=0.8, lc1=0.5, lc2=0.4)

    def test_returns_two_component_array(self):
        acceleration = forward_dynamics(0.3, 0.7, 0.4, -0.6, 1.0, 2.0, **self.parameters)
        self.assertIsInstance(acceleration, np.ndarray)
        self.assertEqual(acceleration.shape, (2,))

    def test_gravity_compensation_at_rest_gives_zero_acceleration(self):
        for theta1, theta2 in ((0.0, 0.0), (0.3, -0.7), (np.pi / 2, 0.0)):
            with self.subTest(theta1=theta1, theta2=theta2):
                gravity = gravity_vector(theta1, theta2, **self.gravity_parameters)
                acceleration = forward_dynamics(
                    theta1, theta2, 0.0, 0.0, gravity[0], gravity[1], **self.parameters
                )
                np.testing.assert_allclose(acceleration, [0.0, 0.0], rtol=0, atol=1e-12)

    def test_inertia_only_equation(self):
        acceleration = forward_dynamics(
            0.3, 0.7, 0.0, 0.0, 1.2, -0.4, g=0.0, **self.parameters
        )
        M = mass_matrix(0.3, 0.7, **self.parameters)
        np.testing.assert_allclose(M @ acceleration, [1.2, -0.4], rtol=0, atol=1e-12)

    def test_first_joint_torque_accelerates_both_joints(self):
        gravity = gravity_vector(0.0, np.pi / 2, **self.gravity_parameters)
        acceleration = forward_dynamics(
            0.0, np.pi / 2, 0.0, 0.0, gravity[0] + 1.0, gravity[1],
            **self.parameters
        )
        # 正味トルク [1, 0] でも慣性結合で第2関節が加速する。
        # 手計算: alpha2=-alpha1, (1.304-64/375)*alpha1=1。
        np.testing.assert_allclose(
            acceleration, [15 / 17, -15 / 17], rtol=0, atol=1e-12
        )

    def test_viscous_damping(self):
        # 外力・重力なしで、M @ qddot = -[b1*omega1, b2*omega2] を確認。
        # theta2=0 により速度依存トルクも0になる。
        M = mass_matrix(0.3, 0.0, **self.parameters)
        for b1, b2 in ((0.2, 0.0), (0.0, 0.3), (0.2, 0.3)):
            with self.subTest(b1=b1, b2=b2):
                acceleration = forward_dynamics(
                    0.3, 0.0, 0.4, -0.6, 0.0, 0.0,
                    b1=b1, b2=b2, g=0.0, **self.parameters
                )
                expected_torque = [-b1 * 0.4, -b2 * (-0.6)]
                np.testing.assert_allclose(
                    M @ acceleration, expected_torque, rtol=0, atol=1e-12
                )

    def test_satisfies_full_equation_of_motion(self):
        # 全項が非ゼロになる条件で、計算結果を元の運動方程式に戻す。
        acceleration = forward_dynamics(
            0.3, 0.7, 0.4, -0.6, 1.2, -0.4,
            b1=0.2, b2=0.3, g=3.7, **self.parameters
        )
        M = mass_matrix(0.3, 0.7, **self.parameters)
        c = coriolis_centrifugal_vector(0.3, 0.7, 0.4, -0.6, L1=1.0, m2=0.8, lc2=0.4)
        gravity = gravity_vector(0.3, 0.7, g=3.7, **self.gravity_parameters)
        damping = np.array([0.2 * 0.4, 0.3 * (-0.6)])
        np.testing.assert_allclose(
            M @ acceleration + c + gravity + damping, [1.2, -0.4],
            rtol=0, atol=1e-12
        )

    def test_same_inputs_return_same_result(self):
        first = forward_dynamics(0.3, 0.7, 0.4, -0.6, 1.0, 2.0, **self.parameters)
        second = forward_dynamics(0.3, 0.7, 0.4, -0.6, 1.0, 2.0, **self.parameters)
        np.testing.assert_array_equal(first, second)


class TestIntegrateDynamics(unittest.TestCase):
    def setUp(self):
        # 一様な細い棒モデルのテスト値であり、実機の仕様ではない。
        self.parameters = dict(
            L1=1.0, m1=1.0, m2=0.8, lc1=0.5, lc2=0.4,
            I1=1 / 12, I2=0.8 * 0.8**2 / 12,
        )
        self.initial = np.array([0.3, 0.7, 0.4, -0.6])

    def test_euler_returns_four_component_array(self):
        result = integrate_dynamics(*self.initial, 1.2, -0.4, 0.01, **self.parameters)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (4,))

    def test_rk2_returns_four_component_array(self):
        result = integrate_dynamics(
            *self.initial, 1.2, -0.4, 0.01, method="rk2", **self.parameters
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (4,))

    def test_rest_without_torque_or_gravity_is_unchanged(self):
        for method in ("euler", "rk2"):
            with self.subTest(method=method):
                result = integrate_dynamics(
                    0.3, 0.7, 0.0, 0.0, 0.0, 0.0, 0.01,
                    g=0.0, method=method, **self.parameters
                )
                np.testing.assert_array_equal(result, [0.3, 0.7, 0.0, 0.0])

    def test_gravity_compensated_rest_is_unchanged(self):
        gravity = gravity_vector(0.3, 0.7, L1=1.0, m1=1.0, m2=0.8, lc1=0.5, lc2=0.4)
        for method in ("euler", "rk2"):
            with self.subTest(method=method):
                result = integrate_dynamics(
                    0.3, 0.7, 0.0, 0.0, gravity[0], gravity[1], 0.01,
                    method=method, **self.parameters
                )
                np.testing.assert_allclose(result, [0.3, 0.7, 0.0, 0.0], rtol=0, atol=1e-14)

    def test_euler_matches_definition(self):
        alpha = forward_dynamics(
            *self.initial, 1.2, -0.4, b1=0.2, b2=0.3, g=3.7, **self.parameters
        )
        derivative = np.array([0.4, -0.6, alpha[0], alpha[1]])
        expected = self.initial + 0.02 * derivative
        actual = integrate_dynamics(
            *self.initial, 1.2, -0.4, 0.02,
            b1=0.2, b2=0.3, g=3.7, method="euler", **self.parameters
        )
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)

    def test_rk2_matches_midpoint_definition(self):
        # 非ゼロの速度・加速度により、中点で角度と角速度が両方変わる。
        alpha = forward_dynamics(
            *self.initial, 1.2, -0.4, b1=0.2, b2=0.3, g=3.7, **self.parameters
        )
        midpoint = self.initial + 0.01 * np.array([0.4, -0.6, alpha[0], alpha[1]])
        alpha_mid = forward_dynamics(
            *midpoint, 1.2, -0.4, b1=0.2, b2=0.3, g=3.7, **self.parameters
        )
        expected = self.initial + 0.02 * np.array([
            midpoint[2], midpoint[3], alpha_mid[0], alpha_mid[1]
        ])
        actual = integrate_dynamics(
            *self.initial, 1.2, -0.4, 0.02,
            b1=0.2, b2=0.3, g=3.7, method="rk2", **self.parameters
        )
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)

    def test_rejects_invalid_method_and_dt(self):
        for method in ("invalid", "Euler", "", None):
            with self.subTest(method=method), self.assertRaises(ValueError):
                integrate_dynamics(*self.initial, 0.0, 0.0, 0.01, method=method, **self.parameters)
        for method in ("euler", "rk2"):
            for dt in (0.0, -0.01, np.nan, np.inf, -np.inf):
                with self.subTest(method=method, dt=dt), self.assertRaises(ValueError):
                    integrate_dynamics(*self.initial, 0.0, 0.0, dt, method=method, **self.parameters)

    def test_same_inputs_return_same_result(self):
        for method in ("euler", "rk2"):
            with self.subTest(method=method):
                first = integrate_dynamics(*self.initial, 1.2, -0.4, 0.01, method=method, **self.parameters)
                second = integrate_dynamics(*self.initial, 1.2, -0.4, 0.01, method=method, **self.parameters)
                np.testing.assert_array_equal(first, second)

    def free_motion(self, dt, method):
        """摩擦・モータトルクなし、重力ありで0.2秒間の状態履歴を返す。"""
        state = self.initial.copy()
        history = [state.copy()]
        for _ in range(round(0.2 / dt)):
            state = integrate_dynamics(*state, 0.0, 0.0, dt, method=method, **self.parameters)
            history.append(state.copy())
        return np.array(history)

    def test_time_step_refinement_converges(self):
        # 微細刻みのRK2解を参照値とし、同じ終端時刻の状態を比較する。
        reference = self.free_motion(0.000125, "rk2")[-1]
        for method in ("euler", "rk2"):
            with self.subTest(method=method):
                errors = []
                for dt in (0.01, 0.005, 0.0025):
                    difference = self.free_motion(dt, method)[-1] - reference
                    errors.append(np.linalg.norm(difference))
                # 刻みを半分にした際、Eulerは約1/2、RK2は約1/4に減る。
                limit = 0.7 if method == "euler" else 0.4
                self.assertLess(errors[1], limit * errors[0])
                self.assertLess(errors[2], limit * errors[1])

    def test_time_step_refinement_reduces_energy_error(self):
        for method in ("euler", "rk2"):
            with self.subTest(method=method):
                errors = []
                for dt in (0.01, 0.005, 0.0025):
                    energies = []
                    for state in self.free_motion(dt, method):
                        theta1, theta2, omega1, omega2 = state
                        omega = np.array([omega1, omega2])
                        M = mass_matrix(theta1, theta2, **self.parameters)
                        kinetic = 0.5 * omega @ M @ omega
                        # U = m1*g*y重心1 + m2*g*y重心2。高さ0を位置エネルギーの基準とする。
                        potential = 9.81 * (
                            (1.0 * 0.5 + 0.8 * 1.0) * np.sin(theta1)
                            + 0.8 * 0.4 * np.sin(theta1 + theta2)
                        )
                        energies.append(kinetic + potential)
                    errors.append(np.max(np.abs(np.array(energies) - energies[0])))
                self.assertLess(errors[1], errors[0])
                self.assertLess(errors[2], errors[1])


if __name__ == "__main__":
    unittest.main()
