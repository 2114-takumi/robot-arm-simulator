"""2リンクロボットの慣性行列と速度依存トルクの単体テスト。"""

import unittest

import numpy as np

from robot_arm.dynamics import coriolis_centrifugal_vector, mass_matrix


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


if __name__ == "__main__":
    unittest.main()
