"""CSVから描画座標への変換と端点の選択を検証する。"""

from io import StringIO
import unittest

import numpy as np

from scripts.animate_coupled_dynamics import link_coordinates, load_csv, select_frames


class TestAnimateCoupledDynamics(unittest.TestCase):
    def test_csv_and_animation_endpoints(self):
        # 実際の列順で、相対角の解釈が区別できる最終姿勢を保存する。
        csv = StringIO(
            "time_s,theta1_rad,theta2_rad,omega1_rad_s,omega2_rad_s,T_J,V_J,E_J\n"
            "0,0,0,0,0,0,0,0\n"
            "0.5,0.3,0.7,1,2,3,4,7\n"
            f"1,{np.pi / 2},{-np.pi / 2},1,2,3,4,7\n"
        )
        time, angles = load_csv(csv)
        np.testing.assert_array_equal(time, [0, 0.5, 1])
        np.testing.assert_allclose(angles, [[0, 0], [0.3, 0.7], [np.pi / 2, -np.pi / 2]])
        frames = select_frames(time)
        self.assertEqual(frames[0], 0)
        self.assertEqual(frames[-1], len(time) - 1)
        x, y = link_coordinates(angles)
        np.testing.assert_allclose(x[frames[0]], [0, 1, 1.8])
        np.testing.assert_allclose(y[frames[0]], [0, 0, 0])
        np.testing.assert_allclose(x[frames[-1]], [0, 0, 0.8], atol=1e-14)
        np.testing.assert_allclose(y[frames[-1]], [0, 1, 1], atol=1e-14)

    def test_link_lengths_and_finite_coordinates(self):
        angles = np.column_stack((np.linspace(-4, 4, 801), np.linspace(2, -3, 801)))
        x, y = link_coordinates(angles)
        self.assertTrue(np.all(np.isfinite(x)))
        self.assertTrue(np.all(np.isfinite(y)))
        lengths = np.hypot(np.diff(x, axis=1), np.diff(y, axis=1))
        np.testing.assert_allclose(lengths[:, 0], 1.0, rtol=0, atol=1e-14)
        np.testing.assert_allclose(lengths[:, 1], 0.8, rtol=0, atol=1e-14)
        frames = select_frames(np.arange(801) * 0.0025)
        self.assertEqual(len(frames), 61)
        np.testing.assert_array_equal(frames[[0, -1]], [0, 800])


if __name__ == "__main__":
    unittest.main()
