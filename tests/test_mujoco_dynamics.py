"""MuJoCo と自作2リンク動力学モデルの独立した照合テスト。

実行: python -m unittest tests/test_mujoco_dynamics.py -v
MuJoCo 未導入の場合は、このファイルのテストをスキップする。
既存の models/ 以下には依存せず、MJCF をテスト内に定義する。
"""

import unittest

import numpy as np

try:
    import mujoco
except ImportError:
    mujoco = None

from robot_arm.dynamics import (
    coriolis_centrifugal_vector,
    forward_dynamics,
    gravity_vector,
    integrate_dynamics,
    mass_matrix,
)


# 鉛直XY平面、各関節の回転軸はZ方向。接触・摩擦・アクチュエータなし。
# 慣性テンソルのZ成分を、細い一様棒の重心まわりの慣性に合わせる。
# X成分はMuJoCoが要求する正の慣性値（Z軸回転には寄与しない）。
MJCF = """
<mujoco model="two_link_arm_validation">
  <option timestep="0.002" gravity="0 -9.81 0" integrator="Euler"/>
  <worldbody>
    <body name="link1" pos="0 0 0">
      <joint name="joint1" type="hinge" axis="0 0 1" damping="0"/>
      <inertial pos="0.5 0 0" mass="1"
                diaginertia="0.0001 0.08333333333333333 0.08333333333333333"/>
      <body name="link2" pos="1 0 0">
        <joint name="joint2" type="hinge" axis="0 0 1" damping="0"/>
        <inertial pos="0.4 0 0" mass="0.8"
                  diaginertia="0.0001 0.04266666666666667 0.04266666666666667"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

PARAMS = dict(
    L1=1.0, m1=1.0, m2=0.8, lc1=0.5, lc2=0.4,
    I1=1.0 / 12.0, I2=0.8 * 0.8**2 / 12.0,
    b1=0.0, b2=0.0, g=9.81,
)
INITIAL_STATE = np.array([np.deg2rad(30.0), np.deg2rad(45.0), 1.0, -0.5])
TORQUE = np.array([12.0, 1.0])


@unittest.skipUnless(mujoco is not None, "MuJoCo is not installed")
class TestMuJoCoDynamics(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string(MJCF)
        self.data = mujoco.MjData(self.model)
        self.assertEqual(self.model.nq, 2)
        self.assertEqual(self.model.nv, 2)

    def set_state(self, q, v=(0.0, 0.0), torque=(0.0, 0.0)):
        self.data.qpos[:] = q
        self.data.qvel[:] = v
        self.data.qfrc_applied[:] = torque
        mujoco.mj_forward(self.model, self.data)

    def test_mass_matrix_matches_mujoco(self):
        for q in ((0.0, 0.0), (0.0, np.pi / 2), (0.4, -0.7)):
            with self.subTest(q=q):
                self.set_state(q)
                full = np.zeros((2, 2))
                for column in range(2):
                    basis = np.zeros(2)
                    basis[column] = 1.0
                    result = np.zeros(2)
                    mujoco.mj_mulM(self.model, self.data, result, basis)
                    full[:, column] = result
                expected = mass_matrix(*q, **{
                    key: PARAMS[key] for key in
                    ("L1", "m1", "m2", "lc1", "lc2", "I1", "I2")
                })
                np.testing.assert_allclose(full, expected, rtol=0, atol=1e-9)

    def test_gravity_matches_mujoco(self):
        for q in ((0.0, 0.0), (np.pi / 2, 0.0), (0.0, np.pi / 2),
                  (np.deg2rad(30.0), np.deg2rad(45.0))):
            with self.subTest(q=q):
                self.set_state(q)
                expected = gravity_vector(*q, **{
                    key: PARAMS[key] for key in
                    ("L1", "m1", "m2", "lc1", "lc2", "g")
                })
                np.testing.assert_allclose(
                    self.data.qfrc_bias, expected, rtol=0, atol=1e-9
                )

    def test_coriolis_centrifugal_matches_mujoco(self):
        q = (np.deg2rad(30.0), np.deg2rad(45.0))
        v = (1.0, -0.5)
        self.set_state(q, (0.0, 0.0))
        gravity_bias = self.data.qfrc_bias.copy()
        self.set_state(q, v)
        velocity_bias = self.data.qfrc_bias - gravity_bias
        expected = coriolis_centrifugal_vector(
            *q, *v, L1=PARAMS["L1"], m2=PARAMS["m2"], lc2=PARAMS["lc2"]
        )
        np.testing.assert_allclose(velocity_bias, expected, rtol=0, atol=1e-9)

    def test_forward_dynamics_matches_mujoco(self):
        q = INITIAL_STATE[:2]
        v = INITIAL_STATE[2:]
        self.set_state(q, v, TORQUE)
        expected = forward_dynamics(*q, *v, *TORQUE, **PARAMS)
        np.testing.assert_allclose(self.data.qacc, expected, rtol=0, atol=1e-8)

    def test_semi_implicit_euler_one_step_matches_mujoco(self):
        dt = 0.002
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
        self.model.opt.timestep = dt
        self.set_state(INITIAL_STATE[:2], INITIAL_STATE[2:], TORQUE)
        alpha = forward_dynamics(*INITIAL_STATE, *TORQUE, **PARAMS)
        new_velocity = INITIAL_STATE[2:] + dt * alpha
        new_angle = INITIAL_STATE[:2] + dt * new_velocity
        mujoco.mj_step(self.model, self.data)
        np.testing.assert_allclose(self.data.qvel, new_velocity, rtol=0, atol=1e-8)
        np.testing.assert_allclose(self.data.qpos, new_angle, rtol=0, atol=1e-8)

    def run_mujoco_rk4(self, dt=0.0001):
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
        self.model.opt.timestep = dt
        self.set_state(INITIAL_STATE[:2], INITIAL_STATE[2:], TORQUE)
        for _ in range(round(1.0 / dt)):
            mujoco.mj_step(self.model, self.data)
        return np.concatenate((self.data.qpos.copy(), self.data.qvel.copy()))

    @staticmethod
    def run_custom(method, dt):
        state = INITIAL_STATE.copy()
        for _ in range(round(1.0 / dt)):
            state = integrate_dynamics(
                *state, *TORQUE, dt, **PARAMS, method=method
            )
        return state

    def test_euler_first_order_and_rk2_second_order_convergence(self):
        # 高精度の独立基準解を使用。角度・角速度を別々に評価する。
        reference = self.run_mujoco_rk4(dt=0.0001)
        for method, upper_ratio in (("euler", 0.55), ("rk2", 0.30)):
            errors = []
            for dt in (0.002, 0.001, 0.0005):
                result = self.run_custom(method, dt)
                errors.append(np.abs(result - reference))
            for part, slc in (("angle", slice(0, 2)),
                              ("velocity", slice(2, 4))):
                with self.subTest(method=method, part=part):
                    magnitudes = [float(np.max(e[slc])) for e in errors]
                    self.assertGreater(magnitudes[0], 0)
                    self.assertLess(magnitudes[1] / magnitudes[0], upper_ratio)
                    self.assertLess(magnitudes[2] / magnitudes[1], upper_ratio)


if __name__ == "__main__":
    unittest.main()
