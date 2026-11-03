r"""Step 8-6: compare the existing computed-torque controller against MuJoCo.

Run from the repository root:
    .\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_mujoco_computed_torque.py" -v

No existing project files or outputs are modified by these tests.
The controller is sampled at the beginning of each step and its torque is held
constant within that step for BOTH solvers. The custom solver uses RK2; MuJoCo
uses RK4. Their results should be close, not bit-for-bit identical.
"""

import unittest

import numpy as np

try:
    import mujoco
except ImportError:
    mujoco = None

from robot_arm.computed_torque import computed_torque
from robot_arm.dynamics import integrate_dynamics
from scripts.run_computed_torque_demo import (
    CTC_KD, CTC_KP, DT, DURATION, TARGET, simulate,
)
from scripts.run_coupled_dynamics_demo import physical_parameters


# Identical physical model to tests/test_mujoco_dynamics.py.
# The standalone MJCF avoids changing the user's untracked models/ directory.
MJCF = """
<mujoco model="computed_torque_validation">
  <option timestep="0.0025" gravity="0 -9.81 0" integrator="RK4"/>
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


def run_mujoco(dt, duration):
    """Return (time, state, commanded torque), including the final sample.

    qfrc_applied is the generalized joint torque; it stays fixed through each
    MuJoCo RK4 step. The controller is recomputed at the next sample.
    """
    params = physical_parameters()
    steps = round(duration / dt)
    assert np.isclose(steps * dt, duration, rtol=0, atol=1e-12)
    model = mujoco.MjModel.from_xml_string(MJCF)
    model.opt.timestep = dt
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    time = np.arange(steps + 1) * dt
    states = np.zeros((steps + 1, 4))
    torques = np.zeros((steps + 1, 2))
    for i in range(steps + 1):
        states[i, :2] = data.qpos
        states[i, 2:] = data.qvel
        torques[i] = computed_torque(
            data.qpos, data.qvel, TARGET, CTC_KP, CTC_KD, params
        )
        if i < steps:
            data.qfrc_applied[:] = torques[i]
            mujoco.mj_step(model, data)
    return time, states, torques


def run_custom_rk2(dt, duration):
    """Same sampled controller, with the existing custom RK2 dynamics."""
    params = physical_parameters()
    steps = round(duration / dt)
    assert np.isclose(steps * dt, duration, rtol=0, atol=1e-12)
    time = np.arange(steps + 1) * dt
    states = np.zeros((steps + 1, 4))
    torques = np.zeros((steps + 1, 2))
    for i in range(steps + 1):
        torques[i] = computed_torque(
            states[i, :2], states[i, 2:], TARGET, CTC_KP, CTC_KD, params
        )
        if i < steps:
            states[i + 1] = integrate_dynamics(
                *states[i], *torques[i], dt, **params,
                b1=0.0, b2=0.0, g=9.81, method="rk2"
            )
    return time, states, torques


@unittest.skipUnless(mujoco is not None, "MuJoCo is not installed")
class TestMuJoCoComputedTorque(unittest.TestCase):
    def test_initial_acceleration_is_requested_acceleration(self):
        """At t=0, computed torque cancels gravity in MuJoCo."""
        model = mujoco.MjModel.from_xml_string(MJCF)
        data = mujoco.MjData(model)
        tau = computed_torque(
            np.zeros(2), np.zeros(2), TARGET,
            CTC_KP, CTC_KD, physical_parameters()
        )
        data.qfrc_applied[:] = tau
        mujoco.mj_forward(model, data)
        requested = CTC_KP * TARGET
        np.testing.assert_allclose(data.qacc, requested, rtol=0, atol=1e-8)

    def test_four_second_closed_loop_matches_existing_demo(self):
        """Same feedback law, independent physics engine, different integrator."""
        t_m, states_m, torques_m = run_mujoco(DT, DURATION)
        t_c, states_c, torques_c, errors_c = simulate("computed_torque")
        np.testing.assert_array_equal(t_m, t_c)
        np.testing.assert_array_equal(errors_c, TARGET - states_c[:, :2])

        angle_difference = float(np.max(np.abs(states_m[:, :2] - states_c[:, :2])))
        velocity_difference = float(np.max(np.abs(states_m[:, 2:] - states_c[:, 2:])))
        torque_difference = float(np.max(np.abs(torques_m - torques_c)))
        final_error = np.abs(TARGET - states_m[-1, :2])
        print(
            "\nStep 8-6: MuJoCo RK4 vs custom RK2 (4 s, dt=0.0025 s)"
            f"\n  max angle difference: {angle_difference:.9g} rad"
            f"\n  max velocity difference: {velocity_difference:.9g} rad/s"
            f"\n  max torque difference: {torque_difference:.9g} N m"
            f"\n  MuJoCo final angle error: {final_error.tolist()} rad",
            flush=True,
        )
        self.assertTrue(np.all(np.isfinite(states_m)))
        self.assertTrue(np.all(np.isfinite(torques_m)))
        self.assertLess(angle_difference, 0.01)       # < 0.57 deg
        self.assertLess(velocity_difference, 0.05)    # rad/s
        self.assertLess(torque_difference, 0.5)       # N m
        self.assertTrue(np.all(final_error < 0.001))  # per joint, rad

    def test_rk2_vs_rk4_difference_decreases_with_smaller_dt(self):
        """Integration error should shrink when both use a smaller sample time."""
        angle_errors = []
        for dt in (0.005, 0.0025):
            _, states_m, _ = run_mujoco(dt, 2.0)
            _, states_c, _ = run_custom_rk2(dt, 2.0)
            angle_errors.append(float(np.max(np.abs(states_m[:, :2] - states_c[:, :2]))))
        print(
            f"\nStep 8-6: max angle difference (dt=0.005 / 0.0025): "
            f"{angle_errors[0]:.9g} / {angle_errors[1]:.9g} rad",
            flush=True,
        )
        self.assertGreater(angle_errors[0], 1e-12)
        self.assertLess(angle_errors[1], 0.6 * angle_errors[0])


if __name__ == "__main__":
    unittest.main()
