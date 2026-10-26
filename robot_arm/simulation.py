import numpy as np

from .kinematics import calculate_jacobian, forward_kinematics


def simulate_pid(
    kd,
    kp,
    ki,
    *,
    initial_angle,
    target_angle,
    dt,
    inertia,
    damping,
    disturbance,
):
    theta = initial_angle
    omega = 0.0
    integral = 0.0

    history = [theta]

    for step in range(200):
        error = target_angle - theta

        integral += error * dt

        tau = (
            kp * error
            + ki * integral
            - kd * omega
        )

        tau += disturbance

        alpha = (tau - damping * omega) / inertia

        omega += alpha * dt
        theta += omega * dt

        history.append(theta)

    return history


def simulate_rk2(
    test_dt,
    *,
    initial_theta1,
    initial_theta2,
    L1,
    L2,
    desired_velocity,
    ideal_x,
    ideal_y,
):
    theta = np.array([initial_theta1, initial_theta2], dtype=float)
    steps = int(round(1.0 / test_dt))

    for _ in range(steps):
        # Velocity at current configuration
        J_current = calculate_jacobian(
            theta[0], theta[1], L1, L2
        )
        omega_current = np.linalg.solve(
            J_current, desired_velocity
        )

        # Predict midpoint
        theta_mid = theta + 0.5 * test_dt * omega_current

        # Velocity at midpoint
        J_mid = calculate_jacobian(
            theta_mid[0], theta_mid[1], L1, L2
        )
        omega_mid = np.linalg.solve(
            J_mid, desired_velocity
        )

        # RK2 update
        theta += test_dt * omega_mid

    # Final end-effector position
    _, _, x, y = forward_kinematics(
        theta[0], theta[1], L1, L2
    )

    return np.hypot(x - ideal_x, y - ideal_y) * 1000
