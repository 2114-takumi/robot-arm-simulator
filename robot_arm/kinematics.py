import numpy as np


def forward_kinematics(theta1, theta2, L1, L2):
    x1 = L1 * np.cos(theta1)
    y1 = L1 * np.sin(theta1)

    x2 = x1 + L2 * np.cos(theta1 + theta2)
    y2 = y1 + L2 * np.sin(theta1 + theta2)

    return x1, y1, x2, y2

def inverse_kinematics(x, y, L1, L2):
    # Distance from the origin to the target
    r = np.hypot(x, y)

    # Check whether the target is reachable
    if not (abs(L1 - L2) <= r <= L1 + L2):
        raise ValueError("Target is outside the reachable workspace")

    # Calculate the second joint angle
    cos_theta2 = (
        r**2 - L1**2 - L2**2
    ) / (2 * L1 * L2)
    cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
    theta2 = np.arccos(cos_theta2)

    # Calculate the first joint angle
    alpha = np.arctan2(y, x)
    beta = np.arctan2(
        L2 * np.sin(theta2),
        L1 + L2 * np.cos(theta2)
    )
    theta1 = alpha - beta

    return theta1, theta2

def calculate_jacobian(theta1, theta2, L1, L2):
    j11 = -L1 * np.sin(theta1) - L2 * np.sin(theta1 + theta2)
    j12 = -L2 * np.sin(theta1 + theta2)
    j21 = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
    j22 = L2 * np.cos(theta1 + theta2)

    J = np.array([
        [j11, j12],
        [j21, j22]
    ])

    return J


def calculate_jacobian_dot(theta1, theta2, omega1, omega2, L1, L2):

    theta12 = theta1 + theta2
    omega12 = omega1 + omega2

    jdot11 = (
        -L1 * np.cos(theta1) * omega1
        -L2 * np.cos(theta12) * omega12
    )

    jdot12 = -L2 * np.cos(theta12) * omega12

    jdot21 = (
        -L1 * np.sin(theta1) * omega1
        -L2 * np.sin(theta12) * omega12
    )

    jdot22 = -L2 * np.sin(theta12) * omega12

    return np.array([
        [jdot11, jdot12],
        [jdot21, jdot22]
    ])
