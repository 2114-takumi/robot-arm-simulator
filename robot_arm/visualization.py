from .kinematics import forward_kinematics


def update(frame, *, get_angle_histories, L1, L2, line):
    theta1_angles, theta2_angles = get_angle_histories()
    theta1_new = theta1_angles[frame]
    theta2_new = theta2_angles[frame]
    x1, y1, x2, y2 = forward_kinematics(
        theta1_new, theta2_new, L1, L2
    )
    line.set_data(
        [0, x1, x2],
        [0, y1, y2]
    )
    return line,
