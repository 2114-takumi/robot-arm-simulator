
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

L1 = 1.0
L2 = 0.8

theta1 = np.deg2rad(30)
theta2 = np.deg2rad(-45)

def forward_kinematics(theta1, theta2, L1, L2):
    x1 = L1 * np.cos(theta1)
    y1 = L1 * np.sin(theta1)

    x2 = x1 + L2 * np.cos(theta1 + theta2)
    y2 = y1 + L2 * np.sin(theta1 + theta2)

    return x1, y1, x2, y2


x1, y1, x, y = forward_kinematics(theta1, theta2, L1, L2)


print(f"End Effector: x={x}, y={y}")


# ロボットアームの描画
line, = plt.plot([0, x1, x], [0, y1, y], marker="o", linewidth=3)

plt.axis("equal")
plt.grid(True)
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.title("2-Link Robot Arm")



# アニメーション中に軸が変わらないように固定
plt.xlim(-2, 2)
plt.ylim(-2, 2)

def update(frame):
    theta2_new = np.deg2rad(-45 + frame)

    x1, y1, x2, y2 = forward_kinematics(
        theta1, theta2_new, L1, L2
    )

    line.set_data([0, x1, x2], [0, y1, y2])

    return line,

ani = FuncAnimation(
    plt.gcf(),
    update,
    frames=range(0, 91, 2),
    interval=50,
    repeat=True
)

plt.show()

