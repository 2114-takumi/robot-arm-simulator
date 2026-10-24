
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

# 目標座標
target_x = 1.2
target_y = 0.8

# 原点から目標位置までの距離
r = np.sqrt(target_x**2 + target_y**2)

# 到達可能か判定
reachable = abs(L1 - L2) <= r <= L1 + L2

print(f"Target distance: {r:.3f} m")
print(f"Reachable: {reachable}")


# 逆運動学：第2関節の角度を計算
cos_theta2 = (r**2 - L1**2 - L2**2) / (2 * L1 * L2)

theta2_ik = np.arccos(cos_theta2)

print(f"IK theta2: {np.rad2deg(theta2_ik):.2f} deg")


# 逆運動学：第1関節の角度を計算

alpha = np.arctan2(target_y, target_x)

beta = np.arctan2(
    L2 * np.sin(theta2_ik),
    L1 + L2 * np.cos(theta2_ik)
)

theta1_ik = alpha - beta

print(f"IK theta1: {np.rad2deg(theta1_ik):.2f} deg")



# 逆運動学の検証
x1_ik, y1_ik, x_ik, y_ik = forward_kinematics(
    theta1_ik, theta2_ik, L1, L2
)

# 目標位置と実際の先端位置の誤差
error_x = target_x - x_ik
error_y = target_y - y_ik

position_error = np.sqrt(error_x**2 + error_y**2)

print(f"IK End Effector: x={x_ik:.6f}, y={y_ik:.6f}")
print(f"Position Error: {position_error:.10f} m")


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

# 逆運動学で求めたアームの姿勢を表示
plt.plot(
    [0, x1_ik, x_ik],
    [0, y1_ik, y_ik],
    linestyle="--",
    marker="o",
    linewidth=2,
    label="IK Pose"
)

# 目標位置を表示
plt.scatter(
    target_x,
    target_y,
    marker="x",
    s=100,
    label="Target"
)

plt.legend()

plt.show()

