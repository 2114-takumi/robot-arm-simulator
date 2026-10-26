
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

# 2%整定時間を計算する関数
def calculate_settling_time(times, angles, target):
    initial_error = abs(target - angles[0])
    threshold = 0.02 * initial_error

    for i in range(len(angles)):
        # この時刻以降、すべて許容範囲内にあるか
        remaining_errors = np.abs(
            np.array(angles[i:]) - target
        )

        if np.all(remaining_errors <= threshold):
            return times[i]

    return None

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

J_target = calculate_jacobian(
    theta1_ik,
    theta2_ik,
    L1,
    L2
)

print("\n--- Jacobian at Target Position ---")
print(J_target)
print("Shape:", J_target.shape)



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
    theta1_new = theta1_pid_history[frame]
    theta2_new = after_history[frame]
    x1, y1, x2, y2 = forward_kinematics(
        theta1_new, theta2_new, L1, L2
    )
    line.set_data(
        [0, x1, x2],
        [0, y1, y2]
    )
    return line,

ani = FuncAnimation(
    plt.gcf(),
    update,
    frames=range(0, 201, 2),
    interval=100,
    repeat=False
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

# P制御の比例ゲイン
Kp = 2.0

# 現在角度と目標角度の誤差
error1 = theta1_ik - theta1
error2 = theta2_ik - theta2

# P制御による角速度指令
u1 = Kp * error1
u2 = Kp * error2

print(f"Joint 1 Error: {np.rad2deg(error1):.2f} deg")
print(f"Joint 2 Error: {np.rad2deg(error2):.2f} deg")
print(f"Joint 1 Command: {np.rad2deg(u1):.2f} deg/s")
print(f"Joint 2 Command: {np.rad2deg(u2):.2f} deg/s")

# 時間刻み（50ミリ秒）
dt = 0.05

# P制御による1ステップの角度更新
theta1_next = theta1 + u1 * dt
theta2_next = theta2 + u2 * dt

print(f"Next theta1: {np.rad2deg(theta1_next):.2f} deg")
print(f"Next theta2: {np.rad2deg(theta2_next):.2f} deg")



# P制御シミュレーション
theta1_current = theta1
theta2_current = theta2

# 時間と角度の履歴を保存
time_history = [0.0]
theta1_history = [theta1_current]
theta2_history = [theta2_current]

for step in range(100):
    # 現在角度と目標角度の誤差
    error1 = theta1_ik - theta1_current
    error2 = theta2_ik - theta2_current

    # P制御による角速度指令
    u1 = Kp * error1
    u2 = Kp * error2

    # 関節角度を更新
    theta1_current += u1 * dt
    theta2_current += u2 * dt

    # 更新後の時刻と角度を記録
    time_history.append((step + 1) * dt)
    theta1_history.append(theta1_current)
    theta2_history.append(theta2_current)


# 第2関節の2%整定時間を計算
settling_time = calculate_settling_time(
    time_history,
    theta2_history,
    theta2_ik
)

if settling_time is not None:
    print(f"Joint 2 Settling Time: {settling_time:.2f} s")
else:
    print("Joint 2 did not settle within the simulation time.")



print(f"P Control theta1: {np.rad2deg(theta1_current):.2f} deg")
print(f"P Control theta2: {np.rad2deg(theta2_current):.2f} deg")

# P制御の応答グラフ
fig, ax = plt.subplots()

ax.plot(
    time_history,
    np.rad2deg(theta1_history),
    label="Joint 1"
)

ax.plot(
    time_history,
    np.rad2deg(theta2_history),
    label="Joint 2"
)

# 目標角度を水平線で表示
ax.axhline(
    np.rad2deg(theta1_ik),
    linestyle="--",
    label="Joint 1 Target"
)

ax.axhline(
    np.rad2deg(theta2_ik),
    linestyle="--",
    label="Joint 2 Target"
)

ax.set_xlabel("Time [s]")
ax.set_ylabel("Joint Angle [deg]")
ax.set_title("P Control Response")
ax.grid(True)
ax.legend()


# 第2関節の簡易動力学モデル
J = 0.1        # 慣性モーメント [kg m^2]
b = 0.05       # 粘性摩擦係数 [N m s/rad]
Kp_torque = 1.0  # トルク制御用の比例ゲイン [N m/rad]
joint2_disturbance = -0.1  # P・PD・PID共通の外乱トルク [N m]

# 初期状態
theta2_dynamic = theta2
omega2_dynamic = 0.0

# 目標角度との誤差
error2_dynamic = theta2_ik - theta2_dynamic

# P制御によるトルク指令
tau2 = Kp_torque * error2_dynamic + joint2_disturbance

# 運動方程式から角加速度を計算
alpha2 = (tau2 - b * omega2_dynamic) / J

# 角速度を更新
omega2_dynamic += alpha2 * dt

# 関節角度を更新
theta2_dynamic += omega2_dynamic * dt

print(f"Torque: {tau2:.3f} N m")
print(f"Angular Acceleration: {alpha2:.3f} rad/s^2")
print(f"Angular Velocity: {omega2_dynamic:.3f} rad/s")
print(f"Dynamic theta2: {np.rad2deg(theta2_dynamic):.2f} deg")


# 慣性を考慮したP制御シミュレーション

# 初期状態を設定し直す
theta2_dynamic = theta2
omega2_dynamic = 0.0

# 履歴を保存するリスト
dynamic_time_history = [0.0]
dynamic_theta_history = [theta2_dynamic]

# 200ステップ（10秒間）計算
for step in range(200):

    # 目標角度との誤差
    error = theta2_ik - theta2_dynamic

    # P制御によるトルク
    tau = Kp_torque * error + joint2_disturbance

    # 運動方程式
    alpha = (tau - b * omega2_dynamic) / J

    # 角速度を更新
    omega2_dynamic += alpha * dt

    # 関節角度を更新
    theta2_dynamic += omega2_dynamic * dt

    # 時間と角度を記録
    dynamic_time_history.append((step + 1) * dt)
    dynamic_theta_history.append(theta2_dynamic)

# グラフを作成
fig_dynamic, ax_dynamic = plt.subplots()

ax_dynamic.plot(
    dynamic_time_history,
    np.rad2deg(dynamic_theta_history),
    label="Dynamic P Control"
)

# 目標角度
ax_dynamic.axhline(
    np.rad2deg(theta2_ik),
    linestyle="--",
    label="Target"
)

ax_dynamic.set_xlabel("Time [s]")
ax_dynamic.set_ylabel("Joint Angle [deg]")
ax_dynamic.set_title("P Control with Inertia")
ax_dynamic.grid(True)
ax_dynamic.legend()


# PD制御シミュレーション


Kd_torque = 0.5

# 初期状態
theta2_pd = theta2
omega2_pd = 0.0

# 角度履歴
pd_theta_history = [theta2_pd]

for step in range(200):

    # 角度誤差
    error = theta2_ik - theta2_pd

    # PD制御によるトルク指令
    tau = Kp_torque * error - Kd_torque * omega2_pd

    tau += joint2_disturbance


    # 運動方程式
    alpha = (tau - b * omega2_pd) / J

    # 角速度を更新
    omega2_pd += alpha * dt

    # 関節角度を更新
    theta2_pd += omega2_pd * dt

    # 角度履歴を保存
    pd_theta_history.append(theta2_pd)

# 既存のP制御グラフにPD制御を追加
ax_dynamic.plot(
    dynamic_time_history,
    np.rad2deg(pd_theta_history),
    label="PD Control"
)

ax_dynamic.set_title("P vs PD Control with Inertia")
ax_dynamic.legend()


# P制御とPD制御の減衰比を計算

zeta_p = b / (2 * np.sqrt(J * Kp_torque))

zeta_pd = (b + Kd_torque) / (
    2 * np.sqrt(J * Kp_torque)
)

# 臨界減衰となるDゲイン
Kd_critical = 2 * np.sqrt(J * Kp_torque) - b

print(f"P Control Damping Ratio: {zeta_p:.3f}")
print(f"PD Control Damping Ratio: {zeta_pd:.3f}")
print(f"Critical Kd: {Kd_critical:.3f}")



# 慣性を考慮したP制御の整定時間
ts_p = calculate_settling_time(
    dynamic_time_history,
    dynamic_theta_history,
    theta2_ik
)

# PD制御の整定時間
ts_pd = calculate_settling_time(
    dynamic_time_history,
    pd_theta_history,
    theta2_ik
)

if ts_p is None:
    print("Dynamic P: Not settled within 10 s")
else:
    print(f"Dynamic P Settling Time: {ts_p:.2f} s")

if ts_pd is None:
    print("PD: Not settled within 10 s")
else:
    print(f"PD Settling Time: {ts_pd:.2f} s")



# PID制御のゲイン
Ki_torque = 0.5

# 初期状態
theta2_pid = theta2
omega2_pid = 0.0
integral_error = 0.0

pid_theta_history = [theta2_pid]

# PID制御シミュレーション
for step in range(200):
    error = theta2_ik - theta2_pid

    # I制御：誤差を時間積分する
    integral_error += error * dt

    # PID制御によるトルク
    tau = (
        Kp_torque * error
        + Ki_torque * integral_error
        - Kd_torque * omega2_pid
    )

    # PD制御と同じ外乱を加える
    tau += joint2_disturbance

    # 運動方程式
    alpha = (tau - b * omega2_pid) / J

    # 角速度・角度の更新
    omega2_pid += alpha * dt
    theta2_pid += omega2_pid * dt

    pid_theta_history.append(theta2_pid)

# PID制御の結果を既存グラフに追加
ax_dynamic.plot(
    dynamic_time_history,
    np.rad2deg(pid_theta_history),
    label="PID Control"
)
ax_dynamic.legend()

# 最終角度と誤差
print(f"PID Final Angle: {np.rad2deg(theta2_pid):.2f} deg")
print(f"PID Final Error: {np.rad2deg(theta2_ik - theta2_pid):.2f} deg")


# PID制御の2%整定時間
ts_pid = calculate_settling_time(
    dynamic_time_history,
    pid_theta_history,
    theta2_ik
)

if ts_pid is None:
    print("PID: Not settled within 10 s")
else:
    print(f"PID Settling Time: {ts_pid:.2f} s")


# PID制御の最大角度
max_theta_pid = np.max(pid_theta_history)

# 目標角度からのオーバーシュート
overshoot_pid = max(0.0, max_theta_pid - theta2_ik)

print(f"PID Maximum Angle: {np.rad2deg(max_theta_pid):.2f} deg")
print(f"PID Overshoot: {np.rad2deg(overshoot_pid):.2f} deg")


# Joint 1 PID parameters
J1 = 0.1
b1 = 0.05

Kp1 = 1.2
Ki1 = 0.3
Kd1 = 0.4

# Joint 1 initial conditions
theta1_pid = theta1
omega1_pid = 0.0
integral1 = 0.0

print("\n--- Joint 1 Initial State ---")
print(f"Initial Angle: {np.rad2deg(theta1_pid):.2f} deg")
print(f"Target Angle: {np.rad2deg(theta1_ik):.2f} deg")
print(f"Initial Velocity: {omega1_pid:.2f} rad/s")
print(f"Integral Error: {integral1:.2f}")

# Joint 1 angle history
theta1_pid_history = [theta1_pid]


# Joint 1 PID simulation
disturbance1 = -0.1

for step in range(200):

    # Calculate angle error
    error1_pid = theta1_ik - theta1_pid

    # Update integral error
    integral1 += error1_pid * dt

    # Calculate PID torque
    tau1 = (
        Kp1 * error1_pid
        + Ki1 * integral1
        - Kd1 * omega1_pid
    )

    # Add external disturbance
    tau1 += disturbance1

    # Calculate angular acceleration
    alpha1 = (tau1 - b1 * omega1_pid) / J1

    # Update angular velocity
    omega1_pid += alpha1 * dt

    # Update joint angle
    theta1_pid += omega1_pid * dt

    # Save joint angle
    theta1_pid_history.append(theta1_pid)

print("\n--- Joint 1 PID Result ---")
print(f"Final Angle: {np.rad2deg(theta1_pid):.2f} deg")
print(f"Target Angle: {np.rad2deg(theta1_ik):.2f} deg")
print(
    f"Final Error: "
    f"{np.rad2deg(theta1_ik - theta1_pid):.2f} deg"
)
print(f"History Length: {len(theta1_pid_history)}")


def simulate_pid(kd, kp=Kp_torque, ki=Ki_torque):
    theta = theta2
    omega = 0.0
    integral = 0.0

    history = [theta]

    for step in range(200):
        error = theta2_ik - theta

        integral += error * dt

        tau = (
            kp * error
            + ki * integral
            - kd * omega
        )

        tau += joint2_disturbance

        alpha = (tau - b * omega) / J

        omega += alpha * dt
        theta += omega * dt

        history.append(theta)

    return history


test_history = simulate_pid(0.5)

print(
    "Simulation matches:",
    np.allclose(test_history, pid_theta_history)
)

kd_values = np.linspace(0.40, 0.55, 16)

best_kd = None
min_overshoot = float("inf")
best_cost_kd = None
min_cost = float("inf")

print("\n--- PID Gain Comparison ---")

for kd in kd_values:
    history = simulate_pid(kd)

    max_angle = np.max(history)
    overshoot = max(0.0, max_angle - theta2_ik)

    if overshoot < min_overshoot:
        min_overshoot = overshoot
        best_kd = kd

    settling_time = calculate_settling_time(
        dynamic_time_history,
        history,
        theta2_ik
    )

    if settling_time is None:
        cost = float("inf")
    else:
        overshoot_deg = np.rad2deg(overshoot)

        cost = (
            0.5 * settling_time / 5.0
            + 0.5 * overshoot_deg / 20.0
        )

    if cost < min_cost:
        min_cost = cost
        best_cost_kd = kd

    print(f"Kd={kd:.3f} | Cost={cost:.3f}")

    if settling_time is None:
        settling_text = "Not settled"
    else:
        settling_text = f"{settling_time:.2f} s"

    print(
        f"Kd={kd:.3f} | "
        f"Overshoot={np.rad2deg(overshoot):.2f} deg | "
        f"Settling Time={settling_text}"
    )

print("\n--- Best D Gain ---")
print(f"Best Kd: {best_kd:.3f}")
print(f"Minimum Overshoot: {np.rad2deg(min_overshoot):.2f} deg")

print("\n--- Best D Gain by Cost ---")
if best_cost_kd is None:
    print("Best Kd by Cost: None (no settled result)")
else:
    print(f"Best Kd by Cost: {best_cost_kd:.3f}")
print(f"Minimum Cost: {min_cost:.3f}")



kp_values = [0.8, 1.0, 1.2]
kd_values_2d = [0.4, 0.5, 0.6]

best_kp = None
best_kd_2d = None
min_cost_2d = float("inf")

print("\n--- P and D Gain Comparison ---")

for kp in kp_values:
    for kd in kd_values_2d:
        history = simulate_pid(kd, kp)

        max_angle = np.max(history)
        overshoot = max(0.0, max_angle - theta2_ik)

        settling_time = calculate_settling_time(
            dynamic_time_history,
            history,
            theta2_ik
        )

        if settling_time is None:
            cost = float("inf")
        else:
            overshoot_deg = np.rad2deg(overshoot)
            cost = (
                0.5 * settling_time / 5.0
                + 0.5 * overshoot_deg / 20.0
            )

        if cost < min_cost_2d:
            min_cost_2d = cost
            best_kp = kp
            best_kd_2d = kd

        if settling_time is None:
            settling_text = "Not settled"
        else:
            settling_text = f"{settling_time:.2f} s"

        print(
            f"Kp={kp:.1f} | "
            f"Kd={kd:.1f} | "
            f"Overshoot={np.rad2deg(overshoot):.2f} deg | "
            f"Settling Time={settling_text}"
        )

print("\n--- Best P and D Gains ---")
if best_kp is None:
    print("Best Kp: None (no settled result)")
    print("Best Kd: None (no settled result)")
else:
    print(f"Best Kp: {best_kp:.2f}")
    print(f"Best Kd: {best_kd_2d:.2f}")
print(f"Minimum Cost: {min_cost_2d:.3f}")


kp_values_3d = [0.8, 1.0, 1.2]
ki_values_3d = [0.3, 0.5, 0.7]
kd_values_3d = [0.4, 0.5, 0.6]

best_gains_3d = None
min_cost_3d = float("inf")

print("\n--- PID 3D Gain Optimization ---")

for kp in kp_values_3d:
    for ki in ki_values_3d:
        for kd in kd_values_3d:

            history = simulate_pid(kd, kp, ki)

            max_angle = np.max(history)
            overshoot = max(0.0, max_angle - theta2_ik)

            settling_time = calculate_settling_time(
                dynamic_time_history,
                history,
                theta2_ik
            )

            if settling_time is None:
                cost = float("inf")
            else:
                overshoot_deg = np.rad2deg(overshoot)

                cost = (
                    0.5 * settling_time / 5.0
                    + 0.5 * overshoot_deg / 20.0
                )

            if cost < min_cost_3d:
                min_cost_3d = cost
                best_gains_3d = (kp, ki, kd)

            print(
                f"Kp={kp:.1f} | "
                f"Ki={ki:.1f} | "
                f"Kd={kd:.1f} | "
                f"Cost={cost:.3f}"
            )

print("\n--- Best PID Gains ---")

if best_gains_3d is None:
    print("No settling combination found.")
else:
    print(f"Best Kp: {best_gains_3d[0]:.2f}")
    print(f"Best Ki: {best_gains_3d[1]:.2f}")
    print(f"Best Kd: {best_gains_3d[2]:.2f}")
    print(f"Minimum Cost: {min_cost_3d:.3f}")


if best_gains_3d is not None:
    best_kp, best_ki, best_kd = best_gains_3d

    best_history = simulate_pid(
        best_kd, best_kp, best_ki
    )

    best_max_angle = np.max(best_history)

    best_overshoot = max(
        0.0,
        best_max_angle - theta2_ik
    )

    best_settling_time = calculate_settling_time(
        dynamic_time_history,
        best_history,
        theta2_ik
    )

    best_final_error = theta2_ik - best_history[-1]

    print("\n--- Optimized PID Performance ---")
    print(
        f"Overshoot: "
        f"{np.rad2deg(best_overshoot):.2f} deg"
    )
    print(
        f"Settling Time: "
        f"{best_settling_time:.2f} s"
        if best_settling_time is not None
        else "Settling Time: Not settled"
    )
    print(
        f"Final Error: "
        f"{np.rad2deg(best_final_error):.2f} deg"
    )


# 最適化前のPID制御
before_history = simulate_pid(
    kd=0.5,
    kp=1.0,
    ki=0.5
)

# 最適化後のPID制御
after_history = simulate_pid(
    kd=best_gains_3d[2],
    kp=best_gains_3d[0],
    ki=best_gains_3d[1]
)

fig_compare, ax_compare = plt.subplots()

ax_compare.plot(
    dynamic_time_history,
    np.rad2deg(before_history),
    label="Before Optimization"
)

ax_compare.plot(
    dynamic_time_history,
    np.rad2deg(after_history),
    label="After Optimization"
)

ax_compare.axhline(
    np.rad2deg(theta2_ik),
    color="red",
    linestyle="--",
    label="Target"
)

ax_compare.set_xlabel("Time [s]")
ax_compare.set_ylabel("Joint 2 Angle [deg]")
ax_compare.set_title("PID Control: Before vs After Optimization")
ax_compare.grid(True)
ax_compare.legend()


# 初期誤差の2%を計算
tolerance = 0.02 * abs(theta2_ik - theta2)

# 整定範囲の上下限
lower_limit = np.rad2deg(theta2_ik - tolerance)
upper_limit = np.rad2deg(theta2_ik + tolerance)

# 整定範囲をグラフに表示
ax_compare.axhspan(
    lower_limit,
    upper_limit,
    color="green",
    alpha=0.15,
    label="2% Settling Band"
)

# 最適化前の整定時間
before_settling_time = calculate_settling_time(
    dynamic_time_history,
    before_history,
    theta2_ik
)

# 最適化後の整定時間
after_settling_time = calculate_settling_time(
    dynamic_time_history,
    after_history,
    theta2_ik
)

# 整定時間を縦線で表示
if before_settling_time is not None:
    ax_compare.axvline(
        before_settling_time,
        color="blue",
        linestyle=":",
        label=f"Before Settling: {before_settling_time:.2f} s"
    )

if after_settling_time is not None:
    ax_compare.axvline(
        after_settling_time,
        color="orange",
        linestyle=":",
        label=f"After Settling: {after_settling_time:.2f} s"
    )

ax_compare.legend()

# Bring Figure 1 (robot arm animation) to the front
fig1 = plt.figure(1)

window = fig1.canvas.manager.window
window.lift()
window.attributes("-topmost", True)
window.attributes("-topmost", False)


# Final joint angles from PID simulation
final_theta1 = theta1_pid_history[-1]
final_theta2 = after_history[-1]

# Calculate final end-effector position
x1, y1, final_x, final_y = forward_kinematics(
    final_theta1,
    final_theta2,
    L1,
    L2
)

# Calculate position error
position_error = np.hypot(
    target_x - final_x,
    target_y - final_y
)

print("\n--- End Effector Position Error ---")
print(f"Target: ({target_x:.4f}, {target_y:.4f}) m")
print(f"Actual: ({final_x:.4f}, {final_y:.4f}) m")
print(f"Position Error: {position_error:.6f} m")
print(f"Position Error: {position_error * 1000:.2f} mm")



# End-effector position error history
fixed_target_error_history = []

for theta1_now, theta2_now in zip(
    theta1_pid_history,
    after_history
):
    # Forward kinematics
    x1, y1, x2, y2 = forward_kinematics(
        theta1_now,
        theta2_now,
        L1,
        L2
    )

    # Position error [m]
    error = np.hypot(
        target_x - x2,
        target_y - y2
    )

    # Convert to millimeters
    fixed_target_error_history.append(error * 1000)

# Position tolerance [mm]
position_tolerance = 20.0
position_settling_time = None

for i in range(len(fixed_target_error_history)):
    remaining_errors = fixed_target_error_history[i:]

    if np.all(
        np.array(remaining_errors) <= position_tolerance
    ):
        position_settling_time = dynamic_time_history[i]
        break

print("\n--- End-Effector Settling Time ---")

if position_settling_time is not None:
    print(
        f"Position Settling Time: "
        f"{position_settling_time:.2f} s"
    )
else:
    print("Position did not settle within 10 s")

# Plot position error
fig_error, ax_error = plt.subplots()

ax_error.plot(
    dynamic_time_history,
    fixed_target_error_history,
    label="End-Effector Position Error"
)

ax_error.set_xlabel("Time [s]")
ax_error.set_ylabel("Position Error [mm]")
ax_error.set_title("End-Effector Position Error vs Time")
ax_error.grid(True)

ax_error.axhline(
    position_tolerance,
    color="red",
    linestyle="--",
    label="20 mm Tolerance"
)

if position_settling_time is not None:
    ax_error.axvline(
        position_settling_time,
        color="green",
        linestyle=":",
        label=f"Settling: {position_settling_time:.2f} s"
    )

ax_error.legend()

print(
    f"Final Position Error: "
    f"{fixed_target_error_history[-1]:.2f} mm"
)

# Final joint angle errors [rad]
delta_theta1 = theta1_pid_history[-1] - theta1_ik
delta_theta2 = after_history[-1] - theta2_ik

# Joint angle error vector
delta_theta = np.array([
    delta_theta1,
    delta_theta2
])

# Predict end-effector position error using Jacobian
delta_position = J_target @ delta_theta

# Calculate predicted position error magnitude
predicted_error = np.linalg.norm(delta_position)

print("\n--- Jacobian Position Error Prediction ---")
print(f"Joint 1 Error: {np.rad2deg(delta_theta1):.4f} deg")
print(f"Joint 2 Error: {np.rad2deg(delta_theta2):.4f} deg")
print(f"Predicted X Error: {delta_position[0] * 1000:.2f} mm")
print(f"Predicted Y Error: {delta_position[1] * 1000:.2f} mm")
print(f"Predicted Position Error: {predicted_error * 1000:.2f} mm")
print(f"Actual Position Error: {position_error * 1000:.2f} mm")


# Actual position error vector [m]
actual_delta_position = np.array([
    final_x - target_x,
    final_y - target_y
])

# Difference between actual and predicted position errors
approximation_error = (
    actual_delta_position - delta_position
)

# Magnitude of approximation error [mm]
approximation_error_mm = (
    np.linalg.norm(approximation_error) * 1000
)

print("\n--- Jacobian Approximation Accuracy ---")
print(
    f"Actual X Error: "
    f"{actual_delta_position[0] * 1000:.4f} mm"
)
print(
    f"Actual Y Error: "
    f"{actual_delta_position[1] * 1000:.4f} mm"
)
print(
    f"Predicted X Error: "
    f"{delta_position[0] * 1000:.4f} mm"
)
print(
    f"Predicted Y Error: "
    f"{delta_position[1] * 1000:.4f} mm"
)
print(
    f"Approximation Error: "
    f"{approximation_error_mm:.4f} mm"
)


# Test larger joint angle errors
test_delta_theta = np.deg2rad(
    np.array([10.0, -10.0])
)

# New joint angles
test_theta1 = theta1_ik + test_delta_theta[0]
test_theta2 = theta2_ik + test_delta_theta[1]

# Actual end-effector position
_, _, test_x, test_y = forward_kinematics(
    test_theta1,
    test_theta2,
    L1,
    L2
)

# Actual position displacement
test_actual_delta = np.array([
    test_x - target_x,
    test_y - target_y
])

# Jacobian prediction
test_predicted_delta = J_target @ test_delta_theta

# Approximation error
test_approximation_error = np.linalg.norm(
    test_actual_delta - test_predicted_delta
)

print("\n--- Large Angle Error Test ---")
print(f"Joint 1 Change: +10.00 deg")
print(f"Joint 2 Change: -10.00 deg")
print(
    f"Actual Displacement: "
    f"{np.linalg.norm(test_actual_delta) * 1000:.2f} mm"
)
print(
    f"Predicted Displacement: "
    f"{np.linalg.norm(test_predicted_delta) * 1000:.2f} mm"
)
print(
    f"Approximation Error: "
    f"{test_approximation_error * 1000:.2f} mm"
)

# Second-order Taylor approximation of the Jacobian error
second_order_error = 0.5 * L1 * test_delta_theta[0]**2

print("\n--- Second-Order Error Prediction ---")
print(f"Second-Order Prediction: {second_order_error * 1000:.4f} mm")
print(f"Measured Approximation Error: {test_approximation_error * 1000:.4f} mm")

# Select the simulation time
k = 20  # t = 20 * 0.05 = 1.0 s

# Joint angles at this time
theta1_now = theta1_pid_history[k]
theta2_now = after_history[k]

# Approximate joint angular velocities [rad/s]
omega1_now = (
    theta1_pid_history[k + 1] - theta1_pid_history[k]
) / dt
omega2_now = (
    after_history[k + 1] - after_history[k]
) / dt

# Angular velocity vector
omega_vector = np.array([omega1_now, omega2_now])

# Jacobian at the current joint angles
J_now = calculate_jacobian(
    theta1_now, theta2_now, L1, L2
)

# Predict end-effector velocity
velocity = J_now @ omega_vector

print("\n--- End-Effector Velocity ---")
print(f"Time: {k * dt:.2f} s")
print(f"Joint 1 Angular Velocity: {omega1_now:.4f} rad/s")
print(f"Joint 2 Angular Velocity: {omega2_now:.4f} rad/s")
print(f"End-Effector Vx: {velocity[0]:.4f} m/s")
print(f"End-Effector Vy: {velocity[1]:.4f} m/s")
print(f"End-Effector Speed: {np.linalg.norm(velocity):.4f} m/s")


# End-effector position at time k
_, _, x_now, y_now = forward_kinematics(
    theta1_pid_history[k],
    after_history[k],
    L1,
    L2
)

# End-effector position at time k+1
_, _, x_next, y_next = forward_kinematics(
    theta1_pid_history[k + 1],
    after_history[k + 1],
    L1,
    L2
)

# Velocity calculated from position differences
velocity_fd = np.array([
    (x_next - x_now) / dt,
    (y_next - y_now) / dt
])

# Difference between the two velocity calculations
velocity_difference = np.linalg.norm(
    velocity_fd - velocity
)

print("\n--- Velocity Calculation Comparison ---")
print(f"Jacobian Vx: {velocity[0]:.6f} m/s")
print(f"Finite Difference Vx: {velocity_fd[0]:.6f} m/s")
print(f"Jacobian Vy: {velocity[1]:.6f} m/s")
print(f"Finite Difference Vy: {velocity_fd[1]:.6f} m/s")
print(f"Velocity Difference: {velocity_difference:.6f} m/s")

print("\n--- Time Interval Accuracy Test ---")

# Joint angle changes during one simulation step
angle_change = np.array([
    theta1_pid_history[k + 1] - theta1_pid_history[k],
    after_history[k + 1] - after_history[k]
])

# Test different numbers of subdivisions
for n in [1, 2, 5, 10, 20]:
    # Smaller time interval
    dt_sub = dt / n

    # Interpolated joint angles after dt_sub
    theta1_sub = theta1_now + angle_change[0] / n
    theta2_sub = theta2_now + angle_change[1] / n

    # End-effector position after dt_sub
    _, _, x_sub, y_sub = forward_kinematics(
        theta1_sub, theta2_sub, L1, L2
    )

    # Finite-difference velocity
    velocity_sub = np.array([
        (x_sub - x_now) / dt_sub,
        (y_sub - y_now) / dt_sub
    ])

    # Difference from Jacobian velocity
    error_sub = np.linalg.norm(
        velocity_sub - velocity
    )

    print(
        f"Subdivisions: {n:2d}, dt: {dt_sub:.4f} s, "
        f"Velocity Error: {error_sub:.8f} m/s"
    )


# Desired end-effector velocity [m/s]
desired_velocity = np.array([0.1, 0.0])

# Calculate required joint angular velocities [rad/s]
joint_velocity_command = np.linalg.solve(
    J_target,
    desired_velocity
)

# Verify the result
calculated_velocity = J_target @ joint_velocity_command

print("\n--- Inverse Velocity Kinematics ---")
print(f"Desired Vx: {desired_velocity[0]:.4f} m/s")
print(f"Desired Vy: {desired_velocity[1]:.4f} m/s")
print(f"Joint 1 Velocity: {joint_velocity_command[0]:.4f} rad/s")
print(f"Joint 2 Velocity: {joint_velocity_command[1]:.4f} rad/s")
print(f"Calculated Vx: {calculated_velocity[0]:.4f} m/s")
print(f"Calculated Vy: {calculated_velocity[1]:.4f} m/s")

# Duration of velocity command [s]
dt_test = 0.05

# Initial joint angles (target configuration)
theta1_start = theta1_ik
theta2_start = theta2_ik

# Update joint angles using commanded velocities
theta1_end = (
    theta1_start + joint_velocity_command[0] * dt_test
)

theta2_end = (
    theta2_start + joint_velocity_command[1] * dt_test
)

# Initial end-effector position
_, _, x_start, y_start = forward_kinematics(
    theta1_start, theta2_start, L1, L2
)

# End-effector position after 0.05 seconds
_, _, x_end, y_end = forward_kinematics(
    theta1_end, theta2_end, L1, L2
)

# Actual displacement
actual_displacement = np.array([
    x_end - x_start,
    y_end - y_start
])

# Predicted displacement
predicted_displacement = desired_velocity * dt_test

print("\n--- Inverse Velocity Motion Test ---")
print(f"Duration: {dt_test:.3f} s")
print(f"Predicted X: {predicted_displacement[0] * 1000:.4f} mm")
print(f"Actual X: {actual_displacement[0] * 1000:.4f} mm")
print(f"Predicted Y: {predicted_displacement[1] * 1000:.4f} mm")
print(f"Actual Y: {actual_displacement[1] * 1000:.4f} mm")

# --- Straight-Line Motion Using Updated Jacobian ---

# Initial joint angles
theta1_line = theta1_ik
theta2_line = theta2_ik

# Simulation settings
line_dt = 0.05
line_steps = 20

# Save end-effector positions
line_x_history = []
line_y_history = []

# Initial position
_, _, x_line, y_line = forward_kinematics(
    theta1_line, theta2_line, L1, L2
)

line_x_history.append(x_line)
line_y_history.append(y_line)

# Update Jacobian and joint angles at every step
for step in range(line_steps):

    # Jacobian at the current configuration
    J_line = calculate_jacobian(
        theta1_line, theta2_line, L1, L2
    )

    # Joint velocities required for horizontal motion
    omega_line = np.linalg.solve(
        J_line, desired_velocity
    )

    # Update joint angles
    theta1_line += omega_line[0] * line_dt
    theta2_line += omega_line[1] * line_dt

    # Calculate new end-effector position
    _, _, x_line, y_line = forward_kinematics(
        theta1_line, theta2_line, L1, L2
    )

    # Save trajectory
    line_x_history.append(x_line)
    line_y_history.append(y_line)

# Ideal final position
ideal_x = target_x + desired_velocity[0] * line_steps * line_dt
ideal_y = target_y + desired_velocity[1] * line_steps * line_dt

# Final position error
line_final_error = np.hypot(
    x_line - ideal_x,
    y_line - ideal_y
)

print("\n--- Updated Jacobian Straight-Line Motion ---")
print(f"Duration: {line_steps * line_dt:.2f} s")
print(f"Ideal Final Position: ({ideal_x:.6f}, {ideal_y:.6f}) m")
print(f"Actual Final Position: ({x_line:.6f}, {y_line:.6f}) m")
print(f"Final Position Error: {line_final_error * 1000:.4f} mm")
print(f"Maximum Y Deviation: {np.max(np.abs(np.array(line_y_history) - target_y)) * 1000:.4f} mm")

# --- Straight-Line Tracking Error Visualization ---

# Convert trajectory history to NumPy arrays
line_x_array = np.array(line_x_history)
line_y_array = np.array(line_y_history)

# Calculate vertical deviation from the ideal path [mm]
y_deviation_mm = (line_y_array - target_y) * 1000

# Create a new figure
fig_line, ax_line = plt.subplots()

# Actual vertical deviation
ax_line.plot(
    line_x_array,
    y_deviation_mm,
    color="blue",
    label="Actual Y Deviation"
)

# Ideal horizontal trajectory
ax_line.axhline(
    y=0,
    color="red",
    linestyle="--",
    label="Ideal Trajectory"
)

# Labels and title
ax_line.set_xlabel("End-Effector X Position [m]")
ax_line.set_ylabel("Y Deviation [mm]")
ax_line.set_title("Straight-Line Tracking Error")

ax_line.grid(True)
ax_line.legend()
print("\n--- Time Step vs Tracking Accuracy ---")

# Test different time steps
for test_dt in [0.05, 0.025, 0.01, 0.005]:

    # Reset joint angles to the target configuration
    theta1_test = theta1_ik
    theta2_test = theta2_ik

    # Keep the total duration at 1 second
    steps = int(round(1.0 / test_dt))

    # Simulate straight-line motion
    for step in range(steps):

        # Jacobian at the current configuration
        J_test = calculate_jacobian(
            theta1_test, theta2_test, L1, L2
        )

        # Required joint angular velocities
        omega_test = np.linalg.solve(
            J_test, desired_velocity
        )

        # Update joint angles
        theta1_test += omega_test[0] * test_dt
        theta2_test += omega_test[1] * test_dt

    # Calculate final end-effector position
    _, _, x_test, y_test = forward_kinematics(
        theta1_test, theta2_test, L1, L2
    )

    # Calculate final position error
    error_test = np.hypot(
        x_test - ideal_x,
        y_test - ideal_y
    )

    print(
        f"dt: {test_dt:.3f} s, "
        f"Steps: {steps:3d}, "
        f"Position Error: {error_test * 1000:.4f} mm"
    )


# --- RK2 Midpoint Method Test ---

# Initial joint angles
theta_rk2 = np.array([
    theta1_ik,
    theta2_ik
], dtype=float)

# Use the same time step as the original Euler method
rk2_dt = 0.05
rk2_steps = 20

for step in range(rk2_steps):

    # Step 1: Joint velocities at the current configuration
    J_current = calculate_jacobian(
        theta_rk2[0], theta_rk2[1], L1, L2
    )

    omega_current = np.linalg.solve(
        J_current, desired_velocity
    )

    # Step 2: Predict the midpoint configuration
    theta_mid = theta_rk2 + 0.5 * rk2_dt * omega_current

    # Step 3: Joint velocities at the midpoint
    J_mid = calculate_jacobian(
        theta_mid[0], theta_mid[1], L1, L2
    )

    omega_mid = np.linalg.solve(
        J_mid, desired_velocity
    )

    # Step 4: Update using midpoint velocities
    theta_rk2 += rk2_dt * omega_mid

# Calculate final end-effector position
_, _, x_rk2, y_rk2 = forward_kinematics(
    theta_rk2[0], theta_rk2[1], L1, L2
)

# Calculate final position error
rk2_error = np.hypot(
    x_rk2 - ideal_x,
    y_rk2 - ideal_y
)

print("\n--- Euler vs RK2 Comparison ---")
print(f"Time Step: {rk2_dt:.3f} s")
print(f"Steps: {rk2_steps}")
print(f"Euler Error: {line_final_error * 1000:.6f} mm")
print(f"RK2 Error: {rk2_error * 1000:.6f} mm")
print(f"RK2 Final Position: ({x_rk2:.6f}, {y_rk2:.6f}) m")

# --- RK2 Time Step Accuracy Test ---

def simulate_rk2(test_dt):
    theta = np.array([theta1_ik, theta2_ik], dtype=float)
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


print("\n--- RK2 Time Step Accuracy Test ---")

previous_error = None

for test_dt in [0.05, 0.025, 0.0125]:
    error = simulate_rk2(test_dt)

    print(f"dt = {test_dt:.4f} s, Error = {error:.8f} mm")

    if previous_error is not None:
        print(f"Error Reduction Ratio = {previous_error / error:.3f}")

    previous_error = error

print("\n--- Inverse Kinematics Function Test ---")

for x, y in [(1.2, 0.8), (1.3, 0.8)]:
    th1, th2 = inverse_kinematics(x, y, L1, L2)
    _, _, actual_x, actual_y = forward_kinematics(
        th1, th2, L1, L2
    )
    error = np.hypot(actual_x - x, actual_y - y)

    print(f"Target: ({x:.3f}, {y:.3f}) m")
    print(
        f"Joint Angles: "
        f"({np.rad2deg(th1):.3f}, "
        f"{np.rad2deg(th2):.3f}) deg"
    )
    print(f"Position Error: {error * 1000:.6f} mm")

# --- Desired Joint Trajectory Generation ---

# Trajectory settings
trajectory_dt = 0.05
trajectory_time = np.linspace(0.0, 1.0, 21)

# Desired joint angle histories
theta1_desired_history = []
theta2_desired_history = []

for t in trajectory_time:

    # Desired end-effector position
    x_desired = 1.2 + 0.1 * t
    y_desired = 0.8

    # Convert desired position to joint angles
    theta1_desired, theta2_desired = inverse_kinematics(
        x_desired, y_desired, L1, L2
    )

    # Save desired joint angles
    theta1_desired_history.append(theta1_desired)
    theta2_desired_history.append(theta2_desired)

print("\n--- Desired Joint Trajectory ---")
print(f"Number of Points: {len(trajectory_time)}")

for i in [0, 10, 20]:
    print(
        f"Time: {trajectory_time[i]:.2f} s | "
        f"Theta1: {np.rad2deg(theta1_desired_history[i]):.3f} deg | "
        f"Theta2: {np.rad2deg(theta2_desired_history[i]):.3f} deg"
    )

# --- Desired Joint Trajectory Plot ---

plt.figure(figsize=(9, 5))

# First joint angle
plt.plot(
    trajectory_time,
    np.rad2deg(theta1_desired_history),
    label="Joint 1 Desired Angle",
    linewidth=2
)

# Second joint angle
plt.plot(
    trajectory_time,
    np.rad2deg(theta2_desired_history),
    label="Joint 2 Desired Angle",
    linewidth=2
)

plt.xlabel("Time [s]")
plt.ylabel("Joint Angle [deg]")
plt.title("Desired Joint Angle Trajectories")

plt.grid(True)
plt.legend()
plt.tight_layout()

# --- PID Trajectory Tracking ---

# PID gains
tracking_kp = 1.2
tracking_ki = 0.3
tracking_kd = 0.4

# Simplified joint dynamics
tracking_inertia = 0.1
tracking_damping = 0.05

# Initial joint angles: start at the first desired pose
tracking_theta = np.array([
    theta1_desired_history[0],
    theta2_desired_history[0]
], dtype=float)

# Initial angular velocities
tracking_omega = np.zeros(2)

# PID internal states
tracking_integral = np.zeros(2)
tracking_previous_error = np.zeros(2)

# Save actual joint angles
theta1_actual_history = [tracking_theta[0]]
theta2_actual_history = [tracking_theta[1]]

# Simulate 20 intervals
for k in range(len(trajectory_time) - 1):

    dt = trajectory_time[k + 1] - trajectory_time[k]

    # Desired joint angles at current time
    desired_theta = np.array([
        theta1_desired_history[k],
        theta2_desired_history[k]
    ])

    # PID error
    error = desired_theta - tracking_theta

    # Integral term
    tracking_integral += error * dt

    # Derivative term
    error_derivative = (
        error - tracking_previous_error
    ) / dt

    # PID output: joint torques
    torque = (
        tracking_kp * error
        + tracking_ki * tracking_integral
        + tracking_kd * error_derivative
    )

    # Joint angular acceleration
    acceleration = (
        torque - tracking_damping * tracking_omega
    ) / tracking_inertia

    # Semi-implicit Euler integration
    tracking_omega += acceleration * dt
    tracking_theta += tracking_omega * dt

    tracking_previous_error = error.copy()

    # Save actual angles
    theta1_actual_history.append(tracking_theta[0])
    theta2_actual_history.append(tracking_theta[1])

# Final target angles
final_desired_theta = np.array([
    theta1_desired_history[-1],
    theta2_desired_history[-1]
])

# Final angular errors
final_angle_error = final_desired_theta - tracking_theta

# Final end-effector position
_, _, actual_x, actual_y = forward_kinematics(
    tracking_theta[0], tracking_theta[1], L1, L2
)

final_position_error = np.hypot(
    actual_x - 1.3,
    actual_y - 0.8
)

print("\n--- PID Trajectory Tracking ---")
print(
    "Final Desired Angles:",
    np.rad2deg(final_desired_theta)
)
print(
    "Final Actual Angles:",
    np.rad2deg(tracking_theta)
)
print(
    "Final Angular Errors:",
    np.rad2deg(final_angle_error)
)
print(
    f"Final Position Error: "
    f"{final_position_error * 1000:.4f} mm"
)

# --- PID Joint Trajectory Tracking Plot ---

fig, axes = plt.subplots(2, 1, figsize=(9, 8))

# Joint 1
axes[0].plot(
    trajectory_time,
    np.rad2deg(theta1_desired_history),
    label="Desired Joint 1",
    linestyle="--",
    linewidth=2
)

axes[0].plot(
    trajectory_time,
    np.rad2deg(theta1_actual_history),
    label="Actual Joint 1",
    linewidth=2
)

axes[0].set_title("Joint 1 Trajectory Tracking")
axes[0].set_ylabel("Joint Angle [deg]")
axes[0].grid(True)
axes[0].legend()

# Joint 2
axes[1].plot(
    trajectory_time,
    np.rad2deg(theta2_desired_history),
    label="Desired Joint 2",
    linestyle="--",
    linewidth=2
)

axes[1].plot(
    trajectory_time,
    np.rad2deg(theta2_actual_history),
    label="Actual Joint 2",
    linewidth=2
)

axes[1].set_title("Joint 2 Trajectory Tracking")
axes[1].set_xlabel("Time [s]")
axes[1].set_ylabel("Joint Angle [deg]")
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()

# --- End-Effector Trajectory Tracking Error ---

trajectory_pid_error_history = []

for k, t in enumerate(trajectory_time):

    # Desired end-effector position
    x_desired = 1.2 + 0.1 * t
    y_desired = 0.8

    # Actual joint angles
    theta1_actual = theta1_actual_history[k]
    theta2_actual = theta2_actual_history[k]

    # Actual end-effector position
    _, _, x_actual, y_actual = forward_kinematics(
        theta1_actual, theta2_actual, L1, L2
    )

    # Position error [mm]
    error_mm = np.hypot(
        x_actual - x_desired,
        y_actual - y_desired
    ) * 1000

    trajectory_pid_error_history.append(error_mm)

# Error statistics
max_error = np.max(trajectory_pid_error_history)
mean_error = np.mean(trajectory_pid_error_history)
final_error = trajectory_pid_error_history[-1]

print("\n--- End-Effector Tracking Error ---")
print(f"Maximum Error: {max_error:.4f} mm")
print(f"Mean Error: {mean_error:.4f} mm")
print(f"Final Error: {final_error:.4f} mm")

# Plot position error
plt.figure(figsize=(9, 5))

plt.plot(
    trajectory_time,
    trajectory_pid_error_history,
    linewidth=2,
    label="Position Tracking Error"
)

plt.xlabel("Time [s]")
plt.ylabel("Position Error [mm]")
plt.title("End-Effector Trajectory Tracking Error")

plt.grid(True)
plt.legend()
plt.tight_layout()


# --- Desired vs Actual End-Effector Trajectory ---

desired_x_history = []
desired_y_history = []

actual_x_history = []
actual_y_history = []

for k, t in enumerate(trajectory_time):

    # Desired end-effector position
    x_desired = 1.2 + 0.1 * t
    y_desired = 0.8

    desired_x_history.append(x_desired)
    desired_y_history.append(y_desired)

    # Actual end-effector position from PID joint angles
    _, _, x_actual, y_actual = forward_kinematics(
        theta1_actual_history[k],
        theta2_actual_history[k],
        L1,
        L2
    )

    actual_x_history.append(x_actual)
    actual_y_history.append(y_actual)

# Plot desired and actual trajectories
plt.figure(figsize=(9, 6))

plt.plot(
    desired_x_history,
    desired_y_history,
    "r--",
    linewidth=2,
    label="Desired Trajectory"
)

plt.plot(
    actual_x_history,
    actual_y_history,
    "b-",
    linewidth=2,
    label="Actual PID Trajectory"
)

plt.xlabel("End-Effector X Position [m]")
plt.ylabel("End-Effector Y Position [m]")
plt.title("Desired vs Actual End-Effector Trajectory")

plt.grid(True)
plt.legend()
plt.axis("equal")
plt.tight_layout()


# --- X and Y Tracking Error Analysis ---

# Signed tracking errors [mm]
x_error_history = (
    np.array(desired_x_history)
    - np.array(actual_x_history)
) * 1000

y_error_history = (
    np.array(desired_y_history)
    - np.array(actual_y_history)
) * 1000

# Find the time of maximum position error
max_error_index = np.argmax(trajectory_pid_error_history)

print("\n--- X and Y Tracking Error ---")

print(
    f"Maximum Error Time: "
    f"{trajectory_time[max_error_index]:.3f} s"
)

print(
    f"X Error at Maximum: "
    f"{x_error_history[max_error_index]:.4f} mm"
)

print(
    f"Y Error at Maximum: "
    f"{y_error_history[max_error_index]:.4f} mm"
)

print(
    f"Final X Error: "
    f"{x_error_history[-1]:.4f} mm"
)

print(
    f"Final Y Error: "
    f"{y_error_history[-1]:.4f} mm"
)

# Plot tracking errors
plt.figure(figsize=(9, 5))

plt.plot(
    trajectory_time,
    x_error_history,
    label="X Tracking Error",
    linewidth=2
)

plt.plot(
    trajectory_time,
    y_error_history,
    label="Y Tracking Error",
    linewidth=2
)

plt.axhline(0, color="black", linestyle="--", linewidth=1)

plt.xlabel("Time [s]")
plt.ylabel("Tracking Error [mm]")
plt.title("X and Y End-Effector Tracking Errors")

plt.grid(True)
plt.legend()
plt.tight_layout()

# --- Jacobian Tracking Error Analysis ---

# Final desired joint angles
theta_desired_final = np.array([
    theta1_desired_history[-1],
    theta2_desired_history[-1]
])

# Final actual joint angles
theta_actual_final = np.array([
    theta1_actual_history[-1],
    theta2_actual_history[-1]
])

# Actual minus desired joint angles [rad]
delta_theta = theta_actual_final - theta_desired_final

# Jacobian at final desired configuration
J_final = calculate_jacobian(
    theta_desired_final[0],
    theta_desired_final[1],
    L1,
    L2
)

# Contributions from each joint [mm]
error_from_joint1 = -J_final[:, 0] * delta_theta[0] * 1000
error_from_joint2 = -J_final[:, 1] * delta_theta[1] * 1000

# Predicted end-effector tracking error [mm]
predicted_error = error_from_joint1 + error_from_joint2

# Actual tracking error [mm]
actual_error = np.array([
    x_error_history[-1],
    y_error_history[-1]
])

print("\n--- Jacobian Tracking Error Analysis ---")

print("Joint 1 Contribution [mm]:", error_from_joint1)
print("Joint 2 Contribution [mm]:", error_from_joint2)

print("Predicted X/Y Error [mm]:", predicted_error)
print("Actual X/Y Error [mm]:", actual_error)

print(
    "Approximation Difference [mm]:",
    np.linalg.norm(predicted_error - actual_error)
)


# --- Desired Joint Angular Velocity ---

desired_joint_velocity_history = []

for k, t in enumerate(trajectory_time):

    # Desired joint angles at time t
    theta1_d = theta1_desired_history[k]
    theta2_d = theta2_desired_history[k]

    # Jacobian at desired configuration
    J_desired = calculate_jacobian(
        theta1_d, theta2_d, L1, L2
    )

    # Desired joint angular velocities [rad/s]
    omega_desired = np.linalg.solve(
        J_desired, desired_velocity
    )

    desired_joint_velocity_history.append(omega_desired)

print("\n--- Desired Joint Angular Velocities ---")

for k in [0, 10, 20]:
    omega_deg = np.rad2deg(
        desired_joint_velocity_history[k]
    )

    print(
        f"Time: {trajectory_time[k]:.2f} s | "
        f"Omega1: {omega_deg[0]:.4f} deg/s | "
        f"Omega2: {omega_deg[1]:.4f} deg/s"
    )

# --- Actual vs Desired Joint Angular Velocities ---

# Convert angle histories to NumPy arrays
theta1_actual_array = np.array(theta1_actual_history)
theta2_actual_array = np.array(theta2_actual_history)

# Actual average angular velocities [rad/s]
omega1_actual = np.diff(theta1_actual_array) / trajectory_dt
omega2_actual = np.diff(theta2_actual_array) / trajectory_dt

print("\n--- Actual vs Desired Angular Velocities ---")

for k in [0, 9, 19]:

    # Desired velocity at the start of the interval
    omega_desired = desired_joint_velocity_history[k]

    # Actual average velocity over the interval
    omega_actual = np.array([
        omega1_actual[k],
        omega2_actual[k]
    ])

    print(
        f"\nTime Interval: "
        f"{trajectory_time[k]:.2f} - "
        f"{trajectory_time[k+1]:.2f} s"
    )

    print(
        "Desired Velocity [deg/s]:",
        np.round(np.rad2deg(omega_desired), 4)
    )

    print(
        "Actual Average Velocity [deg/s]:",
        np.round(np.rad2deg(omega_actual), 4)
    )

# --- Desired Angular Acceleration and Feedforward Torque ---

# Desired angular velocities [rad/s]
omega_desired_array = np.array(
    desired_joint_velocity_history
)

# Desired angular accelerations [rad/s^2]
alpha_desired_array = np.gradient(
    omega_desired_array,
    trajectory_time,
    axis=0
)

# Feedforward torques [Nm]
torque_ff_array = (
    tracking_inertia * alpha_desired_array
    + tracking_damping * omega_desired_array
)

print("\n--- Feedforward Torque Calculation ---")

for k in [0, 10, 20]:
    print(f"\nTime: {trajectory_time[k]:.2f} s")

    print(
        "Desired Acceleration [deg/s^2]:",
        np.round(np.rad2deg(alpha_desired_array[k]), 4)
    )

    print(
        "Feedforward Torque [Nm]:",
        np.round(torque_ff_array[k], 6)
    )

# --- PID + Feedforward Trajectory Tracking ---

theta_ff = np.array([
    theta1_desired_history[0],
    theta2_desired_history[0]
], dtype=float)

omega_ff = np.zeros(2)
integral_ff = np.zeros(2)
previous_error_ff = np.zeros(2)

trajectory_ff_error_history = []

for k, t in enumerate(trajectory_time):

    # Calculate current end-effector position
    _, _, x_ff, y_ff = forward_kinematics(
        theta_ff[0], theta_ff[1], L1, L2
    )

    # Save position tracking error [mm]
    error_mm = np.hypot(
        x_ff - (1.2 + 0.1 * t),
        y_ff - 0.8
    ) * 1000

    trajectory_ff_error_history.append(error_mm)

    # No integration after the final sample
    if k == len(trajectory_time) - 1:
        break

    dt = trajectory_time[k + 1] - trajectory_time[k]

    desired_theta = np.array([
        theta1_desired_history[k],
        theta2_desired_history[k]
    ])

    # PID feedback
    error = desired_theta - theta_ff
    integral_ff += error * dt

    error_derivative = (
        error - previous_error_ff
    ) / dt

    torque_pid = (
        tracking_kp * error
        + tracking_ki * integral_ff
        + tracking_kd * error_derivative
    )

    # Add feedforward torque
    torque_total = torque_pid + torque_ff_array[k]

    # Simplified joint dynamics
    acceleration_ff = (
        torque_total - tracking_damping * omega_ff
    ) / tracking_inertia

    omega_ff += acceleration_ff * dt
    theta_ff += omega_ff * dt

    previous_error_ff = error.copy()

# --- Hold Final Target After Trajectory ---

hold_duration = 1.0
hold_dt = 0.05
hold_steps = int(round(hold_duration / hold_dt))

final_target_theta = np.array([
    theta1_desired_history[-1],
    theta2_desired_history[-1]
])

_, _, hold_target_x, hold_target_y = forward_kinematics(
    final_target_theta[0], final_target_theta[1], L1, L2
)

hold_time_history = [trajectory_time[-1]]
hold_error_history = [trajectory_ff_error_history[-1]]

# --- Velocity at the Start of Holding ---
theta_hold_start = theta_ff.copy()
omega_hold_start = omega_ff.copy()

# Calculate end-effector velocity using Jacobian
J_hold_start = calculate_jacobian(
    theta_hold_start[0],
    theta_hold_start[1],
    L1,
    L2
)
tip_velocity_start = J_hold_start @ omega_hold_start

print("\n--- Velocity at 1.0 s ---")
print(
    "Joint Angular Velocities [deg/s]:",
    np.round(np.rad2deg(omega_hold_start), 4)
)
print(
    "End-Effector Velocity [m/s]:",
    np.round(tip_velocity_start, 6)
)
print(
    "End-Effector Speed [mm/s]:",
    round(np.linalg.norm(tip_velocity_start) * 1000, 4)
)

for step in range(hold_steps):
    # Fixed target: PID feedback only
    error = final_target_theta - theta_ff
    integral_ff += error * hold_dt

    error_derivative = (
        error - previous_error_ff
    ) / hold_dt

    torque = (
        tracking_kp * error
        + tracking_ki * integral_ff
        + tracking_kd * error_derivative
    )

    acceleration = (
        torque - tracking_damping * omega_ff
    ) / tracking_inertia

    omega_ff += acceleration * hold_dt
    theta_ff += omega_ff * hold_dt

    previous_error_ff = error.copy()

    # Save position error relative to the fixed final target [mm]
    _, _, x_hold, y_hold = forward_kinematics(
        theta_ff[0], theta_ff[1], L1, L2
    )
    hold_error_mm = np.hypot(
        x_hold - hold_target_x,
        y_hold - hold_target_y
    ) * 1000

    hold_time_history.append(
        trajectory_time[-1] + (step + 1) * hold_dt
    )
    hold_error_history.append(hold_error_mm)

print("\n--- Holding Final Target ---")
print(f"Error at 1.0 s: {hold_error_history[0]:.4f} mm")
print(f"Maximum Hold Error: {max(hold_error_history):.4f} mm")
print(f"Error at 2.0 s: {hold_error_history[-1]:.4f} mm")

plt.figure(figsize=(9, 5))
plt.plot(
    hold_time_history,
    hold_error_history,
    linewidth=2,
    label="Position Error During Hold"
)
plt.xlabel("Time [s]")
plt.ylabel("Position Error [mm]")
plt.title("Holding the Final Target Position")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Compare tracking errors
print("\n--- PID vs PID + Feedforward ---")

print(
    f"PID Only Maximum Error: "
    f"{np.max(trajectory_pid_error_history):.4f} mm"
)
print(
    f"PID + FF Maximum Error: "
    f"{np.max(trajectory_ff_error_history):.4f} mm"
)

print(
    f"PID Only Mean Error: "
    f"{np.mean(trajectory_pid_error_history):.4f} mm"
)
print(
    f"PID + FF Mean Error: "
    f"{np.mean(trajectory_ff_error_history):.4f} mm"
)

print(
    f"PID Only Final Error: "
    f"{trajectory_pid_error_history[-1]:.4f} mm"
)
print(
    f"PID + FF Final Error: "
    f"{trajectory_ff_error_history[-1]:.4f} mm"
)

# --- PID vs PID + Feedforward Error Plot ---

plt.figure(figsize=(9, 5))

# PID only
plt.plot(
    trajectory_time,
    trajectory_pid_error_history,
    label="PID Only",
    linewidth=2
)

# PID + Feedforward
plt.plot(
    trajectory_time,
    trajectory_ff_error_history,
    label="PID + Feedforward",
    linewidth=2
)

plt.xlabel("Time [s]")
plt.ylabel("Position Tracking Error [mm]")
plt.title("PID vs PID + Feedforward Tracking Error")

plt.grid(True)
plt.legend()
plt.tight_layout()



# --- Step 5-17: Quintic Trajectory Generation ---

T = 1.0
quintic_time = np.linspace(0.0, T, 101)

s = quintic_time / T

# Desired position [m]
quintic_x = 1.2 + 0.1 * (
    10*s**3 - 15*s**4 + 6*s**5
)

# Desired velocity [m/s]
quintic_vx = (0.1 / T) * (
    30*s**2 - 60*s**3 + 30*s**4
)

# Desired acceleration [m/s^2]
quintic_ax = (0.1 / T**2) * (
    60*s - 180*s**2 + 120*s**3
)

print("\n--- Quintic Trajectory ---")

for k in [0, 50, 100]:
    print(
        f"Time: {quintic_time[k]:.2f} s | "
        f"X: {quintic_x[k]:.4f} m | "
        f"Vx: {quintic_vx[k]*1000:.4f} mm/s | "
        f"Ax: {quintic_ax[k]:.4f} m/s^2"
    )

print(
    f"Maximum Desired Speed: "
    f"{np.max(np.abs(quintic_vx))*1000:.4f} mm/s"
)

# Plot the desired velocity
plt.figure(figsize=(9, 5))

plt.plot(
    quintic_time,
    quintic_vx * 1000,
    label="Quintic Trajectory",
    linewidth=2
)

plt.axhline(
    100,
    linestyle="--",
    label="Previous Constant Velocity"
)

plt.xlabel("Time [s]")
plt.ylabel("Desired X Velocity [mm/s]")
plt.title("Quintic vs Constant-Velocity Trajectory")
plt.grid(True)
plt.legend()
plt.tight_layout()


# --- Step 5-18: Quintic Joint Trajectory ---

quintic_tracking_time = np.linspace(0.0, 1.0, 21)

quintic_theta_history = []
quintic_omega_history = []

for t in quintic_tracking_time:

    s = t / T

    # Desired end-effector position
    x_d = 1.2 + 0.1 * (
        10*s**3 - 15*s**4 + 6*s**5
    )
    y_d = 0.8

    # Desired end-effector velocity
    vx_d = (0.1 / T) * (
        30*s**2 - 60*s**3 + 30*s**4
    )
    vy_d = 0.0

    # Inverse kinematics
    theta1_d, theta2_d = inverse_kinematics(
        x_d, y_d, L1, L2
    )

    theta_d = np.array([theta1_d, theta2_d])

    # Inverse velocity kinematics
    J_d = calculate_jacobian(
        theta1_d, theta2_d, L1, L2
    )

    omega_d = np.linalg.solve(
        J_d,
        np.array([vx_d, vy_d])
    )

    quintic_theta_history.append(theta_d)
    quintic_omega_history.append(omega_d)

quintic_theta_history = np.array(
    quintic_theta_history
)

quintic_omega_history = np.array(
    quintic_omega_history
)

print("\n--- Quintic Joint Trajectory ---")

for k in [0, 10, 20]:

    theta_deg = np.rad2deg(
        quintic_theta_history[k]
    )

    omega_deg = np.rad2deg(
        quintic_omega_history[k]
    )

    print(
        f"\nTime: {quintic_tracking_time[k]:.2f} s"
    )

    print(
        "Desired Angles [deg]:",
        np.round(theta_deg, 4)
    )

    print(
        "Desired Angular Velocities [deg/s]:",
        np.round(omega_deg, 4)
    )


# --- Step 5-19: Quintic Feedforward Torque ---

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


quintic_alpha_history = []
quintic_torque_ff_history = []

for k, t in enumerate(quintic_tracking_time):

    s = t / T

    # Desired end-effector acceleration [m/s^2]
    ax_d = (0.1 / T**2) * (
        60*s - 180*s**2 + 120*s**3
    )

    a_d = np.array([ax_d, 0.0])

    theta_d = quintic_theta_history[k]
    omega_d = quintic_omega_history[k]

    J_d = calculate_jacobian(
        theta_d[0], theta_d[1], L1, L2
    )

    J_dot = calculate_jacobian_dot(
        theta_d[0], theta_d[1],
        omega_d[0], omega_d[1],
        L1, L2
    )

    # Desired joint angular acceleration [rad/s^2]
    alpha_d = np.linalg.solve(
        J_d,
        a_d - J_dot @ omega_d
    )

    # Feedforward torque [Nm]
    torque_ff = (
        tracking_inertia * alpha_d
        + tracking_damping * omega_d
    )

    quintic_alpha_history.append(alpha_d)
    quintic_torque_ff_history.append(torque_ff)

quintic_alpha_history = np.array(quintic_alpha_history)
quintic_torque_ff_history = np.array(quintic_torque_ff_history)

print("\n--- Quintic Feedforward Torque ---")

for k in [0, 10, 20]:

    print(f"\nTime: {quintic_tracking_time[k]:.2f} s")

    print(
        "Desired Angular Acceleration [deg/s^2]:",
        np.round(
            np.rad2deg(quintic_alpha_history[k]), 4
        )
    )

    print(
        "Feedforward Torque [Nm]:",
        np.round(
            quintic_torque_ff_history[k], 6
        )
    )
# --- Step 5-20: Quintic PID + Feedforward Tracking ---

quintic_actual_theta = quintic_theta_history[0].copy()
quintic_actual_omega = np.zeros(2)

quintic_integral_error = np.zeros(2)
quintic_previous_error = np.zeros(2)

quintic_position_error_history = []

for k, t in enumerate(quintic_tracking_time):

    # Desired end-effector position
    s = t / T
    x_target = 1.2 + 0.1 * (
        10*s**3 - 15*s**4 + 6*s**5
    )
    y_target = 0.8

    # Actual end-effector position
    _, _, x_actual, y_actual = forward_kinematics(
        quintic_actual_theta[0],
        quintic_actual_theta[1],
        L1,
        L2
    )

    # Position error [mm]
    position_error_mm = np.hypot(
        x_target - x_actual,
        y_target - y_actual
    ) * 1000

    quintic_position_error_history.append(
        position_error_mm
    )

    # Stop after recording the final sample
    if k == len(quintic_tracking_time) - 1:
        break

    dt = (
        quintic_tracking_time[k + 1]
        - quintic_tracking_time[k]
    )

    # Desired joint angles
    desired_theta = quintic_theta_history[k]

    # PID feedback
    error = desired_theta - quintic_actual_theta
    quintic_integral_error += error * dt

    error_derivative = (
        error - quintic_previous_error
    ) / dt

    torque_pid = (
        tracking_kp * error
        + tracking_ki * quintic_integral_error
        + tracking_kd * error_derivative
    )

    # Feedforward torque from Step 5-19
    torque_ff = quintic_torque_ff_history[k]

    torque_total = torque_pid + torque_ff

    # Simplified joint dynamics
    acceleration = (
        torque_total
        - tracking_damping * quintic_actual_omega
    ) / tracking_inertia

    # Semi-implicit Euler integration
    quintic_actual_omega += acceleration * dt
    quintic_actual_theta += quintic_actual_omega * dt

    quintic_previous_error = error.copy()

# Final end-effector velocity
J_actual_final = calculate_jacobian(
    quintic_actual_theta[0],
    quintic_actual_theta[1],
    L1,
    L2
)

quintic_tip_velocity_final = (
    J_actual_final @ quintic_actual_omega
)

print("\n--- Quintic PID + Feedforward Tracking ---")

print(
    f"Maximum Error: "
    f"{np.max(quintic_position_error_history):.4f} mm"
)

print(
    f"Mean Error: "
    f"{np.mean(quintic_position_error_history):.4f} mm"
)

print(
    f"Final Error: "
    f"{quintic_position_error_history[-1]:.4f} mm"
)

print(
    f"Final End-Effector Speed: "
    f"{np.linalg.norm(quintic_tip_velocity_final)*1000:.4f} mm/s"
)


# --- Step 5-21: Quintic Final Target Holding ---

hold_theta_q = quintic_actual_theta.copy()
hold_omega_q = quintic_actual_omega.copy()
hold_integral_q = quintic_integral_error.copy()
hold_previous_error_q = quintic_previous_error.copy()

hold_target_theta_q = quintic_theta_history[-1].copy()

hold_dt_q = 0.05
hold_steps_q = 20

hold_time_q = [1.0]
hold_error_q = [quintic_position_error_history[-1]]

for step in range(hold_steps_q):

    # Fixed target: PID feedback only
    error = hold_target_theta_q - hold_theta_q

    hold_integral_q += error * hold_dt_q

    error_derivative = (
        error - hold_previous_error_q
    ) / hold_dt_q

    torque = (
        tracking_kp * error
        + tracking_ki * hold_integral_q
        + tracking_kd * error_derivative
    )

    # Joint dynamics
    acceleration = (
        torque - tracking_damping * hold_omega_q
    ) / tracking_inertia

    hold_omega_q += acceleration * hold_dt_q
    hold_theta_q += hold_omega_q * hold_dt_q

    hold_previous_error_q = error.copy()

    # Actual end-effector position
    _, _, x_hold_q, y_hold_q = forward_kinematics(
        hold_theta_q[0], hold_theta_q[1], L1, L2
    )

    position_error = np.hypot(
        1.3 - x_hold_q,
        0.8 - y_hold_q
    ) * 1000

    hold_time_q.append(
        1.0 + (step + 1) * hold_dt_q
    )
    hold_error_q.append(position_error)

# End-effector speed at 2.0 s
J_hold_final_q = calculate_jacobian(
    hold_theta_q[0],
    hold_theta_q[1],
    L1,
    L2
)

tip_velocity_final_q = J_hold_final_q @ hold_omega_q

print("\n--- Quintic Final Target Holding ---")

print(
    f"Error at 1.0 s: "
    f"{hold_error_q[0]:.4f} mm"
)

print(
    f"Maximum Hold Error: "
    f"{np.max(hold_error_q):.4f} mm"
)

print(
    f"Error at 2.0 s: "
    f"{hold_error_q[-1]:.4f} mm"
)

print(
    f"End-Effector Speed at 2.0 s: "
    f"{np.linalg.norm(tip_velocity_final_q)*1000:.4f} mm/s"
)

# Plot holding error
plt.figure(figsize=(9, 5))

plt.plot(
    hold_time_q,
    hold_error_q,
    linewidth=2,
    label="Quintic Trajectory + PID + FF"
)

plt.xlabel("Time [s]")
plt.ylabel("Position Error [mm]")
plt.title("Quintic Trajectory: Final Target Holding")
plt.grid(True)
plt.legend()
plt.tight_layout()


# --- Step 5-22: Constant vs Quintic Tracking Error ---

# Reproduce constant-velocity reference trajectory
comparison_time = np.linspace(0.0, 1.0, 21)

constant_theta_desired = []
constant_omega_desired = []

for t in comparison_time:

    x_d = 1.2 + 0.1 * t
    y_d = 0.8

    theta1_d, theta2_d = inverse_kinematics(
        x_d, y_d, L1, L2
    )

    theta_d = np.array([theta1_d, theta2_d])

    J_d = calculate_jacobian(
        theta1_d, theta2_d, L1, L2
    )

    omega_d = np.linalg.solve(
        J_d, np.array([0.1, 0.0])
    )

    constant_theta_desired.append(theta_d)
    constant_omega_desired.append(omega_d)

constant_theta_desired = np.array(
    constant_theta_desired
)

constant_omega_desired = np.array(
    constant_omega_desired
)

# Same numerical acceleration method as earlier
constant_alpha_desired = np.gradient(
    constant_omega_desired,
    comparison_time,
    axis=0
)

constant_torque_ff = (
    tracking_inertia * constant_alpha_desired
    + tracking_damping * constant_omega_desired
)

# Reproduce PID + FF simulation
constant_theta_actual = constant_theta_desired[0].copy()
constant_omega_actual = np.zeros(2)

constant_integral = np.zeros(2)
constant_previous_error = np.zeros(2)

constant_error_history = []

for k, t in enumerate(comparison_time):

    x_target = 1.2 + 0.1 * t
    y_target = 0.8

    _, _, x_actual, y_actual = forward_kinematics(
        constant_theta_actual[0],
        constant_theta_actual[1],
        L1, L2
    )

    position_error = np.hypot(
        x_target - x_actual,
        y_target - y_actual
    ) * 1000

    constant_error_history.append(position_error)

    if k == len(comparison_time) - 1:
        break

    dt = comparison_time[k + 1] - comparison_time[k]

    error = (
        constant_theta_desired[k]
        - constant_theta_actual
    )

    constant_integral += error * dt

    error_derivative = (
        error - constant_previous_error
    ) / dt

    torque_pid = (
        tracking_kp * error
        + tracking_ki * constant_integral
        + tracking_kd * error_derivative
    )

    torque_total = torque_pid + constant_torque_ff[k]

    acceleration = (
        torque_total
        - tracking_damping * constant_omega_actual
    ) / tracking_inertia

    constant_omega_actual += acceleration * dt
    constant_theta_actual += constant_omega_actual * dt

    constant_previous_error = error.copy()

# --- Step 5-22: Save Tracking Error Comparison ---
from pathlib import Path

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    comparison_time,
    constant_error_history,
    label="Constant Velocity + PID + FF",
    linewidth=2
)

ax.plot(
    quintic_tracking_time,
    quintic_position_error_history,
    label="Quintic Trajectory + PID + FF",
    linewidth=2
)

ax.set_xlabel("Time [s]")
ax.set_ylabel("Position Error [mm]")
ax.set_title("Tracking Error: Constant vs Quintic Trajectory")
ax.grid(True)
ax.legend()
fig.tight_layout()

output_dir = Path("assets")
output_dir.mkdir(exist_ok=True)


output_path = output_dir / "trajectory_tracking_comparison.png"

fig.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)

print(f"Saved graph: {output_path}")


print("\n--- Step 5-22: Tracking Error Comparison ---")

print(
    f"Constant Maximum Error: "
    f"{np.max(constant_error_history):.4f} mm"
)

print(
    f"Quintic Maximum Error: "
    f"{np.max(quintic_position_error_history):.4f} mm"
)

print(
    f"Constant Mean Error: "
    f"{np.mean(constant_error_history):.4f} mm"
)

print(
    f"Quintic Mean Error: "
    f"{np.mean(quintic_position_error_history):.4f} mm"
)

plt.show()
