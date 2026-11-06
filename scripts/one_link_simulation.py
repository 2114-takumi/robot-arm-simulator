import mujoco
import math 


model = mujoco.MjModel.from_xml_path("models/one_link_arm.xml")

data = mujoco.MjData(model)
data.qpos[0] = -math.pi / 2

for i in range(1000):
    mujoco.mj_step(model, data)

    if i % 100 == 0:
        print(
            f"時刻: {data.time:.2f} s, "
            f"角度: {data.qpos[0]:.3f} rad, "
            f"角速度: {data.qvel[0]:.3f} rad/s"
        )