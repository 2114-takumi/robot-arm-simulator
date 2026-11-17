import numpy as np
import mujoco

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64MultiArray


class MujocoJointPublisher(Node):

    def __init__(self):
        super().__init__('mujoco_joint_publisher')

        # ---------------------------------
        # ROS 2 Publisher
        # ---------------------------------

        self.publisher_ = self.create_publisher(
            Float64MultiArray,
            '/joint_angles',
            10,
        )

        # ---------------------------------
        # Load MuJoCo model
        # ---------------------------------

        model_path = (
            '/mnt/c/Users/cfca5399/'
            'Projects/personal/python/'
            'robot-arm-simulator/'
            'models/two_link_arm.xml'
        )

        self.model = mujoco.MjModel.from_xml_path(
            model_path
        )

        self.data = mujoco.MjData(
            self.model
        )

        # ---------------------------------
        # Target joint angles
        # ---------------------------------

        self.q_des = np.array([
            0.3,
            -0.4,
        ])

        # Simple PD gains
        self.kp = np.array([
            30.0,
            20.0,
        ])

        self.kd = np.array([
            8.0,
            5.0,
        ])

        # ---------------------------------
        # Timer
        # ---------------------------------
        #
        # 0.01 s
        # = 100 Hz ROS update
        # ---------------------------------

        self.timer = self.create_timer(
            0.01,
            self.update_simulation,
        )


    def update_simulation(self):

        # =================================
        # 1. Read current MuJoCo state
        # =================================

        q = self.data.qpos.copy()
        dq = self.data.qvel.copy()

        # =================================
        # 2. PD controller
        # =================================

        error = self.q_des - q

        tau = (
            self.kp * error
            - self.kd * dq
        )

        # =================================
        # 3. Apply torque
        # =================================

        self.data.qfrc_applied[:] = tau

        # =================================
        # 4. Advance MuJoCo
        # =================================

        mujoco.mj_step(
            self.model,
            self.data,
        )

        # =================================
        # 5. Read NEW joint angles
        # =================================

        q_new = self.data.qpos.copy()

        # =================================
        # 6. ROS message
        # =================================

        msg = Float64MultiArray()

        msg.data = [
            float(q_new[0]),
            float(q_new[1]),
        ]

        # =================================
        # 7. Publish
        # =================================

        self.publisher_.publish(msg)

        self.get_logger().info(
            f'MuJoCo q1={q_new[0]:.3f}, '
            f'q2={q_new[1]:.3f}'
        )


def main(args=None):

    rclpy.init(args=args)

    node = MujocoJointPublisher()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
