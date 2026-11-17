import numpy as np
import mujoco

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


class MujocoJointStatePublisher(Node):

    def __init__(self):
        super().__init__('mujoco_joint_state_publisher')

        # =====================================================
        # 1. ROS 2 Publisher
        #
        # MuJoCoの現在状態を /joint_states に送る
        # =====================================================

        self.publisher_ = self.create_publisher(
            JointState,
            '/joint_states',
            10,
        )

        # =====================================================
        # 2. ROS 2 Subscriber
        #
        # 外部Nodeから目標関節角を受け取る
        # =====================================================

        self.subscription = self.create_subscription(
            Float64MultiArray,
            '/joint_command',
            self.command_callback,
            10,
        )

        # =====================================================
        # 3. Load MuJoCo model
        # =====================================================

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

        # MuJoCoの初期状態を正しく計算
        mujoco.mj_forward(
            self.model,
            self.data,
        )

        # =====================================================
        # 4. Initial target
        #
        # 起動直後は「現在姿勢」を目標にする
        #
        # /joint_command が来たら
        # command_callback() によって書き換えられる
        # =====================================================

        self.q_des = self.data.qpos.copy()

        # =====================================================
        # 5. PD gains
        # =====================================================

        self.kp = np.array([
            30.0,
            20.0,
        ])

        self.kd = np.array([
            8.0,
            5.0,
        ])

        # =====================================================
        # 6. Simulation / ROS update frequency
        #
        # MuJoCo:
        # dt = 0.002 s
        # 500 Hz
        #
        # ROS:
        # 0.01 s
        # 100 Hz
        #
        # 0.01 / 0.002 = 5
        # =====================================================

        self.physics_steps_per_update = 5

        self.timer = self.create_timer(
            0.01,
            self.update_simulation,
        )

        self.get_logger().info(
            'MuJoCo robot initialized.'
        )


    # =========================================================
    # 7. Receive target joint command
    # =========================================================

    def command_callback(self, msg):

        # q1, q2 の2値が必要
        if len(msg.data) < 2:

            self.get_logger().warning(
                'Joint command must contain 2 values.'
            )

            return

        # ROS 2から受信した目標値へ更新
        self.q_des = np.array([
            msg.data[0],
            msg.data[1],
        ])

        self.get_logger().info(
            f'New target received: '
            f'q1={self.q_des[0]:.3f}, '
            f'q2={self.q_des[1]:.3f}'
        )


    # =========================================================
    # 8. MuJoCo simulation update
    # =========================================================

    def update_simulation(self):

        # -----------------------------------------------------
        # MuJoCoを5 physics step進める
        # -----------------------------------------------------

        for _ in range(
            self.physics_steps_per_update
        ):

            # Current state
            q = self.data.qpos.copy()
            dq = self.data.qvel.copy()

            # ---------------------------------------------
            # PD control
            # ---------------------------------------------

            error = (
                self.q_des
                - q
            )

            tau_pd = (
                self.kp * error
                - self.kd * dq
            )
    
            tau = (
                tau_pd + self.data.qfrc_bias[:2]
            )

            # ---------------------------------------------
            # Apply torque
            # ---------------------------------------------

            self.data.qfrc_applied[:2] = tau

            # ---------------------------------------------
            # Advance physics by 0.002 s
            # ---------------------------------------------

            mujoco.mj_step(
                self.model,
                self.data,
            )
        # =====================================================
        # 9. Read latest state
        # =====================================================

        q = self.data.qpos.copy()
        dq = self.data.qvel.copy()
        tau = self.data.qfrc_applied.copy()

        # =====================================================
        # 10. Create JointState message
        # =====================================================

        msg = JointState()

        # Timestamp
        msg.header.stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )

        # Joint names
        msg.name = [
            'joint1',
            'joint2',
        ]

        # Joint angles
        msg.position = [
            float(q[0]),
            float(q[1]),
        ]

        # Joint velocities
        msg.velocity = [
            float(dq[0]),
            float(dq[1]),
        ]

        # Applied torques
        msg.effort = [
            float(tau[0]),
            float(tau[1]),
        ]

        # =====================================================
        # 11. Publish current robot state
        # =====================================================

        self.publisher_.publish(
            msg
        )

        self.get_logger().info(
            f'q1={q[0]:.3f}, '
            f'q2={q[1]:.3f}, '
            f'target=({self.q_des[0]:.3f}, '
            f'{self.q_des[1]:.3f})'
        )


# =============================================================
# 12. Main
# =============================================================

def main(args=None):

    # ROS 2 initialization
    rclpy.init(
        args=args
    )

    # Create Node
    node = MujocoJointStatePublisher()

    try:

        # Keep Node running
        rclpy.spin(
            node
        )

    except KeyboardInterrupt:

        pass

    finally:

        # Clean shutdown
        node.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':
    main()
