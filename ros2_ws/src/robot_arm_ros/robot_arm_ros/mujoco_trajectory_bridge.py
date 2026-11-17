import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer

from control_msgs.action import FollowJointTrajectory
from std_msgs.msg import Float64MultiArray


class MujocoTrajectoryBridge(Node):

    def __init__(self):
        super().__init__('mujoco_trajectory_bridge')

        # MoveItのtrajectoryを
        # MuJoCo用 /joint_command に変換して送る
        self.command_publisher = self.create_publisher(
            Float64MultiArray,
            '/joint_command',
            10,
        )

        # MoveItからFollowJointTrajectoryを受け取る
        self.action_server = ActionServer(
            self,
            FollowJointTrajectory,
            '/mujoco_arm_controller/follow_joint_trajectory',
            self.execute_callback,
        )

        self.get_logger().info(
            'MuJoCo trajectory bridge started.'
        )


    def execute_callback(self, goal_handle):

        trajectory = goal_handle.request.trajectory

        self.get_logger().info(
            f'Received trajectory: '
            f'{len(trajectory.points)} points'
        )

        # ---------------------------------
        # joint名の順番を確認
        # ---------------------------------

        joint_names = list(
            trajectory.joint_names
        )

        try:
            joint1_index = joint_names.index('joint1')
            joint2_index = joint_names.index('joint2')

        except ValueError:

            self.get_logger().error(
                'Trajectory must contain joint1 and joint2.'
            )

            goal_handle.abort()

            result = FollowJointTrajectory.Result()

            result.error_code = (
                FollowJointTrajectory.Result.INVALID_JOINTS
            )

            return result


        previous_time = 0.0


        # =================================
        # trajectoryの各pointを順番に送る
        # =================================

        for point in trajectory.points:

            current_time = (
                point.time_from_start.sec
                +
                point.time_from_start.nanosec * 1e-9
            )

            wait_time = (
                current_time
                - previous_time
            )

            if wait_time > 0.0:
                time.sleep(wait_time)


            # ---------------------------------
            # MoveIt trajectory
            # ->
            # MuJoCo q_des
            # ---------------------------------

            q1_des = point.positions[
                joint1_index
            ]

            q2_des = point.positions[
                joint2_index
            ]


            msg = Float64MultiArray()

            msg.data = [
                float(q1_des),
                float(q2_des),
            ]


            self.command_publisher.publish(
                msg
            )


            self.get_logger().info(
                f'Command: '
                f'q1={q1_des:.3f}, '
                f'q2={q2_des:.3f}, '
                f't={current_time:.3f}'
            )


            previous_time = current_time


        # =================================
        # trajectory完了
        # =================================

        goal_handle.succeed()

        result = FollowJointTrajectory.Result()

        result.error_code = (
            FollowJointTrajectory.Result.SUCCESSFUL
        )

        self.get_logger().info(
            'Trajectory execution completed.'
        )

        return result


def main(args=None):

    rclpy.init(
        args=args
    )

    node = MujocoTrajectoryBridge()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
