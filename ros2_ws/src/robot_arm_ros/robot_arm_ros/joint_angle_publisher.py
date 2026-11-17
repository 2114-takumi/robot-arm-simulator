import math
import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64MultiArray


class JointAnglePublisher(Node):

    def __init__(self):
        super().__init__('joint_angle_publisher')

        self.publisher_ = self.create_publisher(
            Float64MultiArray,
            '/joint_angles',
            10,
        )

        self.time = 0.0

        self.timer = self.create_timer(
            0.1,
            self.publish_joint_angles,
        )

    def publish_joint_angles(self):

        q1 = 0.5 * math.sin(self.time)
        q2 = 0.4 * math.cos(self.time)

        msg = Float64MultiArray()

        msg.data = [
            q1,
            q2,
        ]

        self.publisher_.publish(msg)

        self.get_logger().info(
            f'Publishing q1={q1:.3f}, q2={q2:.3f}'
        )

        self.time += 0.1


def main(args=None):

    rclpy.init(args=args)

    node = JointAnglePublisher()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
