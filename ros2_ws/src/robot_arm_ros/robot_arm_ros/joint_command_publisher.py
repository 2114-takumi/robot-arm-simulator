import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64MultiArray


class JointCommandPublisher(Node):

    def __init__(self):
        super().__init__('joint_command_publisher')

        self.publisher_ = self.create_publisher(
            Float64MultiArray,
            '/joint_command',
            10,
        )

        self.timer = self.create_timer(
            1.0,
            self.publish_command,
        )

    def publish_command(self):

        msg = Float64MultiArray()

        msg.data = [
            0.5,
            -0.2,
        ]

        self.publisher_.publish(msg)

        self.get_logger().info(
            f'Command q1={msg.data[0]:.3f}, '
            f'q2={msg.data[1]:.3f}'
        )


def main(args=None):

    rclpy.init(args=args)

    node = JointCommandPublisher()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
