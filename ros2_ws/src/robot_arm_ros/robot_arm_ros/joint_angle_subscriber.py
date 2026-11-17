import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64MultiArray


class JointAngleSubscriber(Node):

    def __init__(self):
        super().__init__('joint_angle_subscriber')

        self.subscription = self.create_subscription(
            Float64MultiArray,
            '/joint_angles',
            self.listener_callback,
            10,
        )

    def listener_callback(self, msg):

        q1 = msg.data[0]
        q2 = msg.data[1]

        self.get_logger().info(
            f'Received q1={q1}, q2={q2}'
        )


def main(args=None):

    rclpy.init(args=args)

    node = JointAngleSubscriber()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
