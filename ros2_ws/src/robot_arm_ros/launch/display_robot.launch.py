import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    # =================================
    # Find this package
    # =================================

    package_dir = get_package_share_directory(
        'robot_arm_ros'
    )

    # =================================
    # URDF path
    # =================================

    urdf_path = os.path.join(
        package_dir,
        'urdf',
        'two_link_arm.urdf',
    )

    # =================================
    # Read URDF
    # =================================

    with open(
        urdf_path,
        'r',
        encoding='utf-8',
    ) as file:

        robot_description = file.read()


    # =================================
    # 1. MuJoCo JointState Publisher
    # =================================

    mujoco_node = Node(

        package='robot_arm_ros',

        executable='mujoco_joint_state_publisher',

        name='mujoco_joint_state_publisher',

        output='screen',
    )


    # =================================
    # 2. robot_state_publisher
    # =================================

    robot_state_publisher_node = Node(

        package='robot_state_publisher',

        executable='robot_state_publisher',

        name='robot_state_publisher',

        parameters=[
            {
                'robot_description':
                    robot_description
            }
        ],

        output='screen',
    )


    # =================================
    # 3. RViz
    # =================================

    rviz_node = Node(

        package='rviz2',

        executable='rviz2',

        name='rviz2',

        output='screen',
    )


    # =================================
    # Start everything
    # =================================

    return LaunchDescription(
        [
            mujoco_node,
            robot_state_publisher_node,
            rviz_node,
        ]
    )
