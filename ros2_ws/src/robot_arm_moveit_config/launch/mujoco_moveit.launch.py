import os

from launch import LaunchDescription
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory

from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():

    # =====================================================
    # MoveIt configuration
    # =====================================================

    moveit_config = (
        MoveItConfigsBuilder(
            "two_link_arm",
            package_name="robot_arm_moveit_config",
        )
        .trajectory_execution(
            file_path="config/moveit_controllers.yaml"
        )
        .planning_scene_monitor(
            publish_robot_description=True,
            publish_robot_description_semantic=True,
        )
        .planning_pipelines(
            pipelines=["ompl"]
        )
        .to_moveit_configs()
    )

    # =====================================================
    # 1. MoveIt move_group
    # =====================================================

    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict()
        ],
    )

    # =====================================================
    # 2. MuJoCo robot node
    #
    # /joint_command -> MuJoCo
    # MuJoCo -> /joint_states
    # =====================================================

    mujoco_robot = Node(
        package="robot_arm_ros",
        executable="mujoco_joint_state_publisher",
        name="mujoco_joint_state_publisher",
        output="screen",
    )

    # =====================================================
    # 3. MoveIt trajectory -> /joint_command bridge
    # =====================================================

    trajectory_bridge = Node(
        package="robot_arm_ros",
        executable="mujoco_trajectory_bridge",
        name="mujoco_trajectory_bridge",
        output="screen",
    )

    # =====================================================
    # 4. /joint_states + URDF -> /tf
    # =====================================================

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            moveit_config.robot_description
        ],
    )

    # =====================================================
    # 5. world -> base_link
    # =====================================================

    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="world_to_base_link",
        output="screen",
        arguments=[
            "--frame-id",
            "world",
            "--child-frame-id",
            "base_link",
        ],
    )

    # =====================================================
    # 6. RViz
    # =====================================================

    config_package = get_package_share_directory(
        "robot_arm_moveit_config"
    )

    rviz_config = os.path.join(
        config_package,
        "config",
        "moveit.rviz",
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=[
            "-d",
            rviz_config,
        ],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
        ],
    )

    # =====================================================
    # Start everything
    # =====================================================

    return LaunchDescription([
        move_group,
        mujoco_robot,
        trajectory_bridge,
        robot_state_publisher,
        static_tf,
        rviz,
    ])
