from setuptools import find_packages, setup
import os

from glob import glob

package_name = 'robot_arm_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (
          os.path.join(
              'share',
              package_name,
              'urdf',
          ),
          glob('urdf/*')
        ),

        (
          os.path.join(
              'share',
              package_name,
              'launch',
        ),
        glob('launch/*')
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='cfca5399',
    maintainer_email='cfca5399@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
    'console_scripts': [
        'joint_angle_publisher = robot_arm_ros.joint_angle_publisher:main',
        'joint_angle_subscriber = robot_arm_ros.joint_angle_subscriber:main',
	'mujoco_joint_publisher = robot_arm_ros.mujoco_joint_publisher:main',
	'mujoco_joint_state_publisher = robot_arm_ros.mujoco_joint_state_publisher:main',
	'joint_command_publisher = robot_arm_ros.joint_command_publisher:main',
	'mujoco_trajectory_bridge = robot_arm_ros.mujoco_trajectory_bridge:main',
      ],
    },
)

