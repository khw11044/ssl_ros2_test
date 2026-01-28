import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get package share directory
    pkg_share = get_package_share_directory('edie_ssl')
    config_file = os.path.join(pkg_share, 'config', 'ssl_config.yaml')

    return LaunchDescription([
        Node(
            package='edie_ssl',
            executable='ssl_node',
            name='ssl_node',
            output='screen',
            parameters=[config_file],
        ),
    ])
