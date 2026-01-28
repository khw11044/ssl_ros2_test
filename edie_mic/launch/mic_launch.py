# -*- coding: utf-8 -*-
"""
Launch file for EDIE Microphone Node
"""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get package share directory
    pkg_share = get_package_share_directory('edie_mic')
    config_file = os.path.join(pkg_share, 'config', 'audio_config.yaml')

    return LaunchDescription([
        Node(
            package='edie_mic',
            executable='mic_node',
            name='microphone_node',
            output='screen',
            parameters=[
                # Load parameters from YAML or override here
                {'pub.pub_audio_raw_data': '/edie/audio/raw_data'},
                {'audio.sample_rate': 48000},
                {'audio.channels': 2},
                {'audio.bits_per_sample': 32},
                {'audio.chunk_size': 480},
                {'audio.device_name': 'default'},
                {'qos.reliability': 'best_effort'},
                {'qos.durability': 'volatile'},
                {'qos.history_depth': 10},
            ],
        ),
    ])
