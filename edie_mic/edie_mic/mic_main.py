# -*- coding: utf-8 -*-
"""
EDIE Microphone Node - Raw audio data streaming over ROS2
"""
import sys
import os

pkg_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, pkg_path)

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import UInt8MultiArray

from microphone.source import Source
from microphone.element import Element


class ROS2PublisherSink(Element):
    """Sink element that publishes audio data to ROS2 topic"""

    def __init__(self, publisher):
        super().__init__()
        self.publisher = publisher

    def put(self, data):
        """Receive audio data and publish to ROS2 topic"""
        msg = UInt8MultiArray()
        msg.data = list(data)
        self.publisher.publish(msg)
        super().put(data)


class MicrophoneNode(Node):
    """ROS2 Node for streaming raw audio data from microphone"""

    def __init__(self):
        super().__init__('microphone_node')

        # Declare parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('rate', 16000),
                ('channels', 2),
                ('frame_size', 320),
                ('topic', '/edie/audio/raw_data'),
            ]
        )

        # Get parameters
        self.rate = self.get_parameter('rate').value
        self.channels = self.get_parameter('channels').value
        self.frame_size = self.get_parameter('frame_size').value
        self.topic = self.get_parameter('topic').value

        # QoS profile for real-time audio streaming
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Create publisher
        self.publisher = self.create_publisher(
            UInt8MultiArray,
            self.topic,
            qos_profile
        )

        # Create Source (same as ssl_node)
        self.src = Source(
            rate=self.rate,
            frames_size=self.frame_size,
            channels=self.channels
        )

        # Create ROS2 publisher sink and link to source
        self.ros2_sink = ROS2PublisherSink(self.publisher)
        self.src.link(self.ros2_sink)

        # Log configuration
        self.get_logger().info('=' * 60)
        self.get_logger().info('Microphone Node Started')
        self.get_logger().info('=' * 60)
        self.get_logger().info(f'Rate: {self.rate}Hz, Channels: {self.channels}')
        self.get_logger().info(f'Frame size: {self.frame_size}')
        self.get_logger().info(f'Topic: {self.topic}')
        self.get_logger().info(f'Publish Rate: ~{self.rate / self.frame_size:.1f} Hz')
        self.get_logger().info('=' * 60)

        # Start microphone
        self.src.recursive_start()

    def destroy_node(self):
        """Clean up resources"""
        self.get_logger().info('Stopping Microphone Node...')
        self.src.recursive_stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = MicrophoneNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
