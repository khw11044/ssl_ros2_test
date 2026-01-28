# -*- coding: utf-8 -*-
"""
EDIE DeepFilter Microphone Node - Noise-filtered audio data streaming over ROS2
Uses DeepFilterMic virtual source for noise reduction via sounddevice
"""
import sys
import os

# Set PulseAudio source to DeepFilterMic before importing audio modules
os.environ['PULSE_SOURCE'] = 'DeepFilterMic'

pkg_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, pkg_path)

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import UInt8MultiArray

from microphone.mic_sounddevice import Source
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


class DeepFilterMicNode(Node):
    """ROS2 Node for streaming DeepFilter noise-reduced audio data"""

    def __init__(self):
        super().__init__('deepfilter_mic_node')

        # DeepFilterMic 고정 설정: 48000Hz, 1ch, float32
        self.rate = 48000
        self.channels = 1
        self.frame_size = 480  # 10ms at 48kHz
        self.topic = '/edie/audio/df_data'

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

        # Create Source with DeepFilterMic (via pulse)
        self.src = Source(
            rate=self.rate,
            frames_size=self.frame_size,
            channels=self.channels,
            device_name='pulse',
            bits_per_sample=32
        )

        # Create ROS2 publisher sink and link to source
        self.ros2_sink = ROS2PublisherSink(self.publisher)
        self.src.link(self.ros2_sink)

        # Log configuration
        self.get_logger().info('=' * 60)
        self.get_logger().info('DeepFilter Microphone Node Started')
        self.get_logger().info('=' * 60)
        self.get_logger().info(f'Source: DeepFilterMic (via PULSE_SOURCE)')
        self.get_logger().info(f'Rate: {self.rate}Hz, Channels: {self.channels}, Format: float32')
        self.get_logger().info(f'Frame size: {self.frame_size}')
        self.get_logger().info(f'Topic: {self.topic}')
        self.get_logger().info(f'Publish Rate: ~{self.rate / self.frame_size:.1f} Hz')
        self.get_logger().info('=' * 60)

        # Start microphone
        self.src.recursive_start()

    def destroy_node(self):
        """Clean up resources"""
        self.get_logger().info('Stopping DeepFilter Microphone Node...')
        self.src.recursive_stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = DeepFilterMicNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
