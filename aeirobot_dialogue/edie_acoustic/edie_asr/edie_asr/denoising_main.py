# -*- coding: utf-8 -*-
"""
Audio Save Node - Subscribe to audio topic and save to WAV files
Saves: left.wav, right.wav, total.wav
"""
import sys
import wave
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import UInt8MultiArray


class AeiRobotDenosing(Node):
    """ROS2 Node for saving denoising audio data from topic to WAV files"""

    def __init__(self, duration: float):
        super().__init__('audio_save_node')

        self.duration = duration
        self.rate = 16000
        self.channels = 2
        self.sample_width = 2  # 16bit = 2bytes

        # Audio buffer
        self.audio_data = []
        self.start_time = None

        # QoS profile (match with mic_node)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscribe to audio topic
        self.subscription = self.create_subscription(
            UInt8MultiArray,
            '/edie/audio/raw_data',
            self.audio_callback,
            qos_profile
        )

        self.get_logger().info('=' * 50)
        self.get_logger().info('Audio Save Node Started')
        self.get_logger().info('=' * 50)
        self.get_logger().info(f'Recording duration: {self.duration} seconds')
        self.get_logger().info(f'Topic: /edie/audio/raw_data')
        self.get_logger().info('Waiting for audio data...')

    def audio_callback(self, msg):
        """Receive audio data from topic"""
        if self.start_time is None:
            self.start_time = self.get_clock().now()
            self.get_logger().info('Recording started...')

        # Collect audio data
        self.audio_data.append(bytes(msg.data))

        # Check if recording is done
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        if elapsed >= self.duration:
            self.save_and_exit()

    def save_and_exit(self):
        """Save audio data to WAV files and exit"""
        self.get_logger().info('Recording finished. Saving files...')

        # Combine all audio data
        audio_bytes = b''.join(self.audio_data)
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)


        self.get_logger().info('=' * 50)
        self.get_logger().info(f'Saved: total.wav')
        self.get_logger().info(f'Actual duration: {duration_actual:.2f} seconds')
        self.get_logger().info('=' * 50)

        # Exit
        raise SystemExit


def main(args=None):
    rclpy.init(args=args)

    # Get duration from command line argument
    if len(sys.argv) < 2:
        print('Usage: ros2 run edie_mic save_audio <duration_seconds>')
        print('Example: ros2 run edie_mic save_audio 5')
        return

    try:
        duration = float(sys.argv[1])
    except ValueError:
        print(f'Error: Invalid duration "{sys.argv[1]}"')
        return

    node = AeiRobotDenosing(duration)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user')
    finally:
        node.destroy_node()


if __name__ == '__main__':
    main()
