# -*- coding: utf-8 -*-
"""
Audio Save Node - Subscribe to DeepFilter audio topic and save to WAV file
Saves: output.wav (16bit mono)

Usage: ros2 run edie_mic save_audio_1ch <duration_seconds>
Example: ros2 run edie_mic save_audio_1ch 5
"""
import sys
import wave
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import UInt8MultiArray


class AudioSaveNode(Node):
    """ROS2 Node for saving DeepFilter audio data to WAV file"""

    def __init__(self, duration: float):
        super().__init__('audio_save_node')

        self.duration = duration
        self.rate = 48000
        self.channels = 1
        self.output_file = 'output.wav'

        # Audio buffer
        self.audio_data = []
        self.start_time = None

        # QoS profile (match with df_node)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscribe to DeepFilter audio topic
        self.subscription = self.create_subscription(
            UInt8MultiArray,
            '/edie/audio/df_data',
            self.audio_callback,
            qos_profile
        )

        self.get_logger().info('=' * 50)
        self.get_logger().info('Audio Save Node Started')
        self.get_logger().info('=' * 50)
        self.get_logger().info(f'Recording duration: {self.duration} seconds')
        self.get_logger().info(f'Topic: /edie/audio/df_data')
        self.get_logger().info(f'Output: {self.output_file}')
        self.get_logger().info('Waiting for audio data...')

    def audio_callback(self, msg):
        """Receive audio data from topic"""
        if self.start_time is None:
            self.start_time = self.get_clock().now()
            self.get_logger().info('Recording started...')

        # Collect audio data (float32 bytes)
        self.audio_data.append(bytes(msg.data))

        # Check if recording is done
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        if elapsed >= self.duration:
            self.save_and_exit()

    def save_and_exit(self):
        """Save audio data to WAV file and exit"""
        self.get_logger().info('Recording finished. Saving file...')

        # Combine all audio data
        audio_bytes = b''.join(self.audio_data)

        # Convert float32 to int16
        float_data = np.frombuffer(audio_bytes, dtype=np.float32)
        # Clip to [-1, 1] range and scale to int16
        float_data = np.clip(float_data, -1.0, 1.0)
        int16_data = (float_data * 32767).astype(np.int16)

        # Save as 16bit mono WAV
        with wave.open(self.output_file, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16bit = 2bytes
            wf.setframerate(self.rate)
            wf.writeframes(int16_data.tobytes())

        self.get_logger().info(f'Saved: {self.output_file}')
        self.get_logger().info(f'Duration: {len(int16_data) / self.rate:.2f} seconds')
        self.get_logger().info(f'Samples: {len(int16_data)}')

        # Exit
        raise SystemExit


def main(args=None):
    rclpy.init(args=args)

    # Get duration from command line argument
    if len(sys.argv) < 2:
        print('Usage: ros2 run edie_mic save_audio_1ch <duration_seconds>')
        print('Example: ros2 run edie_mic save_audio_1ch 5')
        return

    try:
        duration = float(sys.argv[1])
    except ValueError:
        print(f'Error: Invalid duration "{sys.argv[1]}"')
        return

    node = AudioSaveNode(duration)

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
