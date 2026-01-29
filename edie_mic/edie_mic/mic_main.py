
"""
EDIE Unified Microphone Node - Raw + DeepFilter audio streaming over ROS2
Publishes both raw audio data and DeepFilter noise-reduced audio simultaneously
Uses parecord to specify exact PulseAudio sources for each stream
"""
import sys
import os

pkg_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, pkg_path)

import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import UInt8MultiArray

# Use parecord to specify exact PulseAudio source names (no device conflicts)
from microphone.mic_parecord import Source as PaRecordSource
from microphone.element import Element

# PulseAudio source names (from: pactl list sources short)
RAW_DEVICE = 'alsa_input.platform-rt5651-sound.stereo-fallback'  # Hardware mic (2ch, s24-32le)
DF_DEVICE = 'DeepFilterMic'  # DeepFilter virtual source (1ch, float32)


class ROS2PublisherSink(Element):
    """Sink element that publishes audio data to ROS2 topic"""

    def __init__(self, publisher, logger=None, name=""):
        super().__init__()
        self.publisher = publisher
        self.logger = logger
        self.name = name

    def put(self, data):
        """Receive audio data and publish to ROS2 topic"""
        msg = UInt8MultiArray()
        msg.data = list(data)
        self.publisher.publish(msg)
        super().put(data)


class MicrophoneNode(Node):
    """ROS2 Node for streaming both raw and DeepFilter audio data"""

    def __init__(self):
        super().__init__('unified_microphone_node')

        # Load config file
        self.declare_parameter('config_file', '')
        config_file = self.get_parameter('config_file').value

        if not config_file:
            config_file = os.path.join(os.path.dirname(__file__), '../config/audio_config.yaml')
            self.get_logger().info(f'Using default config file: {config_file}')

        # Load yaml config
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
                if 'microphone_node' in config_data:
                    ros_params = config_data['microphone_node'].get('ros__parameters', {})
                    self.config_pub = ros_params.get('pub', {})
                    self.config_audio = config_data['microphone_node'].get('audio__parameters', {})
                else:
                    self.config_pub = {}
                    self.config_audio = {}
                    self.get_logger().warn('Config file missing microphone_node section, using defaults')

            self.get_logger().info(f'Loaded config from: {config_file}')
        except Exception as e:
            self.get_logger().warn(f'Failed to load config file: {str(e)}, using defaults')
            self.config_pub = {}
            self.config_audio = {}

        # Common audio parameters
        self.audio_rate = self.config_audio.get('rate', 48000)
        self.audio_frame_size = self.config_audio.get('frame_size', 480)

        # Raw audio parameters
        self.raw_topic = self.config_pub.get('pub_audio_raw_data', '/edie/audio/raw_data')
        self.raw_channels = self.config_audio.get('channels', 2)

        # DeepFilter audio parameters (fixed: 48kHz, 1ch, float32)
        self.df_topic = self.config_pub.get('pub_audio_df_data', '/edie/audio/df_data')
        self.df_channels = 1


        # QoS profile for real-time audio streaming
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Create publishers
        self.raw_publisher = self.create_publisher(
            UInt8MultiArray,
            self.raw_topic,
            qos_profile
        )

        self.df_publisher = self.create_publisher(
            UInt8MultiArray,
            self.df_topic,
            qos_profile
        )

        # Create Raw Audio Source (parecord with explicit PulseAudio source)
        self.raw_src = PaRecordSource(
            rate=self.audio_rate,
            frames_size=self.audio_frame_size,
            channels=self.raw_channels,
            device_name=RAW_DEVICE,
            bits_per_sample=16
        )

        # Create DeepFilter Source (parecord with DeepFilterMic source)
        self.df_src = PaRecordSource(
            rate=self.audio_rate,
            frames_size=self.audio_frame_size,
            channels=self.df_channels,
            device_name=DF_DEVICE,
            bits_per_sample=32
        )

        # Create sinks and link to sources
        self.raw_sink = ROS2PublisherSink(self.raw_publisher, self.get_logger(), "raw")
        self.df_sink = ROS2PublisherSink(self.df_publisher, self.get_logger(), "df")

        self.raw_src.link(self.raw_sink)
        self.df_src.link(self.df_sink)

        # Log configuration
        self.get_logger().info('=' * 60)
        self.get_logger().info('Unified Microphone Node Started')
        self.get_logger().info('=' * 60)
        self.get_logger().info('[Raw Audio - via parecord]')
        self.get_logger().info(f'  Source: {RAW_DEVICE}')
        self.get_logger().info(f'  Rate: {self.audio_rate}Hz, Channels: {self.raw_channels}')
        self.get_logger().info(f'  Frame size: {self.audio_frame_size}')
        self.get_logger().info(f'  Topic: {self.raw_topic}')
        self.get_logger().info(f'  Publish Rate: ~{self.audio_rate / self.audio_frame_size:.1f} Hz')
        self.get_logger().info('-' * 60)
        self.get_logger().info('[DeepFilter Audio - via parecord]')
        self.get_logger().info(f'  Source: {DF_DEVICE}')
        self.get_logger().info(f'  Rate: {self.audio_rate}Hz, Channels: {self.df_channels}, Format: float32')
        self.get_logger().info(f'  Frame size: {self.audio_frame_size}')
        self.get_logger().info(f'  Topic: {self.df_topic}')
        self.get_logger().info(f'  Publish Rate: ~{self.audio_rate / self.audio_frame_size:.1f} Hz')
        self.get_logger().info('=' * 60)

        # Start both sources (each runs in its own thread)
        self.raw_src.recursive_start()
        self.df_src.recursive_start()

    def destroy_node(self):
        """Clean up resources"""
        self.get_logger().info('Stopping Unified Microphone Node...')
        self.raw_src.recursive_stop()
        self.df_src.recursive_stop()
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
