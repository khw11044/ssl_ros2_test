"""
ROS2 Sound Source Localization Node
GCC-PHAT + SRP-PHAT 알고리즘을 사용한 음원 방향 추정

각도 범위: -90 ~ +90 (가운데 0, 음수=왼쪽, 양수=오른쪽)

오디오 데이터는 /edie/audio/raw_data 토픽에서 구독
"""
import sys
import os
import yaml

# 패키지 내부 모듈 경로 추가
pkg_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, pkg_path)

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import UInt8MultiArray
from edie_ssl_msgs.msg import SSLResult

from microphone.element import Element
from utils.estimator import GCCEstimator, SRPEstimator


SOUND_SPEED = 343.0


def angle_to_direction(angle: float) -> str:
    """각도(-90~+90)를 방향으로 변환"""
    if angle < -30:
        return "Left"
    elif angle > 30:
        return "Right"
    else:
        return "Front"


class ROS2SubscriberSource(Element):
    """Source element that receives audio data from ROS2 topic"""

    def __init__(self):
        super().__init__()

    def put(self, data):
        """Receive audio data from topic and forward to linked elements"""
        super().put(data)


class SSLNode(Node):
    def __init__(self):
        super().__init__('ssl_node')

        # config 파일 경로 파라미터 받기
        self.declare_parameter('config_file', '')
        config_file = self.get_parameter('config_file').value

        if not config_file:
            config_file = os.path.join(os.path.dirname(__file__), '../config/ssl_config.yaml')
            self.get_logger().info(f'Using default config file: {config_file}')

        # yaml 파일 로드
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
                if 'ssl_node' in config_data:
                    ros_params = config_data['ssl_node'].get('ros__parameters', {})
                    self.config_sub = ros_params.get('sub', {})
                    self.config_pub = ros_params.get('pub', {})
                    self.config_ssl = config_data['ssl_node'].get('ssl__parameters', {})
                else:
                    self.config_sub = {}
                    self.config_pub = {}
                    self.config_ssl = {}
                    self.get_logger().error('Invalid config file format')
                    return

            self.get_logger().info(f'Loaded config from: {config_file}')
        except Exception as e:
            self.get_logger().error(f'Failed to load config file: {str(e)}')
            return

        # 파라미터 선언
        self.declare_parameter('sub_audio_raw_data', self.config_sub.get('sub_audio_raw_data', '/edie/audio/raw_data'))
        self.declare_parameter('pub_ssl_data', self.config_pub.get('pub_ssl_data', '/edie/ssl/data'))
        self.declare_parameter('rate', self.config_ssl.get('rate', 16000))
        self.declare_parameter('channels', self.config_ssl.get('channels', 2))
        self.declare_parameter('frame_size', self.config_ssl.get('frame_size', 320))
        self.declare_parameter('mic_distance', self.config_ssl.get('mic_distance', 0.155))
        self.declare_parameter('publish_rate', self.config_ssl.get('publish_rate', 20.0))
        self.declare_parameter('gcc_chunk', self.config_ssl.get('gcc_chunk', 20))
        self.declare_parameter('db_threshold', self.config_ssl.get('db_threshold', 6.0))
        self.declare_parameter('dir_threshold', self.config_ssl.get('dir_threshold', 2.0))
        self.declare_parameter('history_size', self.config_ssl.get('history_size', 50))
        self.declare_parameter('cooldown', self.config_ssl.get('cooldown', 0.5))
        self.declare_parameter('srp_chunk', self.config_ssl.get('srp_chunk', 10))
        self.declare_parameter('nfft', self.config_ssl.get('nfft', 512))
        self.declare_parameter('n_grid', self.config_ssl.get('n_grid', 18))
        self.declare_parameter('alpha', self.config_ssl.get('alpha', 0.3))
        self.declare_parameter('confidence_threshold', self.config_ssl.get('confidence_threshold', 1.5))

        # 파라미터 가져오기
        self.sub_audio_raw_data = self.get_parameter('sub_audio_raw_data').value
        self.pub_ssl_data = self.get_parameter('pub_ssl_data').value
        self.rate = self.get_parameter('rate').value
        self.channels = self.get_parameter('channels').value
        self.frame_size = self.get_parameter('frame_size').value
        self.mic_distance = self.get_parameter('mic_distance').value
        self.publish_rate = self.get_parameter('publish_rate').value

        gcc_chunk = self.get_parameter('gcc_chunk').value
        db_threshold = self.get_parameter('db_threshold').value
        dir_threshold = self.get_parameter('dir_threshold').value
        history_size = self.get_parameter('history_size').value
        cooldown = self.get_parameter('cooldown').value

        srp_chunk = self.get_parameter('srp_chunk').value
        nfft = self.get_parameter('nfft').value
        n_grid = self.get_parameter('n_grid').value
        alpha = self.get_parameter('alpha').value
        confidence_threshold = self.get_parameter('confidence_threshold').value

        # 최대 TDOA 계산
        max_tdoa = self.mic_distance / SOUND_SPEED

        # ROS2 Subscriber Source 생성 (기존 Source 대체)
        self.src = ROS2SubscriberSource()

        # GCC Estimator 생성
        self.gcc_estimator = GCCEstimator(
            rate=self.rate,
            chunks=gcc_chunk,
            max_tdoa=max_tdoa,
            db_history_size=history_size,
            db_threshold=db_threshold,
            direction_threshold=dir_threshold
        )
        self.gcc_estimator.event_cooldown = cooldown

        # SRP Estimator 생성
        self.srp_estimator = SRPEstimator(
            rate=self.rate,
            mic_dis=self.mic_distance,
            chunks=srp_chunk,
            nfft=nfft,
            n_grid=n_grid,
            alpha=alpha,
            confidence_threshold=confidence_threshold
        )

        # 파이프라인 연결
        self.src.link(self.gcc_estimator)
        self.src.link(self.srp_estimator)

        # QoS profile (match with mic_node)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Audio topic subscriber
        self.subscription = self.create_subscription(
            UInt8MultiArray,
            self.sub_audio_raw_data,
            self.audio_callback,
            qos_profile
        )

        # Publisher 생성
        self.publisher = self.create_publisher(SSLResult, self.pub_ssl_data, 10)

        # Timer 생성
        timer_period = 1.0 / self.publish_rate
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # 로그 출력
        self.get_logger().info('=' * 60)
        self.get_logger().info('SSL Node Started')
        self.get_logger().info('=' * 60)
        self.get_logger().info(f'Rate: {self.rate}Hz, Channels: {self.channels}')
        self.get_logger().info(f'Mic distance: {self.mic_distance * 100}cm')
        self.get_logger().info(f'Publish rate: {self.publish_rate}Hz')
        self.get_logger().info(f'Audio topic: {self.sub_audio_raw_data}')
        self.get_logger().info(f'Angle: -90=Left, 0=Front, +90=Right')
        self.get_logger().info('=' * 60)

    def audio_callback(self, msg):
        """오디오 토픽 콜백 - 데이터를 파이프라인에 전달"""
        audio_data = bytes(msg.data)
        self.src.put(audio_data)

    def timer_callback(self):
        """주기적으로 DOA 결과 publish"""
        msg = SSLResult()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'microphone'

        has_data = False

        # GCC 결과 가져오기
        gcc_result = self.gcc_estimator.get_direction_and_event()

        if gcc_result[0] is not None:
            theta, avg_dir, z_score_shifting, norm_dir, current_db, avg_db, relative_db, event, event_reason = gcc_result

            # GCC 데이터 설정
            msg.gcc_angle = int(round(theta))
            msg.gcc_direction = angle_to_direction(theta)
            msg.gcc_db = float(current_db)
            msg.gcc_event = event is not None

            if event:
                msg.gcc_event_angle = int(round(theta))
                msg.gcc_event_direction = event  # "LEFT" or "RIGHT"
                msg.gcc_event_db = float(current_db)
            else:
                msg.gcc_event_angle = 0
                msg.gcc_event_direction = ""
                msg.gcc_event_db = 0.0

            has_data = True

        # SRP 결과 가져오기
        srp_result = self.srp_estimator.get_srp_distribution_norm()

        if srp_result is not None:
            values = srp_result['values']
            peak_idx = np.argmax(values)
            peak_value = float(values[peak_idx])

            # SRP 각도 변환: 0~180 → -90~+90
            peak_angle_raw = float(srp_result['peak_angle'])
            peak_angle = peak_angle_raw - 90

            # 분포 각도 변환
            angles_shifted = [float(a) - 90 for a in srp_result['angles']]

            # SRP 데이터 설정
            msg.srp_distribution = [float(v) for v in values]
            msg.srp_peak_percent = int(round(peak_value * 100))
            msg.srp_peak_angle = int(round(peak_angle))
            msg.srp_direction = angle_to_direction(peak_angle)

            has_data = True

        # 데이터가 있으면 publish
        if has_data:
            self.publisher.publish(msg)

    def destroy_node(self):
        """노드 종료 시 정리"""
        self.get_logger().info('Stopping SSL Node...')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = SSLNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
