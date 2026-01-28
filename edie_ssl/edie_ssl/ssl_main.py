"""
ROS2 Sound Source Localization Node
GCC-PHAT + SRP-PHAT 알고리즘을 사용한 음원 방향 추정

각도 범위: -90 ~ +90 (가운데 0, 음수=왼쪽, 양수=오른쪽)
"""
import sys
import os

# 패키지 내부 모듈 경로 추가
pkg_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, pkg_path)

import numpy as np
import rclpy
from rclpy.node import Node
from edie_ssl_msgs.msg import SSLResult

from microphone.source import Source
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


class SSLNode(Node):
    def __init__(self):
        super().__init__('ssl_node')

        # 파라미터 선언 및 로드
        self.declare_parameters(
            namespace='',
            parameters=[
                ('rate', 16000),
                ('channels', 2),
                ('frame_size', 320),
                ('mic_distance', 0.155),
                ('publish_rate', 20.0),
                ('gcc_chunk', 20),
                ('db_threshold', 6.0),
                ('dir_threshold', 2.0),
                ('history_size', 50),
                ('cooldown', 0.5),
                ('srp_chunk', 10),
                ('nfft', 512),
                ('n_grid', 18),
                ('alpha', 0.3),
                ('confidence_threshold', 1.5),
            ]
        )

        # 파라미터 가져오기
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

        # Source 생성
        self.src = Source(
            rate=self.rate,
            frames_size=self.frame_size,
            channels=self.channels
        )

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

        # Publisher 생성
        self.publisher = self.create_publisher(SSLResult, 'ssl_result', 10)

        # Timer 생성
        timer_period = 1.0 / self.publish_rate
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # 마이크 시작
        self.src.recursive_start()

        # 로그 출력
        self.get_logger().info('=' * 60)
        self.get_logger().info('SSL Node Started')
        self.get_logger().info('=' * 60)
        self.get_logger().info(f'Rate: {self.rate}Hz, Channels: {self.channels}')
        self.get_logger().info(f'Mic distance: {self.mic_distance * 100}cm')
        self.get_logger().info(f'Publish rate: {self.publish_rate}Hz')
        self.get_logger().info(f'Angle: -90=Left, 0=Front, +90=Right')
        self.get_logger().info('=' * 60)

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
        self.src.recursive_stop()
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
