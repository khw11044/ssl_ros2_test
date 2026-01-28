# EDIE Microphone Node

ROS2 Python 기반 원시 오디오 데이터 스트리밍 노드

## 개요

마이크로부터 원시 오디오 데이터를 캡처하여 ROS2 토픽으로 스트리밍합니다.

## 주요 기능

- PyAudio callback 기반 실시간 마이크 입력
- 48000Hz, 2ch, float32 포맷 지원
- 480 samples/chunk (100Hz publish)
- QoS: best_effort + volatile (실시간 스트리밍 최적화)
- Topic: `/edie/audio/raw_data` (UInt8MultiArray)

## 파일 구조

```
edie_mic/
├── edie_mic/
│   ├── __init__.py
│   ├── mic_main.py              # 메인 노드
│   ├── config/
│   │   └── audio_config.yaml    # 오디오 설정
│   └── microphone/
│       ├── element.py
│       ├── mic_pyaudio.py
│       └── ...
├── launch/
│   └── mic_launch.py            # Launch 파일
├── package.xml
├── setup.py
└── README.md
```

## 설정 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `audio.sample_rate` | 48000 | 샘플레이트 (Hz) |
| `audio.channels` | 2 | 채널 수 |
| `audio.bits_per_sample` | 32 | 비트 깊이 (float32) |
| `audio.chunk_size` | 480 | 청크 크기 (samples) |
| `audio.device_name` | default | 오디오 장치 이름 |
| `qos.reliability` | best_effort | QoS 신뢰성 |
| `qos.durability` | volatile | QoS 내구성 |
| `qos.history_depth` | 10 | QoS 히스토리 깊이 |

## 의존성

- `rclpy`
- `std_msgs`
- `pyaudio`

## 빌드

```bash
cd ~/sound_ws
colcon build --symlink-install --packages-select edie_mic
source install/setup.bash
```

## 실행

```bash
# 방법 1: 직접 실행
ros2 run edie_mic mic_node

# 방법 2: launch 파일
ros2 launch edie_mic mic_launch.py
```

## 토픽 확인

```bash
# 토픽 목록
ros2 topic list

# 발행 주파수 확인
ros2 topic hz /edie/audio/raw_data

# 데이터 확인
ros2 topic echo /edie/audio/raw_data
```

## 토픽 정보

| 토픽 | 타입 | 설명 |
|------|------|------|
| `/edie/audio/raw_data` | `std_msgs/UInt8MultiArray` | 원시 오디오 데이터 |
