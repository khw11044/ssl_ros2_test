import pyaudio
import numpy as np

def list_devices():
    """사용 가능한 오디오 입력 장치 목록"""
    p = pyaudio.PyAudio()
    print("\n=== Available Input Devices ===")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] >= 2:
            print(f"  [{i}] {info['name']} (channels: {info['maxInputChannels']})")
    p.terminate()
    print()

def visualize_ild(ild, bar_len=31):
    """ILD를 시각적으로 표시 (±20dB 범위)"""
    center = bar_len // 2
    ild_clamped = np.clip(ild, -20, 20)
    pos = center + int(ild_clamped / 20 * center)
    pos = max(0, min(bar_len - 1, pos))

    bar = ['-'] * bar_len
    bar[center] = '|'
    bar[pos] = '*'
    return ''.join(bar)


def get_direction(result, left_th=0, right_th=3):
    """ILD 기반 방향 판단"""
    if result < left_th:
        return "Sound came from:  ◀ LEFT "
    elif result > right_th:
        return "Sound came from:  RIGHT ▶"
    else:
        return "Sound came from:  CENTER "
    
    
def visualize_gcc(results):
    """gcc 방향 결과 시각적으로 표시 """
    # 결과 출력
    print("Time (s)    Direction")
    print("-" * 40)
    for time_sec, theta in results:
        # 방향 시각화
        bar_len = 30
        pos = int((theta + 90) / 180 * bar_len)
        pos = max(0, min(bar_len - 1, pos))
        bar = ['-'] * bar_len
        bar[pos] = '|'
        bar[bar_len // 2] = '+'

        print(f"{time_sec:6.2f}s    {theta:+6.1f}° [{''.join(bar)}]")


def print_linear_distribution(angles_deg, values, width=72):
    """선형 막대 그래프로 시각화"""
    values = np.array(values)
    min_val = values.min()
    max_val = values.max()
    if max_val - min_val > 1e-10:
        norm_values = (values - min_val) / (max_val - min_val)
    else:
        norm_values = np.zeros_like(values)

    peak_idx = np.argmax(values)
    peak_angle = angles_deg[peak_idx]

    print("\n" + "=" * (width + 15))
    print(f"  SRP-PHAT 360° Direction Distribution")
    print(f"  Peak Direction: {peak_angle:.1f}°")
    print("=" * (width + 15))

    # 샘플링 (너무 많으면 간격 조절)
    step = max(1, len(angles_deg) // 36)  # 최대 36개 행

    for i in range(0, len(angles_deg), step):
        angle = angles_deg[i]
        val = norm_values[i]
        bar_len = int(val * width)

        # 피크 위치 표시
        if i == peak_idx or (i < peak_idx < i + step):
            marker = " << PEAK"
            bar = "#" * bar_len
        else:
            marker = ""
            bar = "=" * bar_len

        print(f"{angle:>6.1f}° |{bar:<{width}}|{marker}")

    print("=" * (width + 15))
    
    
def angle_to_direction(angle):
    """SRP 각도(0~180)를 방향으로 변환"""
    if angle < 60:
        return "RIGHT"
    elif angle > 120:
        return "LEFT"
    else:
        return "FRONT"
    
    
def print_realtime_distribution(angles_deg, values, peak_angle, gcc_angle, width=60,
                                confidence=None, direction=None):
    """실시간 180도 분포 시각화 (커서 이동으로 업데이트)"""
    # 정규화 (이미 정규화된 값이면 그대로 사용)
    min_val = values.min()
    max_val = values.max()
    if max_val - min_val > 1e-10:
        norm_values = (values - min_val) / (max_val - min_val)
    else:
        norm_values = np.zeros_like(values)

    peak_idx = np.argmax(values)
    if direction is None:
        direction = angle_to_direction(peak_angle)

    lines = []
    lines.append("=" * (width + 25))
    if confidence is not None:
        lines.append(f"  SRP-PHAT 180° | Peak: {peak_angle:5.1f}° ({gcc_angle:+6.1f}°) → {direction} [conf: {confidence:.2f}]")
    else:
        lines.append(f"  SRP-PHAT 180° | Peak: {peak_angle:5.1f}° ({gcc_angle:+6.1f}°) → {direction}")
    lines.append("=" * (width + 25))
    lines.append(f"  0°=RIGHT, 90°=FRONT, 180°=LEFT")
    lines.append("-" * (width + 25))

    # 모든 각도 표시 (0~180도)
    step = max(1, len(angles_deg) // 18)  # 최대 18줄

    for i in range(0, len(angles_deg), step):
        angle = angles_deg[i]
        val = norm_values[i]
        bar_len = int(val * width)

        # 방향 라벨
        if angle < 30:
            label = "R"
        elif angle > 150:
            label = "L"
        elif 75 <= angle <= 105:
            label = "F"
        else:
            label = " "

        # 피크 근처 표시
        if i <= peak_idx < i + step:
            marker = " << PEAK"
            bar = "#" * bar_len
        else:
            marker = ""
            bar = "=" * bar_len

        lines.append(f"{label} {angle:>5.1f}° |{bar:<{width}}|{marker}")

    lines.append("=" * (width + 25))

    # 출력
    output = "\r" + "\n".join(lines)
    print(output, end='')

    # 커서를 맨 위로 이동
    num_lines = len(lines)
    print(f"\033[{num_lines}A", end='')