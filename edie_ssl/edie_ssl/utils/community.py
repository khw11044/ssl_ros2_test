import time
import json
import numpy as np

def send_udp(sock, host, port, result, udp_format='json'):
    """UDP로 SRP 결과 송신"""
    if result is None:
        return False

    try:
        if udp_format == 'json':
            msg = {
                'timestamp': time.time(),
                'angles': [float(a) for a in result['angles']],
                'values': [float(v) for v in result['values']],
                'raw_values': [float(v) for v in result['raw_values']],
                'peak_angle': result['peak_angle'],
                'gcc_angle': result['gcc_angle'],
                'direction': result['direction'],
                'peak_value': float(result['values'][np.argmax(result['values'])]),
                'confidence': round(result['confidence'], 3),
                'is_confident': result['is_confident']
            }
            data = json.dumps(msg).encode()
        else:  # csv
            vals_str = ','.join([f'{v:.4f}' for v in result['values']])
            data = f"{result['peak_angle']:.1f},{result['gcc_angle']:.1f},{result['direction']},{result['confidence']:.3f},{vals_str}".encode()

        sock.sendto(data, (host, port))
        return True
    except Exception as e:
        print(f"\nUDP send error: {e}")
        return False


def print_status(result, send_count):
    """간단한 상태 출력"""
    if result is None:
        return

    peak = result['peak_angle']
    gcc = result['gcc_angle']
    direction = result['direction']
    confidence = result['confidence']

    # 신뢰도에 따른 마커
    if result['is_confident']:
        conf_marker = "●"
    else:
        conf_marker = "○"

    # 간단한 바 시각화
    bar_len = 30
    values = result['values']
    peak_idx = np.argmax(values)

    bar = ""
    step = max(1, len(values) // bar_len)
    for i in range(0, len(values), step):
        if i <= peak_idx < i + step:
            bar += "#"
        elif values[i] > 0.5:
            bar += "="
        elif values[i] > 0.2:
            bar += "-"
        else:
            bar += "."

    print(f"\r{conf_marker} {peak:5.1f}° ({gcc:+5.1f}°) → {direction:9s} conf:{confidence:.2f} [{bar}] sent:{send_count}", end='', flush=True)
