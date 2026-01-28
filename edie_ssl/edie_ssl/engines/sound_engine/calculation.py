

import numpy as np


def _bilinear_zpk(z, p, k, fs):
    """Bilinear transform: 아날로그 zpk를 디지털 zpk로 변환"""
    z = np.atleast_1d(z)
    p = np.atleast_1d(p)

    degree = max(len(p), len(z))

    # s = 2*fs*(z-1)/(z+1) 변환의 역변환
    # z_d = (1 + s/(2*fs)) / (1 - s/(2*fs))
    fs2 = 2.0 * fs

    # 디지털 영점
    z_d = (1 + z / fs2) / (1 - z / fs2)

    # 디지털 극점
    p_d = (1 + p / fs2) / (1 - p / fs2)

    # 영점 개수가 극점보다 적으면 -1로 채움
    z_d = np.append(z_d, -np.ones(degree - len(z)))

    # 게인 조정
    num = np.real(k * np.prod(fs2 - z) / np.prod(fs2 - p))

    return z_d, p_d, num


def _zpk2sos(z, p, k):
    """Zero-pole-gain을 second-order sections로 변환"""
    z = np.atleast_1d(z)
    p = np.atleast_1d(p)

    n_sections = (max(len(p), len(z)) + 1) // 2
    sos = np.zeros((n_sections, 6))

    # 복소수 극점/영점을 켤레 쌍으로 묶기
    p = np.concatenate([p, np.zeros(2 * n_sections - len(p))])
    z = np.concatenate([z, np.zeros(2 * n_sections - len(z))])

    # 극점을 크기 순으로 정렬 (안정성을 위해 단위원에 가까운 것부터)
    p_idx = np.argsort(np.abs(p))[::-1]
    p = p[p_idx]

    z_idx = np.argsort(np.abs(z))[::-1]
    z = z[z_idx]

    for i in range(n_sections):
        # 2개씩 묶어서 SOS 섹션 생성
        p_pair = p[2*i:2*i+2]
        z_pair = z[2*i:2*i+2]

        # b 계수: (1 - z1*z^-1)(1 - z2*z^-1) = 1 - (z1+z2)*z^-1 + z1*z2*z^-2
        b0 = 1.0
        b1 = -np.real(z_pair[0] + z_pair[1])
        b2 = np.real(z_pair[0] * z_pair[1])

        # a 계수: (1 - p1*z^-1)(1 - p2*z^-1)
        a0 = 1.0
        a1 = -np.real(p_pair[0] + p_pair[1])
        a2 = np.real(p_pair[0] * p_pair[1])

        sos[i] = [b0, b1, b2, a0, a1, a2]

    # 첫 번째 섹션에 게인 적용
    sos[0, :3] *= k

    return sos


def _freqz_zpk(z, p, k, worN, fs):
    """ZPK 형태의 주파수 응답 계산"""
    z = np.atleast_1d(z)
    p = np.atleast_1d(p)
    worN = np.atleast_1d(worN)

    # 주파수를 정규화된 각주파수로 변환
    w = 2 * np.pi * worN / fs

    # e^(jw) 계산
    zm1 = np.exp(1j * w)

    # H(z) = k * prod(z - z_i) / prod(z - p_i)
    h = np.ones(len(w), dtype=np.complex128) * k

    for zi in z:
        h *= (zm1 - zi)
    for pi in p:
        h /= (zm1 - pi)

    return w, h


def _sosfilt(sos, x):
    """SOS 필터 적용 (direct form II transposed)"""
    x = np.asarray(x, dtype=np.float64)
    y = x.copy()

    for section in sos:
        b0, b1, b2, a0, a1, a2 = section
        # a0으로 정규화
        b0, b1, b2 = b0/a0, b1/a0, b2/a0
        a1, a2 = a1/a0, a2/a0

        # Direct form II transposed
        z1, z2 = 0.0, 0.0
        y_new = np.zeros_like(y)

        for i in range(len(y)):
            xi = y[i]
            yi = b0 * xi + z1
            z1 = b1 * xi - a1 * yi + z2
            z2 = b2 * xi - a2 * yi
            y_new[i] = yi

        y = y_new

    return y


def A_weighting_coeffs(fs):
    """
    A-weighting 필터 계수 생성 (IEC 61672-1 표준)

    Args:
        fs: 샘플링 레이트 (Hz)
    Returns:
        sos: Second-order sections 형태의 필터 계수
    """
    # A-weighting 표준 주파수 상수
    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12194.217

    # 아날로그 필터의 극점과 영점 (rad/s)
    zeros = [0, 0, 0, 0]
    poles = [
        -2 * np.pi * f1,
        -2 * np.pi * f1,
        -2 * np.pi * f2,
        -2 * np.pi * f3,
        -2 * np.pi * f4,
        -2 * np.pi * f4,
    ]

    # 1000Hz에서 0dB가 되도록 정규화
    k = (2 * np.pi * f4) ** 2 * (2 * np.pi * f1) ** 2

    # 아날로그 -> 디지털 변환 (bilinear transform)
    z, p, k = _bilinear_zpk(zeros, poles, k, fs)

    # 1000Hz에서 정규화
    w, h = _freqz_zpk(z, p, k, [1000], fs=fs)
    k = k / np.abs(h[0])

    # SOS (second-order sections) 형태로 변환 (수치 안정성)
    sos = _zpk2sos(z, p, k)
    return sos


def A_weight(signal, fs):
    """
    신호에 A-weighting 필터 적용

    Args:
        signal: 입력 오디오 신호
        fs: 샘플링 레이트 (Hz)
    Returns:
        A-weighted 신호
    """
    sos = A_weighting_coeffs(fs)
    return _sosfilt(sos, signal)


def calculate_rms_db(signal):    # calculate_rms_db
    """RMS를 dB로 변환"""
    # ILD 참조값 (16bit PCM 기준)
    REF_AMPLITUDE = 32768.0  # 16bit max value
    
    rms = np.sqrt(np.mean(signal.astype(np.float64) ** 2))
    if rms < 1e-10:
        return -100.0  # 무음
    db = 20 * np.log10(rms / REF_AMPLITUDE)
    return db


def calc_lid(ch_left, ch_right, sample_rate, chunk_size=3200):
    """2채널 오디오에서 ILD 분석 (왼쪽 귀 - 오른쪽 귀)"""
    total_samples = min(len(ch_left), len(ch_right))
    results = []

    # 청크 단위로 분석
    for i in range(0, total_samples - chunk_size, chunk_size):
        chunk_left = ch_left[i:i + chunk_size]
        chunk_right = ch_right[i:i + chunk_size]

        # 무음 필터링
        if np.max(np.abs(chunk_left)) < 500 and np.max(np.abs(chunk_right)) < 500:
            continue

        # 각 채널의 레벨 계산 (dB)
        db_left = calculate_rms_db(chunk_left)
        db_right = calculate_rms_db(chunk_right)

        # ILD 계산: 양수면 왼쪽이 더 큼, 음수면 오른쪽이 더 큼
        ild = db_left - db_right

        time_sec = i / sample_rate
        results.append((time_sec, db_left, db_right, ild))

    return results



def compute_stft(signal, nfft, hop_size):
    """STFT 계산"""
    window = np.hanning(nfft)
    num_frames = (len(signal) - nfft) // hop_size + 1

    stft = np.zeros((nfft // 2 + 1, num_frames), dtype=np.complex128)

    for i in range(num_frames):
        start = i * hop_size
        frame = signal[start:start + nfft] * window
        stft[:, i] = np.fft.rfft(frame)

    return stft