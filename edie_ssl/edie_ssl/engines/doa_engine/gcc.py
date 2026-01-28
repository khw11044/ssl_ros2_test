"""
GCC-PHAT (Generalized Cross Correlation - Phase Transform) DOA Estimator
=========================================================================

TDOA 기반 음원 방향 추정 알고리즘
2채널 마이크에 최적화, 빠르고 간단한 구현

Copyright (c) 2017 Yihui Xiong (original gcc_phat implementation)
Modified for DOA integration
"""

import numpy as np
import math
from numpy.fft import rfft, irfft

# Optional: pyfftw for faster FFT (if available)
try:
    import pyfftw
    pyfftw_available = True
except ImportError:
    pyfftw_available = False

# Optional: mkl_fft for faster FFT (if available)
try:
    import mkl_fft
    mkl_available = True
except ImportError:
    mkl_available = False


# =============================================================================
# Standalone Functions (클래스 없이 사용 가능)
# =============================================================================

def gcc_phat(sig, refsig, fs=48000, max_tau=None, interp=16):
    """
    가장 정석적인 GCC-PHAT 구현 (Standalone 함수)

    두 신호 간의 시간 지연(TDOA)을 추정합니다.

    Parameters
    ----------
    sig : np.ndarray
        첫 번째 채널 신호 (time-domain)
    refsig : np.ndarray
        두 번째 채널 신호 (time-domain, 참조)
    fs : int
        샘플링 주파수 (Hz)
    max_tau : float, optional
        최대 시간 지연 (초). None이면 제한 없음
    interp : int
        보간 배수 (정밀도 향상). 기본값 16

    Returns
    -------
    tau : float
        시간 지연 (초). 양수면 sig가 refsig보다 늦게 도착
    cc : np.ndarray
        교차 상관 함수

    Examples
    --------
    >>> # 2채널 오디오에서 TDOA 추정
    >>> tau, cc = gcc_phat(channel_0, channel_1, fs=48000)
    >>> print(f"Time delay: {tau*1000:.3f} ms")
    """
    # FFT 길이 (두 신호 길이의 합)
    n = sig.shape[0] + refsig.shape[0]

    # FFT
    SIG = rfft(sig, n=n)
    REFSIG = rfft(refsig, n=n)

    # Cross-correlation in frequency domain
    R = SIG * np.conj(REFSIG)

    # Phase Transform (정규화)
    cc = irfft(R / (np.abs(R) + 1e-8), n=(interp * n))

    # Shift 범위 제한
    max_shift = int(interp * n / 2)
    if max_tau is not None:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    # 중심 정렬
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))

    # 최대값 찾기
    shift = np.argmax(cc) - max_shift

    # 시간 지연 계산
    tau = shift / float(interp * fs)

    return tau, cc


def gcc_phat_to_angle(tau, mic_distance, c=343.0):
    """
    시간 지연(tau)을 각도로 변환

    Parameters
    ----------
    tau : float
        시간 지연 (초)
    mic_distance : float
        마이크 간 거리 (미터)
    c : float
        음속 (m/s). 기본값 343.0

    Returns
    -------
    angle_deg : float
        방위각 (도). -90 ~ +90 범위
        0도 = 정면, +90도 = 오른쪽, -90도 = 왼쪽
    angle_rad : float
        방위각 (라디안)

    Examples
    --------
    >>> tau = 0.0002  # 0.2ms
    >>> angle = gcc_phat_to_angle(tau, mic_distance=0.155)
    >>> print(f"Source direction: {angle:.1f}°")
    """
    # 최대 시간 지연
    max_tau = mic_distance / c

    # 정규화 및 clipping
    tau_normalized = np.clip(tau / max_tau, -1.0, 1.0)

    # 각도 계산
    angle_rad = np.arcsin(tau_normalized)
    angle_deg = np.degrees(angle_rad)

    return angle_deg, angle_rad


def estimate_doa_simple(channel_0, channel_1, mic_distance=0.155, fs=48000, c=343.0):
    """
    2채널 오디오로부터 음원 방향 추정 (간단한 wrapper)

    Parameters
    ----------
    channel_0 : np.ndarray
        첫 번째 채널 신호
    channel_1 : np.ndarray
        두 번째 채널 신호
    mic_distance : float
        마이크 간 거리 (미터). 기본값 0.155 (15.5cm)
    fs : int
        샘플링 주파수 (Hz)
    c : float
        음속 (m/s)

    Returns
    -------
    dict
        'tau': 시간 지연 (초)
        'tau_ms': 시간 지연 (밀리초)
        'angle_deg': 방위각 (도)
        'angle_rad': 방위각 (라디안)
        'direction': 방향 텍스트 ('왼쪽', '정면', '오른쪽')

    Examples
    --------
    >>> result = estimate_doa_simple(ch0, ch1, mic_distance=0.155)
    >>> print(f"Direction: {result['direction']} ({result['angle_deg']:.1f}°)")
    """
    # 최대 시간 지연 계산
    max_tau = mic_distance / c

    # GCC-PHAT로 시간 지연 추정
    tau, cc = gcc_phat(channel_0, channel_1, fs=fs, max_tau=max_tau)

    # 각도 변환
    angle_deg, angle_rad = gcc_phat_to_angle(tau, mic_distance, c)

    # 방향 텍스트
    if -30 <= angle_deg <= 30:
        direction = "정면"
    elif angle_deg > 30:
        direction = "오른쪽"
    else:
        direction = "왼쪽"

    return {
        'tau': tau,
        'tau_ms': tau * 1000,
        'angle_deg': angle_deg,
        'angle_rad': angle_rad,
        'direction': direction
    }


# =============================================================================
# Class-based Implementation
# =============================================================================


class GCCPHAT:
    """
    GCC-PHAT 기반 DOA 추정 클래스

    2채널 마이크로폰 사이의 시간 지연(TDOA)을 추정하여 방향 계산
    """

    def __init__(self, L, fs, nfft, c=343.0, num_src=1, n_grid=360,
                 use_bandpass=False, freq_range=None, transform='numpy', **kwargs):
        """
        Parameters
        ----------
        L : np.ndarray
            마이크 위치 (2 or 3, n_mics)
        fs : int
            샘플링 주파수
        nfft : int
            FFT 길이 (사용 안 함, 호환성을 위해 유지)
        c : float
            음속 (m/s)
        num_src : int
            음원 수 (사용 안 함, 항상 1개로 가정)
        n_grid : int
            그리드 수 (사용 안 함)
        use_bandpass : bool
            Bandpass filter 사용 여부
        freq_range : list
            Bandpass filter 주파수 범위 [low, high] Hz
        transform : str
            FFT 패키지: 'numpy' (기본), 'fftw', 'mkl'
        """
        if L.shape[1] != 2:
            raise ValueError("GCC-PHAT는 2채널 마이크만 지원합니다.")

        self.L = L
        self.fs = fs
        self.nfft = nfft
        self.c = c
        self.num_src = num_src
        self.use_bandpass = use_bandpass

        # Bandpass filter 설정
        if use_bandpass and freq_range is not None:
            self.freq_range = freq_range
        else:
            self.freq_range = [250, 4000]  # 기본값

        # 마이크 간 거리 계산
        self.mic_distance = np.linalg.norm(L[:, 1] - L[:, 0])

        # 최대 시간 지연
        self.max_tau = self.mic_distance / self.c

        # 보간 배수 (정밀도 향상)
        self.interp = 16

        # FFT 함수 선택 (transform 모듈과 동일한 패턴)
        if transform == 'fftw' and pyfftw_available:
            # import warnings
            # warnings.warn(
            #     "pyfftw는 동적 FFT 길이를 위해 매번 객체 생성이 필요하여 "
            #     "GCC-PHAT에서는 numpy를 사용합니다."
            # )
            self.rfft = rfft
            self.irfft = irfft
        elif transform == 'mkl' and mkl_available:
            self.rfft = mkl_fft.rfft_numpy
            self.irfft = mkl_fft.irfft_numpy
        else:
            self.rfft = rfft
            self.irfft = irfft

        # 결과 저장용
        self.azimuth_recon = 0.0
        self.tau_recon = 0.0

    def _gcc_phat(self, sig, refsig):
        """
        GCC-PHAT 계산 (원본)

        Parameters
        ----------
        sig : np.ndarray
            첫 번째 채널 신호
        refsig : np.ndarray
            두 번째 채널 신호 (참조)

        Returns
        -------
        tau : float
            시간 지연 (초)
        cc : np.ndarray
            교차 상관 함수
        """
        # FFT 길이
        n = sig.shape[0] + refsig.shape[0]

        # FFT (transform 모듈과 동일한 패턴 사용)
        SIG = self.rfft(sig, n=n)
        REFSIG = self.rfft(refsig, n=n)

        # Cross-correlation in frequency domain
        R = SIG * np.conj(REFSIG)

        # Phase Transform (정규화)
        cc = self.irfft(R / (np.abs(R) + 1e-8), n=(self.interp * n))

        # Shift 범위 제한
        max_shift = int(self.interp * n / 2)
        if self.max_tau:
            max_shift = np.minimum(int(self.interp * self.fs * self.max_tau), max_shift)

        # 중심 정렬
        cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))

        # 최대값 찾기
        shift = np.argmax(cc) - max_shift

        # 시간 지연 계산
        tau = shift / float(self.interp * self.fs)

        return tau, cc
    

    def locate_sources(self, X, freq_bins=None, use_optimized=True):
        """
        음원 위치 추정

        Parameters
        ----------
        X : np.ndarray
            STFT 결과 (n_mics, n_freq, n_frames)
        freq_bins : np.ndarray
            사용할 주파수 빈 (사용 안 함)
        use_optimized : bool
            True: STFT 결과를 직접 사용 (빠름, 추천)
            False: ISTFT → RFFT 경로 (기존 방식, 호환성)
        """
        # X: (n_mics, n_freq, n_frames)
        n_mics, n_freq, n_frames = X.shape

        # 프레임 평균
        X0_avg = X[0, :, :].mean(axis=1)  # (n_freq,)
        X1_avg = X[1, :, :].mean(axis=1)

        if use_optimized:
            # 최적화 경로: STFT 결과를 직접 사용
            tau, cc = self._gcc_phat_from_stft(X0_avg, X1_avg)
        else:
            # 기존 경로: ISTFT → 윈도우 → RFFT
            # (하위 호환성을 위해 유지)
            sig0 = self.irfft(X0_avg)
            sig1 = self.irfft(X1_avg)

            # 윈도우 적용 (Hanning)
            window = np.hanning(len(sig0))
            sig0 = sig0 * window
            sig1 = sig1 * window

            # GCC-PHAT 계산
            if self.use_bandpass:
                tau, cc = self._gcc_phat_bandpass(sig0, sig1)
            else:
                tau, cc = self._gcc_phat(sig0, sig1)

        # 각도 계산
        # theta = arcsin(tau / max_tau)
        tau_normalized = np.clip(tau / self.max_tau, -1.0, 1.0)
        azimuth_rad = np.arcsin(tau_normalized)

        # 결과 저장
        self.azimuth_recon = np.array([azimuth_rad])
        self.tau_recon = tau

        # 그리드 값 저장 (호환성)
        if not hasattr(self, 'grid'):
            self.grid = type('obj', (object,), {'values': cc})()
        else:
            self.grid.values = cc
