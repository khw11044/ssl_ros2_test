"""
FIR Filter (Finite Impulse Response Filter)
============================================

선형 위상(Linear Phase) 특성을 가진 FIR 필터 구현.
filtfilt를 사용하여 영 위상(Zero-Phase) 필터링 지원.

사용법:
    from engines.filter_engine.GeneralFilters import FIRFilter

    # 저역 통과 필터
    fir = FIRFilter(sample_rate=16000, filter_type='lowpass', cutoff=3000)
    filtered = fir.filter(audio_data)

    # 영 위상 필터링 (위상 왜곡 없음)
    filtered = fir.filtfilt(audio_data)
"""

import numpy as np
from scipy import signal


class FIRFilter:
    """
    FIR 필터 클래스

    Parameters
    ----------
    sample_rate : int
        샘플링 레이트 (Hz)
    filter_type : str
        필터 종류: 'lowpass', 'highpass', 'bandpass', 'bandstop', 'notch'
    cutoff : float or tuple
        차단 주파수 (Hz). bandpass/bandstop은 (low, high) 튜플
    num_taps : int
        필터 탭 수 (홀수, 기본값: 255)
    window : str
        윈도우 함수 (기본값: 'hamming')
    """

    FILTER_TYPES = ['lowpass', 'highpass', 'bandpass', 'bandstop', 'notch']

    def __init__(self, sample_rate, filter_type='lowpass', cutoff=3000,
                 num_taps=255, window='hamming'):
        self.sample_rate = sample_rate
        self.filter_type = filter_type
        self.cutoff = cutoff
        self.num_taps = num_taps if num_taps % 2 == 1 else num_taps + 1
        self.window = window
        self.nyquist = sample_rate / 2.0

        # 필터 계수 설계
        self.coeffs = self._design_filter()

    def _design_filter(self):
        """FIR 필터 계수 설계"""
        if self.filter_type == 'lowpass':
            normalized = self.cutoff / self.nyquist
            return signal.firwin(self.num_taps, normalized,
                                 window=self.window, pass_zero='lowpass')

        elif self.filter_type == 'highpass':
            normalized = self.cutoff / self.nyquist
            return signal.firwin(self.num_taps, normalized,
                                 window=self.window, pass_zero='highpass')

        elif self.filter_type == 'bandpass':
            if isinstance(self.cutoff, (list, tuple)):
                low, high = self.cutoff
            else:
                low = self.cutoff
                high = min(self.cutoff * 10, self.nyquist * 0.95)
            normalized = [low / self.nyquist, high / self.nyquist]
            return signal.firwin(self.num_taps, normalized,
                                 window=self.window, pass_zero='bandpass')

        elif self.filter_type == 'bandstop':
            if isinstance(self.cutoff, (list, tuple)):
                low, high = self.cutoff
            else:
                low = self.cutoff * 0.9
                high = self.cutoff * 1.1
            normalized = [low / self.nyquist, high / self.nyquist]
            return signal.firwin(self.num_taps, normalized,
                                 window=self.window, pass_zero='bandstop')

        elif self.filter_type == 'notch':
            # 노치 필터: 특정 주파수 ±5Hz 제거
            notch_width = 5
            low = (self.cutoff - notch_width) / self.nyquist
            high = (self.cutoff + notch_width) / self.nyquist
            return signal.firwin(self.num_taps, [low, high],
                                 window=self.window, pass_zero='bandstop')

        else:
            raise ValueError(f"Unknown filter type: {self.filter_type}. "
                             f"Choose from {self.FILTER_TYPES}")

    def filter(self, data):
        """
        순방향 필터링 (위상 지연 있음)

        Parameters
        ----------
        data : np.ndarray
            입력 신호 (int16 또는 float)

        Returns
        -------
        np.ndarray
            필터링된 신호 (입력과 동일한 dtype)
        """
        is_int16 = data.dtype == np.int16
        if is_int16:
            audio = data.astype(np.float64) / 32768.0
        else:
            audio = data.astype(np.float64)

        filtered = signal.lfilter(self.coeffs, 1.0, audio)

        if is_int16:
            return np.clip(filtered * 32768.0, -32768, 32767).astype(np.int16)
        return filtered

    def filtfilt(self, data):
        """
        영 위상 필터링 (Zero-Phase, 위상 왜곡 없음)

        순방향 + 역방향 필터링으로 위상 지연을 완전히 제거.
        오프라인(녹음된 오디오) 처리에 적합.

        Parameters
        ----------
        data : np.ndarray
            입력 신호 (int16 또는 float)

        Returns
        -------
        np.ndarray
            필터링된 신호 (입력과 동일한 dtype)
        """
        is_int16 = data.dtype == np.int16
        if is_int16:
            audio = data.astype(np.float64) / 32768.0
        else:
            audio = data.astype(np.float64)

        filtered = signal.filtfilt(self.coeffs, 1.0, audio)

        if is_int16:
            return np.clip(filtered * 32768.0, -32768, 32767).astype(np.int16)
        return filtered

    def get_frequency_response(self, num_points=512):
        """
        주파수 응답 반환

        Returns
        -------
        freqs : np.ndarray
            주파수 배열 (Hz)
        response : np.ndarray
            크기 응답 (dB)
        """
        w, h = signal.freqz(self.coeffs, worN=num_points)
        freqs = w * self.sample_rate / (2 * np.pi)
        response = 20 * np.log10(np.abs(h) + 1e-10)
        return freqs, response

    def reset(self):
        """필터 상태 초기화 (재설계)"""
        self.coeffs = self._design_filter()

    def __repr__(self):
        return (f"FIRFilter(type={self.filter_type}, cutoff={self.cutoff}Hz, "
                f"taps={self.num_taps}, fs={self.sample_rate}Hz)")


# 편의 함수들
def lowpass(data, sample_rate, cutoff, num_taps=255, zero_phase=True):
    """저역 통과 필터 (고주파 노이즈 제거)"""
    fir = FIRFilter(sample_rate, 'lowpass', cutoff, num_taps)
    return fir.filtfilt(data) if zero_phase else fir.filter(data)


def highpass(data, sample_rate, cutoff, num_taps=255, zero_phase=True):
    """고역 통과 필터 (저주파 험/진동 제거)"""
    fir = FIRFilter(sample_rate, 'highpass', cutoff, num_taps)
    return fir.filtfilt(data) if zero_phase else fir.filter(data)


def bandpass(data, sample_rate, low_cutoff, high_cutoff, num_taps=255, zero_phase=True):
    """대역 통과 필터 (특정 주파수 대역만 유지)"""
    fir = FIRFilter(sample_rate, 'bandpass', (low_cutoff, high_cutoff), num_taps)
    return fir.filtfilt(data) if zero_phase else fir.filter(data)


def bandstop(data, sample_rate, low_cutoff, high_cutoff, num_taps=255, zero_phase=True):
    """대역 저지 필터 (특정 주파수 대역 제거)"""
    fir = FIRFilter(sample_rate, 'bandstop', (low_cutoff, high_cutoff), num_taps)
    return fir.filtfilt(data) if zero_phase else fir.filter(data)


def notch(data, sample_rate, freq, num_taps=255, zero_phase=True):
    """노치 필터 (특정 주파수 제거, 예: 60Hz 험)"""
    fir = FIRFilter(sample_rate, 'notch', freq, num_taps)
    return fir.filtfilt(data) if zero_phase else fir.filter(data)
