"""
General Filters
===============

범용 필터 모듈. FIR, Kalman, Bayes, One Euro 등 다양한 필터 제공.

사용법:
    from engines.filter_engine.GeneralFilters import FIRFilter
    from engines.filter_engine.GeneralFilters import lowpass, highpass, bandpass, notch

    # 클래스 사용
    fir = FIRFilter(sample_rate=16000, filter_type='lowpass', cutoff=3000)
    filtered = fir.filtfilt(audio)

    # 편의 함수 사용
    filtered = lowpass(audio, sample_rate=16000, cutoff=3000)
"""

from .firfilter import (
    FIRFilter,
    lowpass,
    highpass,
    bandpass,
    bandstop,
    notch,
)

__all__ = [
    'FIRFilter',
    'lowpass',
    'highpass',
    'bandpass',
    'bandstop',
    'notch',
]
