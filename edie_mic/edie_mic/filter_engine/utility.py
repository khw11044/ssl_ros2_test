"""
적응 필터 팩토리 및 적용 함수

시간 영역 필터: NLMS, BlockLMS, RLS, BlockRLS
주파수 영역 필터: SubbandLMS (apply_subband_filter 사용)
"""

import inspect
import numpy as np

from .AdaptiveFilters.lms import NLMS, BlockLMS
from .AdaptiveFilters.rls import RLS, BlockRLS
from .AdaptiveFilters.subband_lms import SubbandLMS


# 사용 가능한 필터 목록 (시간 영역만)
AVAILABLE_FILTERS = {
    'nlms': NLMS,
    'block_lms': BlockLMS,
    'rls': RLS,
    'block_rls': BlockRLS,
}


def _filter_kwargs(cls, kwargs):
    """클래스가 받을 수 있는 파라미터만 필터링"""
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self'}
    return {k: v for k, v in kwargs.items() if k in valid_params}


def get_filter(name, length=32, **kwargs):
    """
    필터 인스턴스 생성

    Parameters
    ----------
    name : str
        필터 이름 ('nlms', 'block_lms', 'rls', 'block_rls')
    length : int
        필터 길이
    **kwargs : dict
        필터별 추가 파라미터 (해당 필터가 지원하는 파라미터만 자동 적용)
        - NLMS: mu (default 0.5)
        - BlockLMS: mu, L, nlms
        - RLS: lmbd, delta, dtype
        - BlockRLS: lmbd, delta, dtype, L

    Returns
    -------
    AdaptiveFilter
        필터 인스턴스
    """
    name = name.lower()
    if name not in AVAILABLE_FILTERS:
        raise ValueError(f"Unknown filter: {name}. Available: {list(AVAILABLE_FILTERS.keys())}")

    filter_class = AVAILABLE_FILTERS[name]
    filtered_kwargs = _filter_kwargs(filter_class, kwargs)
    return filter_class(length, **filtered_kwargs)


def apply_filter(signal, filter_name='nlms', filter_length=32, **kwargs):
    """
    적응 필터를 사용한 노이즈 제거 (예측 에러 필터)

    Parameters
    ----------
    signal : np.ndarray
        입력 신호
    filter_name : str
        필터 이름 ('nlms', 'block_lms', 'rls', 'block_rls')
    filter_length : int
        필터 길이
    **kwargs : dict
        필터별 추가 파라미터

    Returns
    -------
    np.ndarray
        필터링된 신호
    """
    # 신호 정규화
    signal = signal.astype(np.float64)
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val

    # 필터 생성
    adaptive_filter = get_filter(filter_name, filter_length, **kwargs)

    filtered = np.zeros(len(signal))

    for i in range(filter_length, len(signal)):
        x_n = signal[i - 1]  # 이전 샘플
        d_n = signal[i]      # 현재 샘플 (desired)

        # 필터 업데이트 먼저 (버퍼 채우기)
        adaptive_filter.update(x_n, d_n)

        # 예측값 계산 (Buffer 타입 vs ndarray 분기)
        if hasattr(adaptive_filter.x, 'top'):
            # RLS 계열: Buffer에서 배열 추출
            x_vec = adaptive_filter.x.top(adaptive_filter.length)
            # 워밍업 중이면 skip
            if len(x_vec) != adaptive_filter.length:
                continue
        else:
            # LMS 계열: 직접 ndarray
            x_vec = adaptive_filter.x

        prediction = np.inner(x_vec, adaptive_filter.w)

        # 에러 (예측되지 않은 부분 = 원하는 신호)
        error = d_n - prediction
        filtered[i] = error

    # 스케일 복원
    filtered = filtered * max_val

    return filtered.astype(np.int16)


def list_filters():
    """사용 가능한 필터 목록 반환 (시간 영역)"""
    return list(AVAILABLE_FILTERS.keys())


def apply_subband_filter(primary, reference, num_taps=4, nfft=256, hop_size=None, mu=0.5):
    """
    Subband LMS를 사용한 주파수 영역 적응 필터링

    Parameters
    ----------
    primary : np.ndarray
        주 신호 (desired)
    reference : np.ndarray
        참조 신호 (input)
    num_taps : int
        각 서브밴드 필터의 탭 수
    nfft : int
        FFT 크기
    hop_size : int
        홉 사이즈 (default: nfft // 2)
    mu : float
        스텝 사이즈

    Returns
    -------
    np.ndarray
        필터링된 신호
    """
    if hop_size is None:
        hop_size = nfft // 2

    primary = primary.astype(np.float64)
    reference = reference.astype(np.float64)

    # 정규화
    max_val = max(np.max(np.abs(primary)), np.max(np.abs(reference)))
    if max_val > 0:
        primary = primary / max_val
        reference = reference / max_val

    num_bands = nfft // 2 + 1

    # Subband LMS 필터 초기화
    subband_lms = SubbandLMS(num_taps=num_taps, num_bands=num_bands, mu=mu, nlms=True)

    # 윈도우 함수
    window = np.hanning(nfft)

    # 출력 버퍼
    output = np.zeros(len(primary))
    output_window = np.zeros(len(primary))

    # STFT 기반 처리
    num_frames = (len(primary) - nfft) // hop_size + 1

    for frame_idx in range(num_frames):
        start = frame_idx * hop_size
        end = start + nfft

        # 현재 프레임 추출
        primary_frame = primary[start:end] * window
        reference_frame = reference[start:end] * window

        # FFT
        X_n = np.fft.rfft(reference_frame)  # 참조 신호 (입력)
        D_n = np.fft.rfft(primary_frame)    # 주 신호 (desired)

        # Subband LMS 업데이트
        subband_lms.update(X_n, D_n)

        # 에러 신호 -> 시간 도메인 변환
        error_frame = np.fft.irfft(subband_lms.E)

        # Overlap-add
        output[start:end] += error_frame * window
        output_window[start:end] += window ** 2

    # 윈도우 보정
    output_window[output_window < 1e-8] = 1
    output = output / output_window

    # 스케일 복원
    output = output * max_val

    return output.astype(np.int16)
