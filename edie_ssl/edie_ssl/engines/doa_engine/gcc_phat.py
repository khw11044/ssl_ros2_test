"""
 Estimate time delay using GCC-PHAT
"""

import numpy as np


def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=1):
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

    # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
    # FFT 길이 (두 신호 길이의 합)
    n = sig.shape[0] + refsig.shape[0]

    # Generalized Cross Correlation Phase Transform
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    # Cross-correlation in frequency domain
    R = SIG * np.conj(REFSIG)

    # Phase Transform (정규화)
    cc = np.fft.irfft(R / np.abs(R), n=(interp * n))

    # Shift 범위 제한
    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    # 중심 정렬
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))

    # find max cross correlation index
    # 최대값 찾기
    shift = np.argmax(np.abs(cc)) - max_shift

    # 시간 지연 계산
    tau = shift / float(interp * fs)

    return tau, cc

def gcc_phat_multiband(
    sig,
    refsig,
    fs,
    max_tau=None,
    interp=1,
    hf_band=(2000, 6000),
    lf_band=(0, 800),
    energy_threshold_db=6.0,
    eps=1e-10,
):
    """
    Multi-band, event-gated GCC-PHAT for 2-mic SSL

    Returns:
        dict {
            "full": (tau, cc),
            "hf":   (tau, cc),
            "lf":   (tau, cc),
            "event": tau or None
        }
    """

    # ---------- FFT length ----------
    n = sig.shape[0] + refsig.shape[0]

    # ---------- FFT ----------
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)

    # ---------- GCC-PHAT core ----------
    R = SIG * np.conj(REFSIG)
    PHAT = R / (np.abs(R) + eps)

    # ---------- Frequency axis ----------
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    # ---------- Band weights ----------
    W_full = np.ones_like(freqs)

    W_hf = np.logical_and(freqs >= hf_band[0], freqs <= hf_band[1]).astype(float)
    W_lf = np.logical_and(freqs >= lf_band[0], freqs <= lf_band[1]).astype(float)

    # ---------- GCC per band ----------
    def gcc_from_weight(weight):
        cc = np.fft.irfft(PHAT * weight, n=interp * n)

        max_shift = int(interp * n / 2)
        if max_tau is not None:
            max_shift = min(int(interp * fs * max_tau), max_shift)

        cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
        shift = np.argmax(np.abs(cc)) - max_shift
        tau = shift / float(interp * fs)

        return tau, cc

    tau_full, cc_full = gcc_from_weight(W_full)
    tau_hf, cc_hf = gcc_from_weight(W_hf)
    tau_lf, cc_lf = gcc_from_weight(W_lf)

    # ---------- Energy-based event gating ----------
    def frame_energy_db(x):
        return 10 * np.log10(np.mean(x**2) + eps)

    energy_sig = frame_energy_db(sig)
    energy_ref = frame_energy_db(refsig)
    energy_db = max(energy_sig, energy_ref)

    # baseline은 외부에서 moving average로 관리하는 게 가장 좋다
    # 여기서는 함수 호출 시점 기준으로 "이 프레임이 충분히 큰가"만 판단
    event_tau = tau_full if energy_db >= energy_threshold_db else None

    return {
        "full": (tau_full, cc_full),
        "hf": (tau_hf, cc_hf),
        "lf": (tau_lf, cc_lf),
        "event": event_tau,
        "energy_db": energy_db,
    }


def main():
    refsig = np.linspace(1, 10, 10)

    for i in range(0, 10):
        sig = np.concatenate((np.linspace(0, 0, i), refsig, np.linspace(0, 0, 10 - i)))
        offset, _ = gcc_phat(sig, refsig)
        print(offset)


if __name__ == "__main__":
    main()
