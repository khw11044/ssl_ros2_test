"""
2채널 마이크를 사용한 실시간 음원 방향 추정 (DOA)
마이크 간 거리: 15.5cm
"""

import time
import collections
import numpy as np
from microphone.element import Element
from engines.doa_engine.gcc import gcc_phat
from engines.doa_engine.srp import SRP
from engines.sound_engine.calculation import calculate_rms_db
from engines.filter_engine.utility import get_filter, list_filters


class FilterElement(Element):
    """실시간 적응 필터 Element (시간 영역)

    파이프라인: Source → FilterElement → DOAEstimator
    """

    def __init__(self, filter_name='nlms', filter_length=32, channels=2, **kwargs):
        """
        Parameters
        ----------
        filter_name : str
            필터 이름 ('nlms', 'block_lms', 'rls', 'block_rls')
        filter_length : int
            필터 길이
        channels : int
            채널 수
        **kwargs : dict
            필터별 추가 파라미터 (mu, lmbd 등)
        """
        super().__init__()
        self.filter_name = filter_name
        self.filter_length = filter_length
        self.channels = channels
        self.kwargs = kwargs

        # 각 채널별 필터 생성
        self.filters = [
            get_filter(filter_name, filter_length, **kwargs)
            for _ in range(channels)
        ]

        # 워밍업 카운터
        self.sample_count = 0

    def put(self, data):
        audio = np.frombuffer(data, dtype='int16').astype(np.float64)

        # 무음이면 그대로 전달
        max_val = np.max(np.abs(audio))
        if max_val < 100:  # 무음 임계값
            super().put(data)
            return

        # 정규화
        audio = audio / max_val

        # 채널 분리
        channels_data = [audio[i::self.channels] for i in range(self.channels)]

        # 각 채널에 필터 적용
        filtered_channels = []
        for ch_idx, (ch_data, filt) in enumerate(zip(channels_data, self.filters)):
            filtered = np.zeros(len(ch_data))

            for i in range(len(ch_data)):
                self.sample_count += 1

                x_n = ch_data[i - 1] if i > 0 else 0.0
                d_n = ch_data[i]

                # 필터 업데이트 (try-except로 수치 오류 방지)
                try:
                    filt.update(x_n, d_n)
                except (FloatingPointError, ValueError):
                    filtered[i] = d_n
                    continue

                # 워밍업 후 예측
                if self.sample_count > self.filter_length * 2:
                    try:
                        if hasattr(filt.x, 'top'):
                            x_vec = filt.x.top(filt.length)
                            if len(x_vec) == filt.length:
                                norm = np.inner(x_vec, x_vec)
                                if norm > 1e-10:  # division by zero 방지
                                    prediction = np.inner(x_vec, filt.w)
                                    filtered[i] = d_n - prediction
                                else:
                                    filtered[i] = d_n
                            else:
                                filtered[i] = d_n
                        else:
                            norm = np.inner(filt.x, filt.x)
                            if norm > 1e-10:  # division by zero 방지
                                prediction = np.inner(filt.x, filt.w)
                                filtered[i] = d_n - prediction
                            else:
                                filtered[i] = d_n
                    except (FloatingPointError, ValueError):
                        filtered[i] = d_n
                else:
                    filtered[i] = d_n

            # NaN/Inf 제거
            filtered = np.nan_to_num(filtered, nan=0.0, posinf=1.0, neginf=-1.0)
            filtered = np.clip(filtered, -1.0, 1.0)
            filtered_channels.append(filtered)

        # 채널 인터리브 (L, R, L, R, ...)
        output = np.zeros(len(audio))
        for i, ch in enumerate(filtered_channels):
            output[i::self.channels] = ch

        # 스케일 복원 및 int16 변환 (클리핑 포함)
        output = output * max_val
        output = np.clip(output, -32768, 32767).astype(np.int16)

        # 다음 Element로 전달
        super().put(output.tobytes())

    def reset(self):
        """필터 상태 초기화"""
        self.filters = [
            get_filter(self.filter_name, self.filter_length, **self.kwargs)
            for _ in range(self.channels)
        ]
        self.sample_count = 0


class ILDEstimator(Element):
    """2채널 마이크용 ILD(Interaural Level Difference) 추정"""

    def __init__(self,
                 rate=16000,
                 channels=2,
                 threshold=500,
                 history_size=10,
                 calibration_chunks=0):
        super().__init__()
        self.sample_rate = rate
        self.channels = channels
        self.threshold = threshold
        self.history_size = history_size

        # ILD 히스토리 (이동 평균용)
        self.ild_history = collections.deque(maxlen=history_size)

        # 캘리브레이션
        self.calibration_chunks = calibration_chunks
        self.calibration_ild = []
        self.ild_offset = 0.0
        self.ild_std = 0.0
        self.is_calibrated = calibration_chunks == 0
        self.chunks_processed = 0

        # 현재 측정값
        self._db_left = None
        self._db_right = None
        self._ild_raw = None
        self._ild = None
        self._ild_avg = None
        self._is_silence = True

    def put(self, data):
        audio = np.frombuffer(data, dtype='int16')

        ch_left = audio[0::2]
        ch_right = audio[1::2]

        # dB 계산
        db_left = calculate_rms_db(ch_left)
        db_right = calculate_rms_db(ch_right)
        ild_raw = db_left - db_right

        # 캘리브레이션 단계
        if not self.is_calibrated:
            self.calibration_ild.append(ild_raw)
            self.chunks_processed += 1

            if self.chunks_processed >= self.calibration_chunks:
                self.ild_offset = np.mean(self.calibration_ild)
                self.ild_std = np.std(self.calibration_ild)
                self.is_calibrated = True
            return

        # 무음 필터링
        max_amp = max(np.max(np.abs(ch_left)), np.max(np.abs(ch_right)))
        if max_amp < self.threshold:
            self._is_silence = True
            super().put(data)
            return

        self._is_silence = False

        # ILD 계산 (캘리브레이션 오프셋 적용)
        ild = ild_raw - self.ild_offset

        # 이동 평균
        self.ild_history.append(ild)
        ild_avg = np.mean(self.ild_history)

        # 값 저장
        self._db_left = db_left
        self._db_right = db_right
        self._ild_raw = ild_raw
        self._ild = ild
        self._ild_avg = ild_avg

        super().put(data)

    def get_ild(self):
        """현재 ILD 값 반환 (calibrated, averaged)"""
        if self._is_silence:
            return None
        return self._ild_avg

    def get_ild_detail(self):
        """상세 ILD 정보 반환"""
        return {
            'db_left': self._db_left,
            'db_right': self._db_right,
            'ild_raw': self._ild_raw,
            'ild': self._ild,
            'ild_avg': self._ild_avg,
            'is_silence': self._is_silence,
            'is_calibrated': self.is_calibrated,
            'calibration_progress': self.chunks_processed / self.calibration_chunks if self.calibration_chunks > 0 else 1.0,
            'ild_offset': self.ild_offset,
            'ild_std': self.ild_std
        }

    def get_direction(self, left_th=-3, right_th=3):
        """ILD 기반 방향 반환"""
        ild = self.get_ild()
        if ild is None:
            return None

        if ild < left_th:
            return "LEFT"
        elif ild > right_th:
            return "RIGHT"
        else:
            return "CENTER"





class GCCEstimator(Element):
    """2채널 마이크용 DOA : GCC + 상대적 dB 이벤트 감지"""

    def __init__(self,
                 rate=16000,
                 channels=2,
                 chunks=20,
                 max_tdoa=340.0/0.155,
                 silence_threshold=100,
                 db_history_size=50,
                 db_threshold=6.0,
                 direction_threshold=30.0
                 ):
        super().__init__()
        self.queue = collections.deque(maxlen=chunks)
        self.sample_rate = rate
        self.channels = channels
        self.max_tdoa = max_tdoa
        self.silence_threshold = silence_threshold

        # 상대적 dB 계산을 위한 히스토리
        self.db_history = collections.deque(maxlen=db_history_size)
        self.db_threshold = db_threshold  # 상대적 dB 임계값

        # 상대적 방향 계산을 위한 히스토리
        self.direction_history = collections.deque(maxlen=db_history_size)
        self.direction_threshold = direction_threshold  # 상대적 방향 임계값

        # 이벤트 쿨다운 (연속 이벤트 방지)
        self.last_event_time = 0
        self.event_cooldown = 0.5  # 초
        

    def put(self, data):
        self.queue.append(data)
        super().put(data)

    def get_direction(self):
        if len(self.queue) < 5:
            return None

        buf = b''.join(self.queue)
        buf = np.frombuffer(buf, dtype='int16')

        # 신호 크기 체크 (무음 필터링)
        if np.max(np.abs(buf)) < self.silence_threshold:
            return None

        ch0 = buf[1::2]
        ch1 = buf[0::2]

        tau, _ = gcc_phat(ch0, ch1, fs=self.sample_rate, max_tau=self.max_tdoa, interp=4)

        # tau를 각도로 변환 (-90 ~ +90도)
        ratio = np.clip(tau / self.max_tdoa, -1, 1)
        theta = np.arcsin(ratio) * 180 / np.pi

        return theta
    
    def get_direction_and_event(self):
        """방향과 상대적 dB 이벤트 반환"""
        if len(self.queue) < 5:
            return None, None, None, None

        buf = b''.join(self.queue)
        buf = np.frombuffer(buf, dtype='int16')

        ch0 = buf[1::2]
        ch1 = buf[0::2]

        # 현재 dB 계산 (두 채널 평균)
        current_db = (calculate_rms_db(ch0) + calculate_rms_db(ch1)) / 2

        # 히스토리에 추가
        self.db_history.append(current_db)

        # 평균 dB 계산
        if len(self.db_history) < 10:
            avg_db = current_db
        else:
            avg_db = np.mean(list(self.db_history)[:-1])  # 현재 제외한 평균

        # 상대적 dB
        relative_db = current_db - avg_db

        # 무음 필터링
        if np.max(np.abs(buf)) < 500:
            return None, current_db, avg_db, relative_db

        # GCC-PHAT DOA
        tau, _ = gcc_phat(ch0, ch1, fs=self.sample_rate, max_tau=self.max_tdoa , interp=4)
        ratio = np.clip(tau / self.max_tdoa, -1, 1)
        theta = np.arcsin(ratio) * 180 / np.pi

        # 방향 히스토리에 추가
        self.direction_history.append(theta)

        # 평균 및 표준편차 계산
        if len(self.direction_history) < 10:
            avg_direction = theta
            std_direction = 1.0  # 초기값
        else:
            history = list(self.direction_history)[:-1]
            avg_direction = np.mean(history)
            std_direction = np.std(history)
            if std_direction < 1.0:  # 표준편차가 너무 작으면 최소값 설정
                std_direction = 1.0

        # Z-score 정규화 (평균 0, 표준편차 1 기준)
        # |z| > 2 이면 95% 신뢰구간 밖 = 비정상적 변화
        z_score_shifting = theta - avg_direction
        norm_direction = z_score_shifting / std_direction

        # 이벤트 감지 (dB 변화 OR 방향 변화)
        event = None
        event_reason = None
        current_time = time.time()

        if current_time - self.last_event_time > self.event_cooldown:
            # # dB 기반 이벤트
            # if relative_db >= self.db_threshold:
            #     self.last_event_time = current_time
            #     event_reason = f"dB Δ{relative_db:+.1f}"
            #     if norm_direction < -self.direction_threshold:
            #         event = "RIGHT"
            #     elif norm_direction > self.direction_threshold:
            #         event = "LEFT"
            #     else:
            #         event = "CENTER"

            # 방향 변화 기반 이벤트 (dB도 어느정도 있어야 함)
            if abs(norm_direction) >= self.direction_threshold and relative_db >= self.db_threshold / 2:
                self.last_event_time = current_time
                event_reason = f"Z-score {norm_direction:+.2f}"
                if norm_direction < 0:
                    event = "LEFT"
                else:
                    event = "RIGHT"

        return theta, avg_direction, z_score_shifting, norm_direction, current_db, avg_db, relative_db, event, event_reason


class SRPEstimator(Element):
    """실시간 SRP-PHAT DOA (RealtimeSRPUDP와 동일한 로직, UDP 제외)"""

    def __init__(self, rate=16000, mic_dis=0.155, chunks=10, nfft=512, n_grid=18,
                 alpha=0.3, confidence_threshold=1.5):
        super().__init__()
        self.queue = collections.deque(maxlen=chunks)
        self.sample_rate = rate
        self.nfft = nfft
        self.n_grid = n_grid
        self.mic_dis = mic_dis

        # 후처리 파라미터
        self.alpha = alpha  # EMA 스무딩 계수 (0~1, 낮을수록 부드러움)
        self.confidence_threshold = confidence_threshold  # 신뢰도 임계값
        self.prev_values = None  # 이전 스무딩 값

        # 마이크 배열 생성
        L = np.array([
            [-self.mic_dis/2, self.mic_dis/2],  # x
            [0, 0]                       # y
        ])

        # SRP 객체 생성 (전체 360도로 계산)
        SOUND_SPEED = 340.0
        self.srp = SRP(
            L=L,
            fs=rate,
            nfft=nfft,
            c=SOUND_SPEED,
            num_src=1,
            mode='far',
            n_grid=n_grid * 2  # 360도 전체
        )

        # 각도 배열 미리 계산 (0~180도만 사용)
        angles_rad = self.srp.grid.azimuth
        angles_deg = np.degrees(angles_rad)
        angles_deg = (angles_deg + 360) % 360

        # 0~180도 범위만 필터링
        mask = angles_deg <= 180
        self.half_idx = np.where(mask)[0]
        self.angles_deg = angles_deg[mask]
        self.sort_idx = np.argsort(self.angles_deg)
        self.angles_deg = self.angles_deg[self.sort_idx]
        self.half_idx = self.half_idx[self.sort_idx]

    def put(self, data):
        self.queue.append(data)
        super().put(data)

    def compute_stft(self, signal):
        """단일 프레임 STFT"""
        window = np.hanning(self.nfft)
        hop_size = self.nfft // 2
        num_frames = max(1, (len(signal) - self.nfft) // hop_size + 1)

        stft = np.zeros((self.nfft // 2 + 1, num_frames), dtype=np.complex128)

        for i in range(num_frames):
            start = i * hop_size
            if start + self.nfft <= len(signal):
                frame = signal[start:start + self.nfft] * window
                stft[:, i] = np.fft.rfft(frame)

        return stft

    def get_srp_distribution(self):
        """SRP-PHAT 분포 계산 (0~180도만)"""
        if len(self.queue) < 3:
            return None, None, None, None

        buf = b''.join(self.queue)
        buf = np.frombuffer(buf, dtype='int16')

        # 무음 필터링
        if np.max(np.abs(buf)) < 500:
            return None, None, None, None

        ch0 = buf[0::2].astype(np.float64)
        ch1 = buf[1::2].astype(np.float64)

        # STFT 계산
        X0 = self.compute_stft(ch0)
        X1 = self.compute_stft(ch1)

        # (n_mics, n_freq, n_frames) 형태로 결합
        X = np.stack([X0, X1], axis=0)

        # SRP-PHAT 실행
        self.srp.locate_sources(X, freq_range=[300, 3500])

        # 결과 추출 (0~180도만)
        all_values = self.srp.grid.values
        values = all_values[self.half_idx]

        # 피크 찾기
        peak_idx = np.argmax(values)
        peak_angle = self.angles_deg[peak_idx]

        # GCC-PHAT 호환 각도 (-90~90)
        # 0° → -90° (RIGHT), 90° → 0° (FRONT), 180° → +90° (LEFT)
        gcc_angle = peak_angle - 90

        return self.angles_deg, values, peak_angle, gcc_angle

    def get_srp_distribution_norm(self):
        """후처리 포함된 SRP-PHAT 분포 계산 (스무딩 + 정규화 + 신뢰도)"""
        if len(self.queue) < 3:
            return None

        buf = b''.join(self.queue)
        buf = np.frombuffer(buf, dtype='int16')

        # 무음 필터링
        if np.max(np.abs(buf)) < 500:
            return None

        # ch0 = buf[0::2].astype(np.float64)
        # ch1 = buf[1::2].astype(np.float64)
        
        ch0 = buf[1::2].astype(np.float64)
        ch1 = buf[0::2].astype(np.float64)

        # STFT 계산
        X0 = self.compute_stft(ch0)
        X1 = self.compute_stft(ch1)

        # (n_mics, n_freq, n_frames) 형태로 결합
        X = np.stack([X0, X1], axis=0)

        # SRP-PHAT 실행
        self.srp.locate_sources(X, freq_range=[300, 3500])

        # 결과 추출 (0~180도만)
        all_values = self.srp.grid.values
        raw_values = all_values[self.half_idx]

        # 스무딩 적용
        smoothed_values = self.apply_smoothing(raw_values)

        # 정규화 (0~1)
        min_val = smoothed_values.min()
        max_val = smoothed_values.max()
        if max_val - min_val > 1e-10:
            norm_values = (smoothed_values - min_val) / (max_val - min_val)
        else:
            norm_values = np.zeros_like(smoothed_values)

        # 신뢰도 계산 (스무딩된 값 기준)
        confidence = self.compute_confidence(smoothed_values)
        is_confident = bool(confidence > 0.5)

        # 피크 찾기
        peak_idx = np.argmax(smoothed_values)
        peak_angle = float(self.angles_deg[peak_idx])
        gcc_angle = peak_angle - 90  # GCC-PHAT 호환 (-90~90)
        direction = self.angle_to_direction(peak_angle, is_confident)

        return {
            'angles': self.angles_deg,
            'values': norm_values,
            'raw_values': raw_values,
            'peak_angle': peak_angle,
            'gcc_angle': gcc_angle,
            'direction': direction,
            'confidence': confidence,
            'is_confident': is_confident
        }

    def angle_to_direction(self, angle, confident):
        """SRP 각도(0~180)를 방향으로 변환"""
        if not confident:
            return "UNCERTAIN"
        if angle < 60:
            return "RIGHT"
        elif angle > 120:
            return "LEFT"
        else:
            return "FRONT"

    def compute_confidence(self, values):
        """
        피크 신뢰도 계산
        - 분포가 뾰족할수록 높은 신뢰도
        - peak/mean 비율 사용
        """
        mean_val = np.mean(values)
        max_val = np.max(values)

        if mean_val < 1e-10:
            return 0.0

        # 피크 대 평균 비율
        peak_to_mean = max_val / mean_val

        # 신뢰도를 0~1로 정규화 (threshold 기준)
        # ratio가 threshold 이상이면 confidence = 1
        confidence = min(1.0, (peak_to_mean - 1.0) / (self.confidence_threshold - 1.0))
        confidence = max(0.0, confidence)

        return confidence


    def apply_smoothing(self, current_values):
        """EMA 스무딩 적용"""
        if self.prev_values is None:
            self.prev_values = current_values.copy()
            return current_values

        # Exponential Moving Average
        smoothed = self.alpha * current_values + (1 - self.alpha) * self.prev_values
        self.prev_values = smoothed.copy()

        return smoothed

    def process_and_send(self):
        """SRP-PHAT 계산 및 UDP 송신"""
        if len(self.queue) < 3:
            return None

        buf = b''.join(self.queue)
        buf = np.frombuffer(buf, dtype='int16')

        # 무음 필터링
        if np.max(np.abs(buf)) < 500:
            return None

        # ch0 = buf[0::2].astype(np.float64)
        # ch1 = buf[1::2].astype(np.float64)
        
        ch0 = buf[1::2].astype(np.float64)
        ch1 = buf[0::2].astype(np.float64)

        # STFT 계산
        X0 = self.compute_stft(ch0)
        X1 = self.compute_stft(ch1)

        # (n_mics, n_freq, n_frames) 형태로 결합
        X = np.stack([X0, X1], axis=0)

        # SRP-PHAT 실행
        self.srp.locate_sources(X, freq_range=[300, 3500])

        # 결과 추출 (0~180도만)
        all_values = self.srp.grid.values
        raw_values = all_values[self.half_idx]

        # 스무딩 적용
        smoothed_values = self.apply_smoothing(raw_values)

        # 정규화 (0~1)
        min_val = smoothed_values.min()
        max_val = smoothed_values.max()
        if max_val - min_val > 1e-10:
            norm_values = (smoothed_values - min_val) / (max_val - min_val)
        else:
            norm_values = np.zeros_like(smoothed_values)

        # 신뢰도 계산 (스무딩된 값 기준)
        confidence = self.compute_confidence(smoothed_values)
        is_confident = bool(confidence > 0.5)

        # 피크 찾기
        peak_idx = np.argmax(smoothed_values)
        peak_angle = float(self.angles_deg[peak_idx])
        gcc_angle = peak_angle - 90  # GCC-PHAT 호환 (-90~90)
        direction = self.angle_to_direction(peak_angle, is_confident)

        # UDP 메시지 생성
        if self.udp_format == 'json':
            msg = {
                'timestamp': time.time(),
                'angles': [float(a) for a in self.angles_deg],
                'values': [float(v) for v in norm_values],
                'raw_values': [float(v) for v in raw_values],
                'peak_angle': peak_angle,
                'gcc_angle': gcc_angle,
                'direction': direction,
                'peak_value': float(norm_values[peak_idx]),
                'confidence': round(confidence, 3),
                'is_confident': is_confident
            }
            data = json.dumps(msg).encode()
        else:  # csv
            # 형식: peak_angle,gcc_angle,direction,confidence,val0,val1,val2,...
            vals_str = ','.join([f'{v:.4f}' for v in norm_values])
            data = f'{peak_angle:.1f},{gcc_angle:.1f},{direction},{confidence:.3f},{vals_str}'.encode()

        # UDP 송신
        try:
            self.sock.sendto(data, (self.udp_host, self.udp_port))
            self.send_count += 1
        except Exception as e:
            print(f"UDP send error: {e}")

        return {
            'angles': self.angles_deg,
            'values': norm_values,
            'peak_angle': peak_angle,
            'gcc_angle': gcc_angle,
            'direction': direction,
            'confidence': confidence,
            'is_confident': is_confident
        }
        
        
