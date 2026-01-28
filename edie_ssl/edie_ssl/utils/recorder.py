
import time
import wave
import numpy as np
from microphone.source import Source
from microphone.element import Element

class Recorder(Element):
    """오디오 데이터를 수집하는 Element"""

    def __init__(self):
        super().__init__()
        self.data = []

    def put(self, data):
        self.data.append(data)
        super().put(data)

    def get_audio(self):
        return b''.join(self.data)

    def clear(self):
        self.data = []
        
def main():
    rate = 16000
    channels = 2
    duration = 4  # 녹음 시간 (초)

    src = Source(rate=rate, frames_size=320, channels=channels)
    recorder = Recorder()

    src.link(recorder)

    print(f"Recording {duration} seconds - {channels} channels @ {rate}Hz")

    src.recursive_start()
    time.sleep(duration)
    src.recursive_stop()

    # WAV 파일 저장
    audio_data = recorder.get_audio()

    with wave.open('ch1_record.wav', 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16bit = 2bytes
        wf.setframerate(rate)
        # 2채널 데이터에서 채널 0 추출
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        ch1 = audio_array[0::2]  # 짝수 인덱스 = 채널 0
        wf.writeframes(ch1.tobytes())

    with wave.open('ch2_record.wav', 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        # 2채널 데이터에서 채널 1 추출
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        ch2 = audio_array[1::2]  # 홀수 인덱스 = 채널 1
        wf.writeframes(ch2.tobytes())

    print(f"Saved: ch1_record.wav, ch2_record.wav")
    print(f"Duration: {len(ch1) / rate:.2f} seconds")


if __name__ == '__main__':
    main()