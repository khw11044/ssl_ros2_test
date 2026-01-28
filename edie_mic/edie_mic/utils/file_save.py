# -*- coding: utf-8 -*-

"""
Save audio to file
"""

import wave
import time
from microphone.element import Element
from microphone.source import Source

class FileSink(Element):
    def __init__(self, name, rate=16000, channels=1, width=2):
        super(FileSink, self).__init__()
        self.file_name = name
        self.rate = rate
        self.channels = channels
        self.width = width

        self._wav = None

    def put(self, data):
        self._wav.writeframes(data)

    def start(self):
        self._wav = wave.open(self.file_name, 'w')
        self._wav.setframerate(self.rate)
        self._wav.setnchannels(self.channels)
        self._wav.setsampwidth(self.width)

    def stop(self):
        self._wav.close()


def main():
    duration = 5  # 5초 녹음

    src = Source(rate=16000, channels=1)
    sink = FileSink("output.wav", rate=16000, channels=1)

    src.link(sink)  # Source → Sink 연결

    src.recursive_start()
    time.sleep(duration)  # 5초 동안 녹음
    src.recursive_stop()