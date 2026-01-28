# -*- coding: utf-8 -*-
"""Audio source that reads .wav file"""

import logging
import wave
import numpy as np
from microphone.element import Element


logger = logging.getLogger(__file__)


class Loader(Element):
    def __init__(self, wav, frames_size=160):
        super(Loader, self).__init__()

        self._wav = wave.open(wav, 'r')
        self.channels = self._wav.getnchannels()
        self.rate = self._wav.getframerate()
        self.frames_size = frames_size
        self.done = False
        
    @staticmethod
    def load_wav(filename):
        """WAV 파일 로드"""
        with wave.open(filename, 'rb') as wf:
            rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
            data = np.frombuffer(frames, dtype=np.int16)
        return data, rate

