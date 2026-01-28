# -*- coding: utf-8 -*-
"""Audio source using sounddevice (PortAudio wrapper)"""

import sounddevice as sd
import numpy as np

from .element import Element


class Source(Element):
    """Audio source that records audio using sounddevice

    sounddevice provides better compatibility with PipeWire/PulseAudio
    and modern Python versions (3.10+)
    """

    def __init__(self, rate=16000, frames_size=None, channels=1, device_name=None, bits_per_sample=16):
        """Setup a sounddevice callback stream to record audio

        Args:
            rate: sample rate
            frames_size: number of each channel's frames per chunk
            channels: channels' number
            device_name: device name to search (e.g., 'pulse', 'default')
            bits_per_sample: sample width - 16 or 32 (float)
        """
        super(Source, self).__init__()

        self.rate = rate
        self.frames_size = frames_size if frames_size else rate // 100
        self.channels = channels
        self.device_name = device_name if device_name else 'default'
        self.bits_per_sample = bits_per_sample

        # Select dtype based on bits_per_sample
        if bits_per_sample == 32:
            self.dtype = 'float32'
        elif bits_per_sample == 24:
            self.dtype = 'int24'  # Note: limited support
        elif bits_per_sample == 16:
            self.dtype = 'int16'
        else:
            self.dtype = 'int8'

        self.stream = sd.InputStream(
            samplerate=self.rate,
            channels=self.channels,
            dtype=self.dtype,
            blocksize=self.frames_size,
            device=self.device_name,
            callback=self._callback
        )

    def _callback(self, indata, frames, time, status):
        """sounddevice stream callback"""
        if status:
            print(f'Audio status: {status}')
        super(Source, self).put(indata.tobytes())

    def start(self):
        """Start recording"""
        self.stream.start()

    def stop(self):
        """Stop recording"""
        self.stream.stop()
        self.stream.close()
