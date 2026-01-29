# -*- coding: utf-8 -*-
"""Audio source using PulseAudio `parecord` - allows specifying exact source device"""

import subprocess
import threading

from .element import Element


class Source(Element):
    """Audio source that records audio using `parecord`

    Unlike sounddevice, parecord allows specifying the exact PulseAudio source
    without relying on environment variables, enabling multiple sources simultaneously.
    """

    def __init__(self, rate=16000, frames_size=None, channels=1, device_name=None, bits_per_sample=16):
        """Set parameters of recording

        Args:
            rate: sample rate
            frames_size: number of each channel's frames per chunk
            channels: channels' number
            device_name: PulseAudio source name (from: pactl list sources short)
                         e.g., 'alsa_input.platform-rt5651-sound.stereo-fallback', 'DeepFilterMic'
            bits_per_sample: sample width - 16 (s16le) or 32 (float32le)
        """
        super(Source, self).__init__()

        self.rate = rate
        self.frames_size = frames_size if frames_size else rate // 100
        self.channels = channels
        self.device_name = device_name
        self.bits_per_sample = bits_per_sample
        self.bytes_per_sample = bits_per_sample // 8

        # parecord format mapping
        if bits_per_sample == 32:
            self.format = 'float32le'
        elif bits_per_sample == 24:
            self.format = 's24le'
        elif bits_per_sample == 16:
            self.format = 's16le'
        else:
            self.format = 's16le'

        self.done = False
        self.thread = None
        self.process = None

    def run(self):
        """Run `parecord` and read its stdout pipe"""
        cmd = [
            'parecord',
            '--raw',
            '--format=' + self.format,
            '--channels=' + str(self.channels),
            '--rate=' + str(self.rate),
        ]

        if self.device_name:
            cmd.append('--device=' + self.device_name)

        print(f'[parecord] {" ".join(cmd)}')
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

        frames_bytes = int(self.frames_size * self.channels * self.bytes_per_sample)
        while not self.done:
            audio = self.process.stdout.read(frames_bytes)
            if not audio:
                break
            super(Source, self).put(audio)

        if self.process:
            self.process.kill()

    def start(self):
        """Start a recording thread"""
        self.done = False
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """Stop recording"""
        self.done = True
        if self.process:
            self.process.kill()
            self.process = None
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3)
