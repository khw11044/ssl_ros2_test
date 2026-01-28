# -*- coding: utf-8 -*-

import os

if os.system('which arecord >/dev/null') != 0:
    from .mic_pyaudio import Source
else:
    from .mic_alsa import Source

__all__ = ['Source']
