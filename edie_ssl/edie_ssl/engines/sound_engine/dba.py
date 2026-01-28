# -*- coding: utf-8 -*-
#
# dB(A) A-weighting
#
# Requirements:
#   pip install numpy scipy


"""
# 주요 기능

A-weighting은 인간의 청각 특성을 반영한 주파수 가중치입니다. 
사람의 귀는 모든 주파수를 동일하게 인식하지 않기 때문에, 실제 청감에 맞는 소음 레벨을 측정할 때 사용합니다.

## A_weighting_
저주파(~500Hz 이하)와 고주파(~6kHz 이상)를 감쇠
1~6kHz 대역(인간이 민감한 영역)을 상대적으로 강조

dBFS: 녹음 레벨 모니터링, 클리핑 방지, 디지털 오디오 처리
dB(A): 소음 측정, 환경 소음 평가, 법적 소음 기준 (대부분 dB(A) 기준)

"""

import numpy as np
from microphone.element import Element
from .calculation import A_weighting_coeffs, A_weight

class DBA(Element):
    def __init__(self, rate, channels, bits_per_sample=16):
        super(DBA, self).__init__()
        self.rate = rate
        self.channels = channels
        if bits_per_sample == 32:
            self.type = 'int32'
            self.width = 4
            self.top = 20 * np.log10(2 ** 31 - 1)
        else:
            self.type = 'int16'
            self.width = 2
            self.top = 20 * np.log10(2 ** 15 - 1)

    def put(self, data):
        buf = np.fromstring(data, dtype=self.type)
        v = [[], []]
        for ch in range(self.channels):
            mono = buf[ch::self.channels]
            # dbfs = 20 * np.log10(np.sqrt(np.mean(np.square(mono, dtype='float')))) - self.top
            dbfs = 10 * np.log10(np.mean(np.square(mono, dtype='float'))) - self.top
            v[0].append(int(dbfs))

            w = A_weight(mono, self.rate)

            # dba = 20 * np.log10(np.sqrt(np.mean(w**2))) - self.top
            dba = 10 * np.log10(np.mean(w ** 2)) - self.top
            v[1].append(int(dba))

        super(DBA, self).put(data)
        print('dBFS {}, dB(A) {}'.format(v[0], v[1]))


def main():
    import time
    from microphone.source import Source

    src = Source(rate=48000, channels=2, frames_size=4800)
    dba = DBA(rate=src.rate, channels=src.channels)

    src.pipeline(dba)

    src.pipeline_start()
    while True:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            break

    src.pipeline_stop()


if __name__ == '__main__':
    main()
