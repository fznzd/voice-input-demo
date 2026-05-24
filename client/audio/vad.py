"""端侧 VAD 静音检测。

默认使用 EnergyVAD（RMS 能量阈值，无需编译）。
若已安装 webrtcvad，可通过 method="webrtc" 切换。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class VoiceActivityDetector(ABC):
    speech_ratio_threshold: float = 0.1

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def is_speech(self, pcm_chunk: bytes) -> bool: ...

    @abstractmethod
    def speech_ratio(self, pcm_chunk: bytes) -> float: ...


class EnergyVAD(VoiceActivityDetector):
    """基于 RMS 能量的 VAD，适合 MVP / 无编译环境。"""

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        frame_ms: int = 20,
        rms_threshold: float = 400.0,
        speech_ratio_threshold: float = 0.1,
        adaptive: bool = True,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_bytes = int(sample_rate * frame_ms / 1000) * 2
        self.rms_threshold = rms_threshold
        self.speech_ratio_threshold = speech_ratio_threshold
        self.adaptive = adaptive
        self._noise_floor = rms_threshold * 0.5

    def reset(self) -> None:
        self._noise_floor = self.rms_threshold * 0.5

    def _frame_rms(self, frame: bytes) -> float:
        samples = np.frombuffer(frame, dtype=np.int16)
        if samples.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))

    def _effective_threshold(self) -> float:
        if not self.adaptive:
            return self.rms_threshold
        return max(self.rms_threshold, self._noise_floor * 2.5)

    def speech_ratio(self, pcm_chunk: bytes) -> float:
        if len(pcm_chunk) < self.frame_bytes:
            return 0.0

        speech_frames = 0
        total_frames = 0
        threshold = self._effective_threshold()

        for offset in range(0, len(pcm_chunk) - self.frame_bytes + 1, self.frame_bytes):
            frame = pcm_chunk[offset : offset + self.frame_bytes]
            if len(frame) != self.frame_bytes:
                break
            rms = self._frame_rms(frame)
            total_frames += 1
            if rms >= threshold:
                speech_frames += 1
            elif self.adaptive:
                self._noise_floor = 0.95 * self._noise_floor + 0.05 * rms

        if total_frames == 0:
            return 0.0
        return speech_frames / total_frames

    def is_speech(self, pcm_chunk: bytes) -> bool:
        return self.speech_ratio(pcm_chunk) >= self.speech_ratio_threshold


class WebRtcVAD(VoiceActivityDetector):
    """webrtcvad 实现（需 pip install webrtcvad 及 C++ 编译环境）。"""

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        frame_ms: int = 20,
        aggressiveness: int = 2,
        speech_ratio_threshold: float = 0.1,
    ) -> None:
        import webrtcvad

        if sample_rate not in (8000, 16000, 32000, 48000):
            raise ValueError(f"webrtcvad 不支持采样率: {sample_rate}")
        if frame_ms not in (10, 20, 30):
            raise ValueError(f"webrtcvad 帧长必须是 10/20/30 ms: {frame_ms}")

        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_bytes = int(sample_rate * frame_ms / 1000) * 2
        self.speech_ratio_threshold = speech_ratio_threshold
        self._vad = webrtcvad.Vad(aggressiveness)

    def reset(self) -> None:
        pass

    def speech_ratio(self, pcm_chunk: bytes) -> float:
        if len(pcm_chunk) < self.frame_bytes:
            return 0.0

        speech_frames = 0
        total_frames = 0

        for offset in range(0, len(pcm_chunk) - self.frame_bytes + 1, self.frame_bytes):
            frame = pcm_chunk[offset : offset + self.frame_bytes]
            if len(frame) != self.frame_bytes:
                break
            total_frames += 1
            if self._vad.is_speech(frame, self.sample_rate):
                speech_frames += 1

        if total_frames == 0:
            return 0.0
        return speech_frames / total_frames

    def is_speech(self, pcm_chunk: bytes) -> bool:
        return self.speech_ratio(pcm_chunk) >= self.speech_ratio_threshold


def create_vad(
    method: str = "energy",
    **kwargs,
) -> VoiceActivityDetector:
    if method == "webrtc":
        return WebRtcVAD(**kwargs)
    if method == "energy":
        return EnergyVAD(**kwargs)
    raise ValueError(f"未知 VAD 方法: {method}")
