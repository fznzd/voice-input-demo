"""麦克风 PCM 采集（sounddevice）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import sounddevice as sd


@dataclass(frozen=True)
class DeviceInfo:
    index: int
    name: str
    max_input_channels: int
    default_samplerate: float


class AudioCapture:
    """16kHz / mono / int16 PCM 流式采集。"""

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_ms: int = 200,
        device: int | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_ms = chunk_ms
        self.device = device
        self.blocksize = int(sample_rate * chunk_ms / 1000)
        self._stream: sd.InputStream | None = None
        self._on_frame: Callable[[bytes], None] | None = None

    @staticmethod
    def list_devices() -> list[DeviceInfo]:
        devices: list[DeviceInfo] = []
        for idx, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0:
                devices.append(
                    DeviceInfo(
                        index=idx,
                        name=dev["name"],
                        max_input_channels=int(dev["max_input_channels"]),
                        default_samplerate=float(dev["default_samplerate"]),
                    )
                )
        return devices

    def start(self, on_frame: Callable[[bytes], None]) -> None:
        if self._stream is not None:
            raise RuntimeError("AudioCapture 已在运行")

        self._on_frame = on_frame

        def callback(indata: np.ndarray, _frames: int, _time, status) -> None:
            if status:
                print(f"[audio] {status}")
            if self._on_frame is None:
                return
            pcm = indata.copy().astype(np.int16).tobytes()
            self._on_frame(pcm)

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            blocksize=self.blocksize,
            device=self.device,
            callback=callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None
        self._on_frame = None
