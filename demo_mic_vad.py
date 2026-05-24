"""麦克风实时采集 + VAD 演示。

用法:
  python demo_mic_vad.py                  # 录 10 秒，终端显示 speech/silence
  python demo_mic_vad.py --duration 5
  python demo_mic_vad.py --save out.wav    # 仅保存 VAD 判定为语音的片段
  python demo_mic_vad.py --list-devices
"""

from __future__ import annotations

import argparse
import sys
import time
import wave
from pathlib import Path

import numpy as np

from client.audio.capture import AudioCapture
from client.audio.vad import create_vad


def pcm_rms(pcm: bytes) -> float:
    samples = np.frombuffer(pcm, dtype=np.int16)
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


def save_pcm_wav(path: Path, pcm_chunks: list[bytes], sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for chunk in pcm_chunks:
            wf.writeframes(chunk)


def main() -> int:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="麦克风采集 + VAD 演示")
    parser.add_argument("--duration", type=float, default=10.0, help="录音时长（秒）")
    parser.add_argument("--device", type=int, default=None, help="麦克风设备 index")
    parser.add_argument("--chunk-ms", type=int, default=200, help="每帧毫秒数")
    parser.add_argument(
        "--vad",
        choices=["energy", "webrtc"],
        default="energy",
        help="VAD 算法（默认 energy，无需编译；webrtc 需安装 webrtcvad）",
    )
    parser.add_argument(
        "--vad-aggressiveness",
        type=int,
        default=2,
        choices=[0, 1, 2, 3],
        help="webrtc VAD 灵敏度 0~3",
    )
    parser.add_argument(
        "--rms-threshold",
        type=float,
        default=400.0,
        help="energy VAD 的 RMS 阈值",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="保存 VAD 过滤后的语音片段为 wav",
    )
    parser.add_argument("--list-devices", action="store_true", help="列出输入设备")
    args = parser.parse_args()

    if args.list_devices:
        print("可用输入设备:")
        for dev in AudioCapture.list_devices():
            print(
                f"  [{dev.index}] {dev.name} "
                f"(channels={dev.max_input_channels}, rate={dev.default_samplerate:.0f})"
            )
        return 0

    capture = AudioCapture(chunk_ms=args.chunk_ms, device=args.device)
    vad_kwargs = {"aggressiveness": args.vad_aggressiveness}
    if args.vad == "energy":
        vad_kwargs = {"rms_threshold": args.rms_threshold}
    try:
        vad = create_vad(args.vad, **vad_kwargs)
    except ModuleNotFoundError:
        print("webrtcvad 未安装，已回退到 energy VAD", file=sys.stderr)
        vad = create_vad("energy", rms_threshold=args.rms_threshold)

    print(f"VAD 模式: {args.vad}")

    stats = {
        "total_chunks": 0,
        "speech_chunks": 0,
        "silence_chunks": 0,
        "upload_bytes": 0,
        "total_bytes": 0,
    }
    speech_buffers: list[bytes] = []
    stop_at = time.monotonic() + args.duration

    def on_frame(pcm: bytes) -> None:
        if time.monotonic() >= stop_at:
            return

        stats["total_chunks"] += 1
        stats["total_bytes"] += len(pcm)

        ratio = vad.speech_ratio(pcm)
        is_speech = ratio >= vad.speech_ratio_threshold
        rms = pcm_rms(pcm)
        bar = "#" * min(30, int(rms / 500))

        if is_speech:
            stats["speech_chunks"] += 1
            stats["upload_bytes"] += len(pcm)
            speech_buffers.append(pcm)
            tag = "SPEECH"
        else:
            stats["silence_chunks"] += 1
            tag = "silence"

        print(
            f"[{tag:7}] rms={rms:6.0f} vad={ratio:.0%} |{bar:<30}|",
            flush=True,
        )

    print(f"开始录音 {args.duration}s，请对着麦克风说话…（Ctrl+C 可提前结束）")
    print("-" * 60)

    capture.start(on_frame)
    try:
        while time.monotonic() < stop_at:
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        capture.stop()

    print("-" * 60)
    saved_pct = (
        100.0 * stats["upload_bytes"] / stats["total_bytes"]
        if stats["total_bytes"]
        else 0.0
    )
    print("统计:")
    print(f"  总帧数:     {stats['total_chunks']}")
    print(f"  语音帧:     {stats['speech_chunks']}")
    print(f"  静音帧:     {stats['silence_chunks']}")
    print(f"  可上传数据: {stats['upload_bytes'] / 1024:.1f} KB / {stats['total_bytes'] / 1024:.1f} KB ({saved_pct:.0f}%)")

    if args.save and speech_buffers:
        save_pcm_wav(args.save, speech_buffers, capture.sample_rate)
        print(f"  已保存语音片段: {args.save}")
    elif args.save:
        print("  未检测到语音，未生成文件")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
