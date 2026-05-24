"""语音识别 Demo：读取音频文件并输出文字。"""

from __future__ import annotations

import argparse
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np
from client.audio.capture import AudioCapture
from faster_whisper import WhisperModel

DEFAULT_SAMPLE = Path(__file__).parent / "samples" / "record_out (4).wav"
MODEL_DIR = Path(__file__).parent / "model"
DEFAULT_MODEL = "base"
HF_MIRROR = "https://hf-mirror.com"
# 引导 Whisper 以简体字形输出（Whisper 不保证 100% 简体）
SIMPLIFIED_PROMPT = "以下是普通话的句子。"


def setup_hf_mirror() -> None:
    """国内网络默认走 Hugging Face 镜像，避免下载超时。"""
    os.environ.setdefault("HF_ENDPOINT", HF_MIRROR)


def ensure_sample(target: Path) -> None:
    """用 Windows 内置 TTS 生成中文测试音频（无需联网）。"""
    if target.exists():
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"正在生成本地测试音频 -> {target}")

    import pyttsx3

    engine = pyttsx3.init()
    engine.save_to_file("你好，这是语音输入法测试。", str(target))
    engine.runAndWait()


def save_audio_wav(path: Path, audio: np.ndarray, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio_int16 = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def capture_audio(
    duration: float,
    device: int | None = None,
    chunk_ms: int = 200,
    sample_rate: int = 16000,
) -> np.ndarray:
    capture = AudioCapture(
        sample_rate=sample_rate,
        channels=1,
        chunk_ms=chunk_ms,
        device=device,
    )
    pcm_chunks: list[bytes] = []
    stop_at = time.monotonic() + duration

    def on_frame(pcm: bytes) -> None:
        if time.monotonic() >= stop_at:
            return
        pcm_chunks.append(pcm)

    print(f"开始从麦克风录制 {duration} 秒语音，请对着麦克风说话…（Ctrl+C 可提前结束）")
    capture.start(on_frame)
    try:
        while time.monotonic() < stop_at:
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        capture.stop()

    if not pcm_chunks:
        return np.zeros(0, dtype=np.float32)

    audio = np.frombuffer(b"".join(pcm_chunks), dtype=np.int16).astype(np.float32) / 32768.0
    return audio


def resolve_model_path(model_size: str) -> str:
    """优先使用项目 model/ 目录下的本地模型。"""
    local = MODEL_DIR / f"faster-whisper-{model_size}"
    # 仅使用本地模型目录，不要自动切换到远程仓库或下载模型。
    if (local / "model.bin").exists():
        return str(local)
    return ""


def load_model(model_size: str) -> WhisperModel:
    model_path = resolve_model_path(model_size)
    if model_path and Path(model_path).is_dir():
        print(f"加载本地模型: {model_path}")
        try:
            return WhisperModel(model_path, device="cpu", compute_type="int8")
        except Exception as exc:
            print(
                "\n本地模型加载失败，可能是模型文件不完整或路径错误。",
                file=sys.stderr,
            )
            raise SystemExit(1) from exc

    # 如果本地模型不存在，尝试自动下载到项目 model/ 目录（使用 download_model.py 中的下载逻辑）。
    # 这样在无本地模型时会自动拉取，而不是直接报错退出，改善一次性体验。
    print(
        f"未找到本地模型: model/faster-whisper-{model_size}，开始自动下载...",
        file=sys.stderr,
    )

    try:
        # 延迟导入以避免在 import 时就触发网络操作
        from download_model import download as download_model

        target = download_model(model_size)
        print(f"下载完成，载入模型: {target}")
        return WhisperModel(str(target), device="cpu", compute_type="int8")
    except Exception as exc:
        print(
            "\n自动下载模型失败：",
            exc,
            file=sys.stderr,
        )
        print(
            f"请手动运行：python download_model.py --model {model_size} 或检查网络/镜像设置。",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


def transcribe(
    audio: Path | np.ndarray,
    model: WhisperModel | None = None,
    model_size: str = DEFAULT_MODEL,
    *,
    simplified: bool = True,
) -> str:
    if model is None:
        model = load_model(model_size)

    if isinstance(audio, np.ndarray):
        print("识别音频: <mic input> (numpy array)")
        audio_arg = audio
    else:
        print(f"识别音频: {audio}")
        audio_arg = str(audio)

    kwargs: dict = {
        "language": "zh",
        "beam_size": 5,
        "vad_filter": True,
    }
    if simplified:
        kwargs["initial_prompt"] = SIMPLIFIED_PROMPT

    segments, info = model.transcribe(audio_arg, **kwargs)

    print(f"检测语言: {info.language} (概率 {info.language_probability:.2f})")
    if simplified:
        print("输出格式: 简体中文")
    print("-" * 40)

    lines: list[str] = []
    for seg in segments:
        line = seg.text.strip()
        if line:
            print(f"[{seg.start:6.1f}s -> {seg.end:6.1f}s] {line}")
            lines.append(line)

    result = "".join(lines)
    print("-" * 40)
    print("完整结果:", result or "(无识别内容)")
    return result


def main() -> int:
    setup_hf_mirror()

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="语音识别 Demo")
    parser.add_argument(
        "audio",
        nargs="?",
        help="音频文件路径（wav/mp3 等），不传则使用示例音频",
    )
    parser.add_argument(
        "--mic",
        action="store_true",
        help="从麦克风直接录制并识别",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="麦克风录音时长（秒），仅在 --mic 时有效",
    )
    parser.add_argument(
        "--chunk-ms",
        type=int,
        default=200,
        help="麦克风采样帧长度（毫秒），仅在 --mic 时有效",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="麦克风设备 index，默认使用系统默认输入设备",
    )
    parser.add_argument(
        "--save-wav",
        type=Path,
        default=None,
        help="保存麦克风录音为 wav 文件，配合 --mic 使用",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Whisper 模型：tiny/base/small/medium（默认 {DEFAULT_MODEL}，优先读 model/ 目录）",
    )
    parser.add_argument(
        "--no-simplified",
        action="store_true",
        help="不使用简体引导 prompt，保留 Whisper 原始输出",
    )
    args = parser.parse_args()

    if args.mic:
        try:
            model = load_model(args.model)
        except SystemExit:
            return 1

        audio = capture_audio(
            duration=args.duration,
            device=args.device,
            chunk_ms=args.chunk_ms,
        )
        if audio.size == 0:
            print("未录制到音频数据。", file=sys.stderr)
            return 1
        if args.save_wav:
            save_audio_wav(args.save_wav, audio)
            print(f"已保存录音到: {args.save_wav}")
        transcribe(audio, model=model, simplified=not args.no_simplified)
        return 0

    if args.audio:
        audio_path = Path(args.audio)
        if not audio_path.exists():
            print(f"文件不存在: {audio_path}", file=sys.stderr)
            return 1
    else:
        audio_path = DEFAULT_SAMPLE
        ensure_sample(audio_path)

    transcribe(audio_path, args.model, simplified=not args.no_simplified)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
