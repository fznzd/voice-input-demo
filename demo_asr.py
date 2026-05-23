"""语音识别 Demo：读取音频文件并输出文字。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from faster_whisper import WhisperModel

DEFAULT_SAMPLE = Path(__file__).parent / "samples" / "demo.wav"


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


def transcribe(audio_path: Path, model_size: str = "base") -> str:
    print(f"加载模型: {model_size}（首次运行会自动下载）")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    print(f"识别音频: {audio_path}")
    segments, info = model.transcribe(
        str(audio_path),
        language="zh",
        beam_size=5,
        vad_filter=True,
    )

    print(f"检测语言: {info.language} (概率 {info.language_probability:.2f})")
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
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="语音识别 Demo")
    parser.add_argument(
        "audio",
        nargs="?",
        help="音频文件路径（wav/mp3 等），不传则使用示例音频",
    )
    parser.add_argument(
        "--model",
        default="base",
        help="Whisper 模型大小：tiny/base/small/medium（默认 base）",
    )
    args = parser.parse_args()

    if args.audio:
        audio_path = Path(args.audio)
        if not audio_path.exists():
            print(f"文件不存在: {audio_path}", file=sys.stderr)
            return 1
    else:
        audio_path = DEFAULT_SAMPLE
        ensure_sample(audio_path)

    transcribe(audio_path, args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
