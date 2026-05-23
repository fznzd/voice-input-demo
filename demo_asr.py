"""语音识别 Demo：读取音频文件并输出文字。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

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


def resolve_model_path(model_size: str) -> str:
    """优先使用项目 model/ 目录下的本地模型。"""
    local = MODEL_DIR / f"faster-whisper-{model_size}"
    if (local / "model.bin").exists():
        return str(local)
    return model_size


def load_model(model_size: str) -> WhisperModel:
    model_path = resolve_model_path(model_size)
    if Path(model_path).is_dir():
        print(f"加载本地模型: {model_path}")
    else:
        print(f"加载模型: {model_size}（首次运行会自动下载到 HuggingFace 缓存）")
    try:
        return WhisperModel(model_path, device="cpu", compute_type="int8")
    except Exception as exc:
        print(
            "\n模型加载失败，常见原因：\n"
            "  1. 未联网或 Hugging Face 访问超时\n"
            "  2. 本地 model/ 目录尚无模型文件\n\n"
            "建议：\n"
            "  - 下载到项目目录：    python download_model.py --model base\n"
            f"  - 或设置镜像后重试：  $env:HF_ENDPOINT = \"{HF_MIRROR}\"\n",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


def transcribe(
    audio_path: Path,
    model_size: str = DEFAULT_MODEL,
    *,
    simplified: bool = True,
) -> str:
    model = load_model(model_size)

    print(f"识别音频: {audio_path}")
    kwargs: dict = {
        "language": "zh",
        "beam_size": 5,
        "vad_filter": True,
    }
    if simplified:
        kwargs["initial_prompt"] = SIMPLIFIED_PROMPT

    segments, info = model.transcribe(str(audio_path), **kwargs)

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
