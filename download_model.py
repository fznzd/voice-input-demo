"""下载 faster-whisper 模型到项目 model 目录。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HF_MIRROR = "https://hf-mirror.com"
MODEL_DIR = Path(__file__).parent / "model"

REPO_MAP = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
}


def download(name: str) -> Path:
    os.environ.setdefault("HF_ENDPOINT", HF_MIRROR)

    from huggingface_hub import snapshot_download

    repo_id = REPO_MAP[name]
    target = MODEL_DIR / f"faster-whisper-{name}"

    print(f"下载 {repo_id} -> {target}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target),
        local_dir_use_symlinks=False,
    )

    model_bin = target / "model.bin"
    if not model_bin.exists():
        raise FileNotFoundError(f"下载未完成，缺少: {model_bin}")

    size_mb = model_bin.stat().st_size / 1024 / 1024
    print(f"完成: {model_bin} ({size_mb:.1f} MB)")
    return target


def main() -> int:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="下载 Whisper 模型到 model/")
    parser.add_argument(
        "--model",
        default="base",
        choices=sorted(REPO_MAP),
        help="模型大小（默认 base）",
    )
    args = parser.parse_args()

    try:
        download(args.model)
    except Exception as exc:
        print(f"\n下载失败: {exc}", file=sys.stderr)
        print(f"可尝试: $env:HF_ENDPOINT = \"{HF_MIRROR}\"", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
