# 语音输入法

## 环境

- Python：`F:\software\python\python.exe`（3.12.9）
- 项目虚拟环境：`.venv`

## 快速运行（语音识别 Demo）

```powershell
cd "F:\实习简历相关\七牛云项目-语音输入法"
$env:Path = "F:\software\python;F:\software\python\Scripts;" + $env:Path
$env:HF_ENDPOINT = "https://hf-mirror.com"   # 国内下载模型镜像

.\.venv\Scripts\python.exe demo_asr.py
```

指定音频文件：

```powershell
.\.venv\Scripts\python.exe demo_asr.py your_audio.wav --model base
```

当前 Demo 使用 **faster-whisper**（本地离线模型），验证「音频 → 文字」链路。

## 从 Demo 到「输入法」还需要什么

| 阶段 | 能力 | 说明 |
|------|------|------|
| ✅ 当前 | 文件/离线识别 | 录完再转，延迟较高 |
| 下一步 | 麦克风实时采集 | `sounddevice` 录音 + VAD 静音检测 |
| 下一步 | 流式 ASR | 换云端 WebSocket（如 `fun-asr-realtime`），边说边出字 |
| 核心 | 文本插入 | 模拟键盘输入，写入 Word/浏览器/微信等 |
| 核心 | 全局快捷键 | 如 `Win+Shift+V` 按住说话 |
| 体验 | 悬浮 UI | 显示 partial/final 识别结果 |
| 增值 | 长语音转写 | 七牛 LASR 异步转写 |

**一句话：Demo 只完成了 ASR；输入法 = 实时采集 + 流式识别 + 全局上屏。**
