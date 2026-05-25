# 语音输入法

## 项目简介

本项目基于 `funasr_stream_v3.py` 实现一个按住说话的本地语音输入法脚本。
脚本从麦克风采集音频，按住 `Ctrl` 键录音，松开后调用 `FunASR` 模型识别中文语音，并把识别结果自动粘贴到当前光标所在的应用中。

## 实现功能

- **实时麦克风录音**：按住 `Ctrl` 键开始录音，松开 `Ctrl` 停止录音。
- **本地语音识别**：使用 `funasr` 的 `AutoModel` 加载中文 ASR 模型进行识别。
- **文本上屏**：识别结果通过剪贴板和 `pyautogui` 自动粘贴到当前焦点窗口。
- **按住说话模式**：避免录音时累积过长音频，用户可短按/长按 `Ctrl` 控制录音时长。
- **自动依赖检查**：程序启动时检查必要库并给出安装提示。

## 依赖库

以下 Python 包是运行 `funasr_stream_v3.py` 所必须的：

- `numpy`
- `pyaudio`
- `pyautogui`
- `pyperclip`
- `keyboard`
- `funasr`
- `modelscope`

## 推荐安装方式

请使用本项目对应 Python 解释器或虚拟环境执行安装：

```powershell

python -m pip install numpy pyaudio pyautogui pyperclip keyboard funasr modelscope


```

## 运行方式
```
python funasr_stream_v3.py
```

运行后，程序会：

1. 检查依赖库
2. 加载 `FunASR` 模型
3. 打开麦克风采集
4. 按住 `Ctrl` 开始录音，松开 `Ctrl` 识别并插入文字

## 注意事项

- 模型首次加载会下载模型文件，下载量较大，可能需要等待几分钟。
- 程序默认使用 `cpu` 设备；如果有 GPU，可将 `funasr_stream_v3.py` 中 `device="cpu"` 修改为 `device="cuda"`。
- 如果出现 `pyautogui` 或 `keyboard` 等缺少依赖的错误，请根据提示安装对应包。
- 当前脚本仅支持 Windows 环境，因为 `keyboard` 和 `pyautogui` 的全局按键/粘贴功能在 Windows 上更稳定。

## 目录说明
菜鸟已经
- `funasr_stream_v3.py`：主程序，负责录音、识别和文本插入。
- `model/`：本地模型缓存目录。
- `modelscope_cache/`：`modelscope` 下载缓存目录。


## demo视频链接（与上传视频相同）
【demo视频】 https://www.bilibili.com/video/BV14WGo6TEbp/?share_source=copy_web&vd_source=1b465bda1d45911e62dd7b40074b3c49
