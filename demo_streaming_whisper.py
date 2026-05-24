"""本地模拟流式输出，因为faster-whisper本身不支持流式，所以只能通过不断读取音频、攒够一定长度后识别来模拟流式效果。
同时增加了简单的能量降噪和重复内容过滤，提升稳定性和实用性。"""
import numpy as np
import pyaudio
from faster_whisper import WhisperModel
import time
import threading
from queue import Queue

# ===================== 稳定识别配置 =====================
SAMPLE_RATE = 16000
CHUNK = 1024
BUFFER_SECONDS = 2       # 2秒缓冲，上下文更足，识别更准
MODEL_SIZE = "base"
# 防重复：同一文本连续出现超过这个次数就忽略
MAX_REPEAT = 3
# 静音阈值（过滤环境噪音）
NOISE_THRESHOLD = 0.005
# 要屏蔽的提示词（自动过滤，不会显示）
BANNED_PROMPTS = ["以下是普通话的句子", "以下是普通话"]
# ============================================================

audio_queue = Queue()
full_text = ""
last_lines = []
model = WhisperModel("./model/faster-whisper-base", device="cpu", compute_type="int8")

def record_thread():
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )
    while True:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_queue.put(data)
        except:
            break

# 启动录音线程
threading.Thread(target=record_thread, daemon=True).start()

def is_noise(audio_np: np.ndarray) -> bool:
    """简单的能量降噪：判断是否为噪音"""
    rms = np.sqrt(np.mean(audio_np ** 2))
    return rms < NOISE_THRESHOLD

def is_repeat(text: str) -> bool:
    """判断文本是否为重复垃圾内容"""
    if not text:
        return True
    # 同一文本重复出现多次，判定为垃圾
    if len(last_lines) >= MAX_REPEAT and all(line == text for line in last_lines[-MAX_REPEAT:]):
        return True
    # 纯数字/标点/单字，判定为垃圾
    if len(text) < 2 or text.isdigit():
        return True
    return False

def clean_text(text: str) -> str:
    """自动过滤掉提示词，不会显示在识别结果里"""
    for banned in BANNED_PROMPTS:
        text = text.replace(banned, "").strip()
    return text

print("✅ 稳定版实时识别已启动，开始说话...（Ctrl+C 停止）\n")
buffer = b""

try:
    while True:
        # 持续读取音频
        while not audio_queue.empty():
            buffer += audio_queue.get()

        # 攒够2秒再识别，保证上下文
        if len(buffer) >= int(SAMPLE_RATE * 2 * BUFFER_SECONDS):
            audio_np = np.frombuffer(buffer, dtype=np.int16).astype(np.float32) / 32768.0

            # 先过滤噪音
            if is_noise(audio_np):
                buffer = buffer[-int(SAMPLE_RATE * 0.5):]
                continue

            # 识别
            segments, _ = model.transcribe(
                audio_np,
                language="zh",
                beam_size=3,
                best_of=3,
                temperature=0.0,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                initial_prompt="以下是普通话的句子，简体中文，正常对话，无重复内容。",
            )

            current_text = "".join(seg.text.strip() for seg in segments)
            # ✅ 关键：自动过滤提示词
            current_text = clean_text(current_text)

            # 过滤无效内容
            if not current_text or is_repeat(current_text):
                last_lines.append(current_text)
                buffer = buffer[-int(SAMPLE_RATE * 0.5):]
                continue

            last_lines.append(current_text)
            if len(last_lines) > MAX_REPEAT * 2:
                last_lines.pop(0)

            # 实时输出
            print(f"\r识别：{full_text} {current_text}", end="", flush=True)
            if current_text.endswith(("。", "！", "？", ".", "，")):
                full_text += current_text + " "
                buffer = b""
            else:
                buffer = buffer[-int(SAMPLE_RATE * 1):]

        time.sleep(0.1)

except KeyboardInterrupt:
    print(f"\n\n✅ 最终识别结果：\n{full_text}")