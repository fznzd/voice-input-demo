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
# ============================================================

audio_queue = Queue()
full_text = ""
last_lines = []
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

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
                buffer = buffer[-int(SAMPLE_RATE * 0.5):]  # 保留最新0.5秒，避免断句丢失
                continue

            # 识别（开启beam_size，提升准确率）
            segments, _ = model.transcribe(
                audio_np,
                language="zh",
                beam_size=3,          # 适度beam_size，平衡速度与准确率
                best_of=3,
                temperature=0.0,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                initial_prompt="以下是普通话的句子，简体中文，正常对话，无重复内容。",
            )

            current_text = "".join(seg.text.strip() for seg in segments)

            # 过滤重复垃圾内容
            if is_repeat(current_text):
                last_lines.append(current_text)
                buffer = buffer[-int(SAMPLE_RATE * 0.5):]
                continue

            # 更新防重复列表
            last_lines.append(current_text)
            if len(last_lines) > MAX_REPEAT * 2:
                last_lines.pop(0)

            # 输出：实时覆盖 + 句子结束固化
            print(f"\r识别：{full_text} {current_text}", end="", flush=True)
            if current_text.endswith(("。", "！", "？", ".", "，")):
                full_text += current_text + " "
                buffer = b""  # 句子结束，清空缓冲
            else:
                # 只保留最后1秒音频，避免上下文断档
                buffer = buffer[-int(SAMPLE_RATE * 1):]

        time.sleep(0.1)

except KeyboardInterrupt:
    print(f"\n\n✅ 最终识别结果：\n{full_text}")