# ====== 强制把模型下到当前目录 & 禁用进度条 ======
import os
os.environ["MODELSCOPE_CACHE"] = os.path.join(os.getcwd(), "modelscope_cache")
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
# =================================================

import numpy as np
import pyaudio
import threading
from queue import Queue
from funasr import AutoModel

CHUNK = 1024
CHANNELS = 1
RATE = 16000
FORMAT = pyaudio.paInt16

audio_queue = Queue()
full_text = ""        # 累积的完整句子
last_text = ""        # 上一次识别出的文本，用于计算增量

model = AutoModel(
    model="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    device="cpu",
    disable_update=True,
    log_level="error",
)

def audio_thread():
    pa = pyaudio.PyAudio()
    stream = pa.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                     input=True, frames_per_buffer=CHUNK)
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_queue.put(np.frombuffer(data, dtype=np.int16))

threading.Thread(target=audio_thread, daemon=True).start()

buffer = np.array([], dtype=np.int16)
print("✅ FunASR 已启动（长窗口2秒），说话试试...\n")
print("当前识别结果：", end="", flush=True)

try:
    while True:
        while not audio_queue.empty():
            buffer = np.concatenate([buffer, audio_queue.get()])

        # 窗口长度：2秒（原0.6秒太短）
        if len(buffer) >= RATE * 2:   # 2秒
            audio = buffer.astype(np.float32) / 32768.0
            res = model.generate(input=audio, language="zh")
            if res and res[0]["text"]:
                current_text = res[0]["text"].strip()

                # 计算增量：如果新文本包含了旧文本，则只取新增加的部分
                if current_text.startswith(last_text):
                    new_part = current_text[len(last_text):]
                else:
                    # 无法计算增量时，直接追加（用空格分隔）
                    new_part = " " + current_text if full_text else current_text

                if new_part:
                    full_text += new_part
                    last_text = current_text
                    # 用 \r 回到行首，然后打印整个句子（覆盖上一行）
                    print(f"\r当前识别结果：{full_text}", end="", flush=True)

            # 重叠保留：保留最后 0.5 秒，避免断层
            buffer = buffer[-RATE // 2:]

except KeyboardInterrupt:
    print(f"\n\n✅ 最终识别结果：{full_text}")