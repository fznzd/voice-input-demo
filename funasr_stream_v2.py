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

# ========== 新增：文本插入需要的库 ==========
import pyautogui
import pyperclip

# ========== 新增：文本插入类 ==========
class TextInserter:
    """模拟键盘输入，写入当前光标位置"""
    
    def __init__(self, mode="clipboard"):
        self.mode = mode
        # 记录上一次插入的内容，防止重复插入
        self.last_inserted = ""
        
    def insert(self, text):
        """插入文本到当前焦点窗口"""
        if not text or text == self.last_inserted:
            return
            
        if self.mode == "clipboard":
            # 剪贴板方式（最快，支持中文）
            pyperclip.copy(text)
            pyautogui.hotkey('ctrl', 'v')
        elif self.mode == "typewrite":
            # 逐字模拟（慢但兼容）
            pyautogui.typewrite(text, interval=0.005)
            
        self.last_inserted = text

# 初始化文本插入器
inserter = TextInserter(mode="clipboard")

# ========== 原有配置 ==========
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
print("✅ FunASR 已启动（长窗口2秒），说话试试...（按Ctrl+C 停止）")
print("🎤 识别结果将自动输入到当前光标位置（请先点击输入框）\n")
print("当前识别结果：", end="", flush=True)

try:
    while True:
        while not audio_queue.empty():
            buffer = np.concatenate([buffer, audio_queue.get()])

        # 窗口长度：2秒
        if len(buffer) >= RATE * 2:
            audio = buffer.astype(np.float32) / 32768.0
            res = model.generate(input=audio, language="zh")
            if res and res[0]["text"]:
                current_text = res[0]["text"].strip()

                # 计算增量
                if current_text.startswith(last_text):
                    new_part = current_text[len(last_text):]
                else:
                    new_part = " " + current_text if full_text else current_text

                if new_part:
                    full_text += new_part
                    last_text = current_text
                    
                    # ========== 新增：插入文本到当前窗口 ==========
                    inserter.insert(new_part)
                    # ===========================================
                    
                    print(f"\r当前识别结果：{full_text}", end="", flush=True)

            # 重叠保留：保留最后 0.5 秒，避免断层
            buffer = buffer[-RATE // 2:]

except KeyboardInterrupt:
    print(f"\n\n✅ 最终识别结果：{full_text}")
    print("👋 程序已退出")