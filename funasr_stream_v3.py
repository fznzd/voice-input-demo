# ====== 强制把模型下到当前目录 & 禁用进度条 ======
import os
os.environ["MODELSCOPE_CACHE"] = os.path.join(os.getcwd(), "modelscope_cache")
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
# =================================================

import threading
import time
from datetime import datetime
from queue import Queue

try:
    import numpy as np
    import pyaudio
    import pyautogui
    import pyperclip
    import keyboard
    from funasr import AutoModel
except ImportError as e:
    pkg = getattr(e, 'name', None)
    if not pkg:
        msg = str(e)
        pkg = msg.split()[-1].strip("'\".")
    print(f"❌ 缺少依赖库: {pkg}")
    print("请运行: pip install keyboard pyaudio pyautogui pyperclip numpy funasr modelscope")
    exit(1)

# ========== 文本插入类 ==========
class TextInserter:
    """模拟键盘输入，写入当前光标位置"""
    
    def __init__(self, mode="clipboard"):
        self.mode = mode
        self.last_inserted = ""
        
    def insert(self, text):
        """插入文本到当前焦点窗口"""
        if not text or text == self.last_inserted:
            return
            
        try:
            if self.mode == "clipboard":
                pyperclip.copy(text)
                time.sleep(0.05)  # 等待剪贴板更新
                pyautogui.hotkey('ctrl', 'v')
            elif self.mode == "typewrite":
                pyautogui.typewrite(text, interval=0.005)
            self.last_inserted = text
        except Exception as e:
            print(f"插入失败: {e}")

# ========== 语音录制类 ==========
class VoiceRecorder:
    """按住说话模式的录音管理器"""
    
    def __init__(self):
        self.is_recording = False
        self.audio_data = []
        
    def start_recording(self):
        """开始录音"""
        self.audio_data = []
        self.is_recording = True
        
    def stop_recording(self):
        """停止录音"""
        self.is_recording = False
        if len(self.audio_data) == 0:
            return None
            
        # 合并所有音频数据
        audio_array = np.concatenate(self.audio_data, axis=0)
        return audio_array.astype(np.float32) / 32768.0
        
    def add_audio_chunk(self, data):
        """添加音频块"""
        if self.is_recording:
            self.audio_data.append(data)

# ========== 原有配置 ==========
CHUNK = 1024
CHANNELS = 1
RATE = 16000
FORMAT = pyaudio.paInt16

# 初始化模型
print("🔄 正在加载模型，首次运行需要下载（约1.5GB），请耐心等待...")
try:
    model = AutoModel(
        model="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        device="cpu",  # 如果有GPU可改为 "cuda"
        disable_update=True,
        log_level="error",
    )
    print("✅ 模型加载成功！")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    exit(1)

# 初始化文本插入器和录音器
inserter = TextInserter(mode="clipboard")
recorder = VoiceRecorder()

# ========== 全局变量 ==========
recording_active = False
audio_queue = Queue()

def audio_capture_and_process():
    """音频采集和处理的主线程"""
    global recording_active

    pa = pyaudio.PyAudio()
    stream = pa.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                     input=True, frames_per_buffer=CHUNK)
    
    print("✅ 音频系统已启动")
    print("=" * 50)
    print("🎮 使用说明：")
    print("   1. 按住 Ctrl 键开始录音")
    print("   2. 说话（说完后松开 Ctrl）")
    print("   3. 自动识别并输入文字")
    print("   4. 按 Ctrl+C 退出程序")
    print("=" * 50)
    print("\n🎤 按住 Ctrl 键开始说话...")
    
    try:
        while True:
            # 读取音频数据
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            
            # 检查快捷键状态（这里改用 Ctrl 键，兼容性更好）
            if keyboard.is_pressed('ctrl'):
                if not recording_active:
                    # 开始录音
                    recording_active = True
                    recorder.start_recording()
                    print("\r🎙️  录音中...（松开 Ctrl 结束）", end="", flush=True)
                
                # 添加音频数据
                recorder.add_audio_chunk(audio_data)


            else:
                if recording_active:
                    # 停止录音并进行识别
                    recording_active = False
                    print("\r🎤 识别中...   ", end="", flush=True)
                    
                    # 获取录音数据
                    audio_float = recorder.stop_recording()
                    
                    if audio_float is not None and len(audio_float) > RATE * 0.3:  # 至少0.3秒
                        try:
                            # 调用模型识别
                            res = model.generate(input=audio_float, language="zh")
                            if res and res[0]["text"]:
                                recognized_text = res[0]["text"].strip()
                                
                                if recognized_text:
                                    print(f"\r✅ {recognized_text}")
                                    # 插入文本
                                    inserter.insert(recognized_text)
                                else:
                                    print("\r⚠️  未识别到语音    ")
                            else:
                                print("\r❌ 识别失败    ")
                        except Exception as e:
                            print(f"\r❌ 识别错误: {e}")
                    else:
                        print("\r⚠️  录音时间太短    ")
                    
                    print("\n🎤 按住 Ctrl 键继续说话...")
                    
    except KeyboardInterrupt:
        print("\n\n👋 程序已退出")
    except Exception as e:
        print(f"\n❌ 运行错误: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

# 启动程序
if __name__ == "__main__":
    # 检查依赖
    try:
        import keyboard
        import pyaudio
        import pyautogui
        import pyperclip
    except ImportError as e:
        print(f"❌ 缺少依赖库: {e}")
        print("请运行: pip install keyboard pyaudio pyautogui pyperclip numpy funasr modelscope")
        exit(1)
    
    # 运行主函数
    audio_capture_and_process()