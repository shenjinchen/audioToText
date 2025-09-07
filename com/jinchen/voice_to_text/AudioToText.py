import sys
from typing import Union, BinaryIO
import sounddevice as sd
import queue
import threading
import numpy as np
from faster_whisper import WhisperModel

from huggingface_hub import HfApi

class ModelParam:
    def __init__(self, model_path='tiny.en', device='cpu', compute_type='int8', ):
        self.model_path = model_path
        self.device = device
        self.compute_type = compute_type

class TranscribeOptions:
    def __init__(self, task='transcribe', language='en', beam_size=5):
        self.config = dict(
            task = task,
            language = language,
            beam_size = beam_size,
            vad_filter = True)

'''
定义 Whisper 模型以及翻译的参数
'''
class WhisperModelWrap:
    def __init__(self, modelParam:ModelParam, transcriberOptions:TranscribeOptions):
        # mirror
        api = HfApi(endpoint='https://hf-mirror.com')
        self.model = WhisperModel(modelParam.model_path, device=modelParam.device, compute_type=modelParam.compute_type)
        self.transcriberOptions = transcriberOptions

    def executeTranscribe(self, audio: Union[str, BinaryIO, np.ndarray]):
        return self.model.transcribe(audio, **self.transcriberOptions.config)

class AudioToTextService:
    def __init__(self):
        # 音频队列，用于存储捕获的音频数据
        self.__audio_queue = queue.Queue()
        # 结果队列，用于存储转录的文本结果
        self.__result_queue = queue.Queue()

    # frames: int,
    #                          time: CData, status: CallbackFlags
    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Warning: {status}")
        self.__audio_queue.put(indata.copy())

    def transcribe_audio(self, model: WhisperModelWrap):
        """转录工作线程，从队列获取音频并进行转录"""
        audio_buffer = []
        while True:
            # 获取音频数据
            audio_data = self.__audio_queue.get()
            if audio_data is None:
                break
            audio_buffer.append(audio_data)

            # 转录音频片段
            if len(audio_buffer) > 0:
                # 将列表中的多个音频数组合并为一个
                combined_audio = np.concatenate(audio_buffer, axis=0)
                # 如果 BlackHole 是立体声（2ch），可以取一个声道或转换为单声道
                if combined_audio.ndim > 1 and combined_audio.shape[1] == 2:
                    combined_audio_mono = combined_audio[:, 0]  # 转换为单声道
                else:
                    combined_audio_mono = combined_audio

                # 转换为float32并归一化
                combined_audio_mono = combined_audio_mono.astype(np.float32)
                if np.max(np.abs(audio_data)) > 0:
                    combined_audio_mono = combined_audio_mono / np.max(np.abs(combined_audio_mono))
                audio_numpy = combined_audio_mono
                segments, info = model.executeTranscribe(audio_numpy)
                # 收集转录结果
                text = ""
                for segment in segments:
                    text += segment.text
                if text:
                    self.__result_queue.put(text)
                audio_buffer = []
            self.__audio_queue.task_done()

    def buildTranscribeAudioThread(self, model: WhisperModelWrap):
        transcribe_thread = threading.Thread(
            target=self.transcribe_audio,
            args=[model],
            daemon=True
        )
        transcribe_thread.start()
        return transcribe_thread

    def buildGetAudioStreamThread(self, device_id, transcribe_thread: threading.Thread, samplerate=16000, channles=2,
                                  blocksize=16000):
        try:
            with sd.InputStream(
                    device=device_id,
                    samplerate=samplerate,
                    channels=channles,
                    callback=self.audio_callback,
                    blocksize=blocksize):
                # 实时打印转录结果
                while True:
                    try:
                        result = self.__result_queue.get()
                        if result is not None and len(result) > 0:
                            print(result)
                        self.__result_queue.task_done()
                    except queue.Empty:
                        continue
        except KeyboardInterrupt:
            print("\n用户中断，正在退出...")
        except Exception as e:
            print(f"发生错误: {e}", file=sys.stderr)
        finally:
            # 清理资源
            self.__audio_queue.put(None)
            transcribe_thread.join()
            print("完成")
