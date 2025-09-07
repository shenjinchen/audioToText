import threading

from com.jinchen.voice_to_text.AudioToText import ModelParam, AudioToTextService, TranscribeOptions,  WhisperModelWrap
import sounddevice as sd

def get_blackhole_device_id():
    devices = sd.query_devices()
    blackhole_id = None
    print("可用的音频设备:")
    for i, device in enumerate(devices):
        print(f"设备索引 {i}: {device['name']} (最大输入通道数: {device['max_input_channels']})")
        # 检查设备名称是否包含'blackhole'（不区分大小写）
        if 'blackhole' in device['name'].lower():
            blackhole_id = i
            print(f"--> 找到BlackHole设备，索引为: {i}")

    if blackhole_id is None:
        print("未找到BlackHole设备，请确保已正确安装。")
    else:
        print(f"\nBlackHole的设备ID是: {blackhole_id}")
    return blackhole_id
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    modelParam: ModelParam = ModelParam('tiny.en', 'cpu', 'int8')
    transcribeOptions: TranscribeOptions = TranscribeOptions('transcribe', 'en', 1)
    model: WhisperModelWrap = WhisperModelWrap(modelParam, transcribeOptions)
    audioToTextService: AudioToTextService = AudioToTextService()
    transcribeAudioThread: threading.Thread = audioToTextService.buildTranscribeAudioThread(model)

    blackhole_id = get_blackhole_device_id()
    audioToTextService.buildGetAudioStreamThread(blackhole_id, transcribeAudioThread)

