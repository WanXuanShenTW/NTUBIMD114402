from melo.api import TTS

tts = TTS(language='ZH', device='cuda')  # 沒 GPU 就用 cpu

tts.tts_to_file(
    text="你好，我是 MeloTTS，今天的天氣真好！",
    speaker_id="default",
    output_path="demo.wav"
)
