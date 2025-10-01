from pathlib import Path
from openai import OpenAI
import chatgpt
import os

client = OpenAI(api_key="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")

# 設定語音檔輸出路徑
speech_file_path = Path(__file__).parent / "speech.mp3"

# 如果檔案已存在就刪除（或直接覆寫也可，不刪也沒問題）
if speech_file_path.exists():
    print(f"🗑️ 檔案已存在，將覆寫：{speech_file_path}")
    speech_file_path.unlink()  # 刪除舊的 speech.mp3

# 呼叫 OpenAI TTS 並輸出
with client.audio.speech.with_streaming_response.create(
    model="gpt-4o-mini-tts",
    voice="coral",
    input=chatgpt.summary,
    instructions="Speak in a cheerful and positive tone.",
) as response:
    response.stream_to_file(speech_file_path)

print(f"✅ 完成語音合成：{speech_file_path}")
