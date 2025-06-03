from flask import Blueprint, jsonify, send_file
from pathlib import Path
from openai import OpenAI
from datetime import datetime
import os
from ..utils import generate_daily_summary

news_voice_bp = Blueprint('news_voice', __name__)

VOICE_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / 'static/news_audio'
VOICE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

@news_voice_bp.route('/generate_news_audio', methods=['POST'])
def generate_news_audio():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    try:
        today_str = datetime.today().strftime("%Y%m%d")
        speech_file_path = VOICE_OUTPUT_DIR / f"news_{today_str}.mp3"

        if speech_file_path.exists():
            speech_file_path.unlink()

        summary = generate_daily_summary()

        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="nova",
            input=summary,
            instructions="請用溫暖親切的語氣唸出今日新聞摘要。",
            speed=0.8
        ) as response:
            response.stream_to_file(speech_file_path)

        return jsonify({
            "status": "success",
            "message": "語音生成完成",
            "filename": speech_file_path.name,
            "summary": summary
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@news_voice_bp.route('/play_today_news_audio', methods=['GET'])
def play_today_news_audio():
    today_str = datetime.today().strftime("%Y%m%d")
    speech_file_path = VOICE_OUTPUT_DIR / f"news_{today_str}.mp3"

    if not speech_file_path.exists():
        return jsonify({"status": "error", "message": "尚未產生今日語音摘要"}), 404

    return send_file(speech_file_path, mimetype='audio/mpeg')
