from flask import Flask, jsonify, Response, Blueprint, send_file
from ..db import get_connection
import os
from openai import OpenAI
from pathlib import Path
from datetime import datetime

reels_bp = Blueprint("reels_bp", __name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

AUDIO_DIR = Path(__file__).resolve().parent.parent.parent / "static/reels_audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

@reels_bp.route('/speak_latest_post', methods=['GET'])
def speak_latest_post():
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT content FROM family_posts
            WHERE account_name = 'Eric'
            ORDER BY post_time DESC
            LIMIT 1
        """)
        row = cursor.fetchone()
        content = row[0] if row else "最近沒有動態"
        
        if content != "最近沒有動態":
            gpt_response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "請將這段社群貼文改寫成適合語音播放的親切說話語氣，不需太正式："},
                    {"role": "user", "content": content}
                ]
            )
            spoken_text = gpt_response.choices[0].message.content
        else:
            spoken_text = content

        tts_response = client.audio.speech.create(
            model="tts-1",
            voice="nova",  
            input=spoken_text
        )

        audio_data = tts_response.read()
        
        today_str = datetime.today().strftime("%Y%m%d")
        audio_file_path = AUDIO_DIR / f"latest_post_{today_str}.mp3"
        with open(audio_file_path, "wb") as f:
            f.write(audio_data)

        return jsonify({
            "status": "success",
            "filename": audio_file_path.name,
            "url": f"/static/reels_audio/{audio_file_path.name}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()