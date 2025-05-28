from flask import Flask, jsonify, Response, Blueprint
from ..db import get_connection
import os
from openai import OpenAI

reels_bp = Blueprint("reels_bp", __name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@reels_bp.route('/speak_latest_post', methods=['GET'])
def speak_latest_post():
    try:
        # 1. 查詢資料庫
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

        return Response(audio_data, mimetype="audio/mpeg")

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

