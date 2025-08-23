from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
from datetime import datetime
from ..db import Database
import os
from openai import OpenAI

reels_router = APIRouter()

AUDIO_DIR = Path(__file__).resolve().parent.parent.parent / "static/reels_audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

@reels_router.get("/speak_latest_post")
async def speak_latest_post():
    """
    生成最新社群貼文的語音檔案。
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    try:
        async with Database.connection() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("""
                    SELECT content FROM family_posts
                    WHERE account_name = 'Eric'
                    ORDER BY post_time DESC
                    LIMIT 1
                """)
                row = await cursor.fetchone()
                content = row[0] if row else "最近沒有動態"

        if content != "最近沒有動態":
            gpt_response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "請將這段社群貼文用繁體中文讀出來。"},
                    {"role": "user", "content": content}
                ]
            )
            spoken_text = gpt_response.choices[0].message.content
        else:
            spoken_text = content

        tts_response = client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=spoken_text,
        )

        audio_data = tts_response.read()

        today_str = datetime.today().strftime("%Y%m%d")
        audio_file_path = AUDIO_DIR / f"latest_post_{today_str}.mp3"
        with open(audio_file_path, "wb") as f:
            f.write(audio_data)

        return JSONResponse(content={
            "status": "success",
            "filename": audio_file_path.name,
            "url": f"/static/reels_audio/{audio_file_path.name}"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"伺服器錯誤: {str(e)}")