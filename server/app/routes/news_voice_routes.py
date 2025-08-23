from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pathlib import Path
from openai import OpenAI
from datetime import datetime
import os
from ..utils.news_utils import generate_daily_summary

news_voice_router = APIRouter()

VOICE_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / 'static/news_audio'
VOICE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

@news_voice_router.post("/generate_news_audio")
async def generate_news_audio():
    """
    生成今日新聞語音摘要。
    """
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

        return JSONResponse(content={
            "status": "success",
            "message": "語音生成完成",
            "filename": speech_file_path.name,
            "summary": summary
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"伺服器錯誤: {str(e)}")

@news_voice_router.get("/play_today_news_audio")
async def play_today_news_audio():
    """
    播放今日新聞語音摘要。
    """
    today_str = datetime.today().strftime("%Y%m%d")
    speech_file_path = VOICE_OUTPUT_DIR / f"news_{today_str}.mp3"

    if not speech_file_path.exists():
        raise HTTPException(status_code=404, detail="尚未產生今日語音摘要")

    return FileResponse(
        path=speech_file_path,
        media_type="audio/mpeg",
        filename=speech_file_path.name
    )