import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime
from ..db import Database
import httpx

notify_line_router = APIRouter()

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")

async def push_line_message(user_id: str, message: str) -> int:
    """
    傳送 LINE 訊息給指定的使用者。
    """
    url = "https://api.line.me/v2/bot/message/push"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"
    }
    payload = {
        "to": user_id,
        "messages": [
            {
                "type": "text",
                "text": message
            }
        ]
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=payload)
        print(f"傳送給 {user_id}，狀態碼：{response.status_code}")
        if response.status_code != 200:
            print(f"回應內容：{response.text}")
        return response.status_code

@notify_line_router.post("/notify_line")
async def notify_line():
    """
    發送跌倒事件通知給 LINE 使用者。
    """
    try:
        detected_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"""⚠️ 注意！偵測到跌倒事件 ⚠️
時間：{detected_time}
請立即檢查爺爺奶奶的狀況，確保他們安全無恙。"""

        async with Database.connection() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("SELECT line_id FROM users WHERE role_id = 2")
                user_ids = [row["line_id"] for row in await cursor.fetchall()]

        success_list, fail_list = [], []
        for uid in user_ids:
            status = await push_line_message(uid, message)
            (success_list if status == 200 else fail_list).append(uid)

        return JSONResponse(content={
            "status": "ok",
            "sent": success_list,
            "failed": fail_list,
            "message": message
        }, status_code=200)

    except Exception as e:
        print(f"[錯誤] 發送失敗：{str(e)}")
        raise HTTPException(status_code=500, detail=f"伺服器錯誤: {str(e)}")