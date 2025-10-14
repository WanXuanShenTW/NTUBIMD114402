import os
import asyncio
from typing import Iterable, Dict, List, Tuple, Optional
import httpx
from datetime import datetime
from app.db import Database
from app.dao.notify_line_dao import (
    list_caregiver_line_uids_by_elder,
    list_line_uids_by_role,
)

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_PUSH_URL = "https://api.line.me/v2/bot/message/push"

# --- 低階推播 ---
async def push_line_message(to_uid: str, message: str) -> int:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
    }
    payload = {"to": to_uid, "messages": [{"type": "text", "text": message}]}
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(LINE_PUSH_URL, headers=headers, json=payload)
        if resp.status_code != 200:
            print(f"[LINE PUSH] to={to_uid} code={resp.status_code} body={resp.text}")
        return resp.status_code

async def push_text_bulk(recipients: Iterable[str], message: str) -> Dict[str, List[str]]:
    recips = list(recipients)
    codes: List[int] = await asyncio.gather(*[push_line_message(uid, message) for uid in recips])
    sent = [u for u, c in zip(recips, codes) if c == 200]
    failed = [u for u, c in zip(recips, codes) if c != 200]
    return {"sent": sent, "failed": failed}

# --- 文案產生 ---
def build_message(*, status: Optional[str], message: Optional[str], detected_at: Optional[str]) -> str:
    if message:
        return message

    status_norm = (status or "").strip().lower()
    ts = detected_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(status_norm)
    if status_norm in ("跌倒", "fall","true"):
        return f"⚠️ 注意！偵測到跌倒事件 ⚠️\n時間：{ts}\n請立即檢查長者狀況，確保安全。"
    if status_norm in ("離床", "bed_exit", "leave_bed"):
        return f"ℹ️ 離床提醒\n時間：{ts}\n請留意長者是否需要協助。"
    if status_norm in ("返回", "return_bed", "back"):
        return f"✅ 已返回床上\n時間：{ts}\n目前無異常。"

    return f"📣 事件：{status or '通知'}\n時間：{ts}"

# --- 高階服務：Route 只呼叫這個 ---
async def notify_from_payload(
    *,
    elder_id: Optional[int],
    status: Optional[str],
    message: Optional[str],
    detected_at: Optional[str],
) -> Dict[str, object]:
    """
    - 有 elder_id → 通知該長者的照護者（找不到則 fallback 到長者本人；這邏輯在 DAO 已處理）
    - 沒 elder_id → 廣播所有照護者（role_id=2）
    """
    msg = build_message(status=status, message=message, detected_at=detected_at)

    async with Database.connection() as conn:
        if elder_id is not None:
            uids, source = await list_caregiver_line_uids_by_elder(conn, elder_id)
        else:
            uids = await list_line_uids_by_role(conn, 2)
            source = "role_broadcast"

        if not uids:
            return {"status": "ok", "message": msg, "source": source, "sent": [], "failed": [], "note": "無收件者"}

        result = await push_text_bulk(uids, msg)
        return {"status": "ok", "message": msg, "source": source, **result}