import json
from typing import Optional
from .stream_infer_manager import stream_infer_manager

async def handle_ws_text(user_id: str, text: str):
    try:
        data = json.loads(text)
    except Exception:
        return
    await stream_infer_manager.ingest(user_id, data)

def notify_user_disconnected(user_id: str):
    stream_infer_manager.drop_user(user_id)