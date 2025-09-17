# app/utils/ws_message_dispatcher.py
import json
from typing import Optional
from .stream_infer_manager import stream_infer_manager

async def handle_ws_text(user_id: str, text: str):
    """
    解析文字訊息（JSON 字串），依 type 分派到推論管理器。
    接受格式：
      {"type":"pose","frame_id":123,"timestamp_ms":...,"image_size":{...},"persons":[...]}
      {"type":"object","frame_id":123,"timestamp_ms":...,"image_size":{...},"detections":[...]}
    其他 type 會忽略。
    """
    try:
        data = json.loads(text)
    except Exception:
        # 非 JSON 直接忽略或視需要記 log
        return
    await stream_infer_manager.ingest(user_id, data)

def notify_user_disconnected(user_id: str):
    stream_infer_manager.drop_user(user_id)
