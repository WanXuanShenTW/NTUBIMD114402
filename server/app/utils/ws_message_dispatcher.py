import json
import asyncio

# -----------------------------
# WebSocket 訊息處理分派器
# -----------------------------

async def handle_ws_text(user_id: str, text: str):
    data = json.loads(text)
    # ✅ 延遲 import，避免在啟動階段載入模型
    from .stream_infer_manager import stream_infer_manager
    manager = stream_infer_manager
    await manager.ingest(user_id, data)

def notify_user_disconnected(user_id: str):
    # ✅ 延遲 import
    from .stream_infer_manager import stream_infer_manager
    if stream_infer_manager is not None:
        loop = asyncio.get_event_loop()
        loop.create_task(stream_infer_manager.force_recover(user_id, reason="disconnect"))
