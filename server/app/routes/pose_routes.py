from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketDisconnect

from ..utils.ws_connection_manager import ws_manager
from ..utils.stream_infer_manager import stream_infer_manager
from ..utils.ws_message_dispatcher import handle_ws_text, notify_user_disconnected

from ..service.fall_event_service import add_fall_event
pose_router = APIRouter(tags=["姿態偵測與跌倒事件"])

# ★ 與新版 manager 對齊：on_fall_start 多一個 start_frame 參數
async def on_fall_start(user_id: str, start_time: str, start_frame: int, result: dict):
    print("[FALL_START]", user_id, start_time, start_frame, result["probs"][result["pred_idx"]])
    # 若你希望一開始就入庫，保留原本功能：
    await add_fall_event(
        user_id=user_id,
        detected_time=start_time,
        location="客廳",
        pose_before_fall="正常行走"
    )

async def on_fall_recover(user_id: str, start_time: str, end_time: str,
                          start_frame: int, end_frame: int, peak_score: float, result: dict):
    print("[FALL_RECOVER]", user_id, start_time, end_time, peak_score)
    # TODO: 這裡保留你原版本的處理方式（若原本有寫 DB，就照舊呼叫）
    return

stream_infer_manager.set_handlers(
    on_fall_start=on_fall_start,
    on_fall_recover=on_fall_recover
)
print("[HOOKS] registered:", list(stream_infer_manager._handlers.keys()))

@pose_router.websocket("/ws/pose")
async def ws_pose(websocket: WebSocket):
    user_id = websocket.query_params.get("user_id")
    if not user_id or not user_id.strip():
        await websocket.close(code=1008, reason="Missing required query param: user_id")
        return
    await ws_manager.connect(user_id, websocket)
    try:
        while True:
            text = await websocket.receive_text()
            await handle_ws_text(user_id, text)
    except WebSocketDisconnect:
        ws_manager.disconnect(user_id, websocket)
        await stream_infer_manager.force_recover(user_id, reason="disconnect")
        if not ws_manager.get_user_connections(user_id):
            notify_user_disconnected(user_id)
    except Exception:
        await stream_infer_manager.force_recover(user_id, reason="error")
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
        ws_manager.disconnect(user_id, websocket)
        if not ws_manager.get_user_connections(user_id):
            notify_user_disconnected(user_id)
    finally:
        await stream_infer_manager.force_recover(user_id, reason="finally")