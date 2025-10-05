import json
from fastapi import APIRouter, WebSocket
import requests
from starlette.websockets import WebSocketDisconnect
import traceback
import aiohttp  

from ..utils.ws_connection_manager import ws_manager
from ..utils.stream_infer_manager import stream_infer_manager
from ..utils.ws_message_dispatcher import handle_ws_text, notify_user_disconnected

from ..service.fall_event_service import add_fall_event
pose_router = APIRouter(tags=["姿態偵測與跌倒事件"])

def on_fall_start(payload: dict):
    # 從 StreamInferManager 收到的 payload 裡會含有剛才整理過的 clip
    clip = payload.get("result", {}).get("clip")  # 這裡已經是 JSON 友善的結構了
    elder_id = int(payload.get("user_id"))        # 統一成 int

    body = {
        "elder_id": elder_id,
        "start": clip.get("start"),
        "end": clip.get("end"),
    }

    # 正確的送法 + 正確的 log
    url = "https://17c5b07c0a11.ngrok-free.app/webhook/elder"
    r = requests.post(url, json=body, timeout=5)
    print("[WEBHOOK] POST", url, "payload=", json.dumps(body, ensure_ascii=False), "status=", r.status_code)

async def on_fall_recover(
    user_id: str,
    start_time: str | None = None,
    end_time: str | None = None,
    peak_score: float | None = None,
    result: dict | None = None,
    score: float | None = None,
    reason: str | None = None,
    **kwargs,
):
    print("[FALL_RECOVER]", user_id, start_time, end_time, peak_score, "reason=", reason)
    return

# 多分類事件（坐/躺）開始
async def on_state_event_start(user_id: str, event_name: str, start_time: str, peak_score: float, 
                               prev_action_name: str, curr_action_name: str, payload: dict):
    print("[STATE_START]", user_id, event_name, start_time, peak_score,
          "prev=", prev_action_name, "curr=", curr_action_name)
    # TODO: 寫 DB / 通知 / 排程（可記錄 prev/curr 便於分析連貫動作）

# 多分類事件（坐/躺）復原
async def on_state_event_recover(user_id: str, event_name: str, start_time: str, end_time: str,
                                 peak_score: float, prev_action_name: str, curr_action_name: str, payload: dict):
    print("[STATE_RECOVER]", user_id, event_name, start_time, end_time, peak_score,
          "prev=", prev_action_name, "curr=", curr_action_name)
    # TODO: 結束事件 / 入庫

stream_infer_manager.set_handlers(
    on_fall_start=on_fall_start,
    on_fall_recover=on_fall_recover,
    on_state_event_start=on_state_event_start,
    on_state_event_recover=on_state_event_recover
)

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
            # 這裡如果 parse/處理有任何錯誤，下面的 except 會印 traceback
            await handle_ws_text(user_id, text)

    except WebSocketDisconnect as e:
        code = getattr(e, "code", None)
        client = getattr(websocket, "client", None)
        host = getattr(client, "host", "?")
        port = getattr(client, "port", "?")
        print(f"[WS][ROUTE] user={user_id} disconnected. code={code} from {host}:{port}")
        ws_manager.disconnect(user_id, websocket)
        await stream_infer_manager.force_recover(user_id, reason=f"disconnect(code={code})")
        if not ws_manager.get_user_connections(user_id):
            notify_user_disconnected(user_id)

    except Exception as e:
        print(f"[WS][ROUTE][ERROR] user={user_id}: {e}")
        traceback.print_exc()
        await stream_infer_manager.force_recover(user_id, reason="error")
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
        ws_manager.disconnect(user_id, websocket)
        if not ws_manager.get_user_connections(user_id):
            notify_user_disconnected(user_id)

    finally:
        # 保險回復
        await stream_infer_manager.force_recover(user_id, reason="finally")


