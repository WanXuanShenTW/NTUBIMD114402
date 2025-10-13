import json
from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketDisconnect
import traceback
import aiohttp

from ..utils.ws_connection_manager import ws_manager
from ..utils.ws_message_dispatcher import handle_ws_text, notify_user_disconnected
from ..service.fall_event_service import add_fall_event

# ✅ 僅匯入 Lazy Loader，不要載入實體
from ..utils.stream_infer_manager import stream_infer_manager

pose_router = APIRouter(tags=["姿態偵測與跌倒事件"])
LOCATION = "客廳"
POSE_BEFORE_FALL = "站立"
# -------------------------------
# Handler callback functions
# -------------------------------

async def on_fall_start(user_id: str, start_time: str, result: dict, clip: dict):
    print("[FALL_START]", user_id, start_time)
    elder_id = int(user_id)
    body = {
        "elder_id": elder_id,
        "start": clip.get("start"),
        "end": clip.get("end"),
    }
    await add_fall_event(
        user_id=elder_id,
        detected_time=start_time,
        location=LOCATION,
        pose_before_fall=POSE_BEFORE_FALL
    )
    print(f"[FALL EVENT] elder_id={user_id} recorded to DB.")
    url = "https://smartcare.southeastasia.cloudapp.azure.com/eric/webhook/elder"
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=body, timeout=5) as response:
            print("[WEBHOOK] POST", url, "payload=", json.dumps(body, ensure_ascii=False), "status=", response.status)
            try:
                response_data = await response.json()
                print("[WEBHOOK RESPONSE] Received:", json.dumps(response_data, ensure_ascii=False))
            except Exception as e:
                print("[WEBHOOK RESPONSE][ERROR]", str(e))

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
    # add_fall_event(
    #     user_id=int(user_id),
    #     detected_time=start_time,
    #     end=end_time,
    #     location=LOCATION,
    #     pose_before_fall=POSE_BEFORE_FALL
    # )
    # print(f"[FALL EVENT] elder_id={user_id} recorded to DB.")
    return

async def on_state_event_start(user_id: str, event_name: str, start_time: str, peak_score: float,
                               prev_action_name: str, curr_action_name: str, payload: dict):
    print(f"[STATE_START] user={user_id} event={event_name} at {start_time} "
          f"peak={peak_score:.3f} prev={prev_action_name} -> curr={curr_action_name}")

async def on_state_event_recover(user_id: str, event_name: str, start_time: str, end_time: str,
                                 peak_score: float, prev_action_name: str, curr_action_name: str, payload: dict):
    print("[STATE_RECOVER]", user_id, event_name, start_time, end_time, peak_score,
          "prev=", prev_action_name, "curr=", curr_action_name)

# ------------------------------------------
# WebSocket 主流程
# ------------------------------------------

@pose_router.websocket("/ws/pose")
async def ws_pose(websocket: WebSocket):
    user_id = websocket.query_params.get("user_id")

    # ✅ Lazy load：第一次有 WebSocket 才初始化模型
    stream_infer_manager.set_handlers(
        on_fall_start=on_fall_start,
        on_fall_recover=on_fall_recover,
        on_state_event_start=on_state_event_start,
        on_state_event_recover=on_state_event_recover
    )

    if not user_id or not user_id.strip():
        await websocket.close(code=1008, reason="Missing required query param: user_id")
        return

    await ws_manager.connect(user_id, websocket)

    try:
        while True:
            text = await websocket.receive_text()
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
        await stream_infer_manager.force_recover(user_id, reason="finally")
