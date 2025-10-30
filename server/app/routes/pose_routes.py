# app/routes/pose_routes.py
import json
import traceback
import aiohttp
from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketDisconnect

from ..utils.ws_connection_manager import ws_manager
from ..utils.ws_message_dispatcher import handle_ws_text, notify_user_disconnected
from ..service.fall_event_service import add_fall_event
from ..service.sit_event_service import add_sit_event
from ..service.sleep_records_service import add_sleep_record

# 使用模組級單例 + 初始化工具
from ..utils import stream_infer_manager as sim_mod

pose_router = APIRouter(tags=["姿態偵測與跌倒事件"])

# 你原本使用的固定欄位（如需）
LOCATION = "客廳"
POSE_BEFORE_FALL = "站立"

# ---------------------------
# 事件 Handler（供推論核心呼叫）
# ---------------------------
async def on_fall_start(user_id: str, start_time: str, result: dict = None,
                        peak_score: float = None, payload: dict = None, clip: dict = None, **kwargs):
    """
    跌倒開始：寫 DB + 基本 log
    """
    elder_id = int(user_id)
    # 從參數 clip 或 kwargs 取得（manager 會以 clip=... 傳入）
    if clip is None:
        clip = kwargs.get("clip")
    _start = clip.get("start") if isinstance(clip, dict) else None
    _end = clip.get("end") if isinstance(clip, dict) else None

    body = {
        "elder_id": elder_id,
        "start": _start,
        "end": _end,
    }
    try:
        print(f"[FALL_START] {user_id} {start_time}")
        # 依你的 service 實作調整欄位
        record_id = await add_fall_event(
            user_id=int(user_id),
            location=LOCATION,
            pose_before_fall=POSE_BEFORE_FALL,
            detected_time=start_time
        )
        print(f"[FALL EVENT] user_id={user_id} recorded to DB.")
        url = "https://smartcare.southeastasia.cloudapp.azure.com/eric/webhook/elder"
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=body, timeout=5) as response:
                print("[WEBHOOK] POST", url, "payload=", json.dumps(body, ensure_ascii=False), "status=", response.status)
                try:
                    response_data = await response.json()
                    print("[WEBHOOK RESPONSE] Received:", json.dumps(response_data, ensure_ascii=False))
                except Exception as e:
                    print("[WEBHOOK RESPONSE][ERROR]", str(e))
    except Exception as e:
        print(f"[FALL_START][ERROR] user_id={user_id}: {e}")
        traceback.print_exc()

async def on_fall_recover(user_id: str, start_time: str, end_time: str,
                          peak_score: float = None, reason: str = "", payload: dict = None, **kwargs):
    """
    跌倒恢復：寫 log（若你有要更新 DB 的 end_time，可在此補寫）
    """
    try:
        print(f"[FALL_RECOVER] {user_id} {start_time} {end_time} {peak_score} reason= {reason}")
    except Exception as e:
        print(f"[FALL_RECOVER][ERROR] user={user_id}: {e}")
        traceback.print_exc()

async def on_state_event_start(user_id: str, event_name: str,
                               start_time: str, peak_score: float = None,
                               prev_action_name: str = None, curr_action_name: str = None,
                               payload: dict = None, clip: dict | None = None, **kwargs):
    """
    多動作：事件開始（例如 walk）
    """
    try:
        print(f"[STATE_START] user={user_id} event={event_name} at {start_time} "
              f"peak={peak_score if peak_score is not None else 'n/a'} "
              f"prev={prev_action_name or 'none'} -> curr={curr_action_name or event_name}")

    except Exception as e:
        print(f"[STATE_START][ERROR] user={user_id}: {e}")
        traceback.print_exc()

async def on_state_event_recover(user_id: str, event_name: str,
                                 start_time: str, end_time: str, peak_score: float = None,
                                 prev_action_name: str = None, curr_action_name: str = None,
                                 payload: dict = None, clip: dict | None = None, **kwargs):
    """
    多動作：事件恢復（例如 walk -> none 或 walk -> 另一個事件）
    """
    try:
        if prev_action_name == "sitstill":
            await add_sit_event(user_id=user_id, start_at=start_time, end_at=end_time)
            print(f"[SIT EVENT] user_id={user_id} recorded to DB.")
        elif prev_action_name == "liestill":
            await add_sleep_record(user_id=user_id, start_time=start_time, end_time=end_time)
            print(f"[SLEEP RECORD] user_id={user_id} recorded to DB.")
        print(f"[STATE_RECOVER] {user_id} {event_name} {start_time} {end_time} "
              f"{peak_score if peak_score is not None else 'n/a'} "
              f"prev= {prev_action_name or 'none'} curr= {curr_action_name or 'none'}")
    except Exception as e:
        print(f"[STATE_RECOVER][ERROR] user={user_id}: {e}")
        traceback.print_exc()


# ---------------------------
# WebSocket 入口
# ---------------------------
@pose_router.websocket("/ws/pose")
async def ws_pose(websocket: WebSocket):
    # 1) 驗證參數
    user_id = websocket.query_params.get("user_id")
    if not user_id or not user_id.strip():
        await websocket.close(code=1008, reason="Missing required query param: user_id")
        return
    user_id = user_id.strip()

    # 2) 連線登記
    await ws_manager.connect(user_id, websocket)
    print(f"[WS] User {user_id} connected")

    # 3) 直接使用模組級單例（現行版本於 import 時建立）
    manager = sim_mod.stream_infer_manager

    # 4) 綁定事件 handlers（可覆蓋既有設定）
    try:
        manager.set_handlers(
            on_fall_start=on_fall_start,
            on_fall_recover=on_fall_recover,
            on_state_event_start=on_state_event_start,
            on_state_event_recover=on_state_event_recover,
        )
    except Exception as e:
        print(f"[STREAM][HANDLERS][ERROR] user={user_id}: {e}")
        traceback.print_exc()

    # 5) 主循環：接收文字訊息交給 dispatcher
    try:
        while True:
            text = await websocket.receive_text()
            await handle_ws_text(user_id, text)

    except WebSocketDisconnect as e:
        # 使用者主動離線
        code = getattr(e, "code", None)
        print(f"[WS][DISCONNECT] user={user_id} code={code}")
        ws_manager.disconnect(user_id, websocket)
        # 清理該 user 的狀態
        try:
            await manager.force_recover(user_id, reason=f"disconnect(code={code})")
        except Exception as e2:
            print(f"[STREAM][force_recover][ERROR] user={user_id}: {e2}")
            traceback.print_exc()
        # 若該 user 已無連線，發送通知
        if not ws_manager.get_user_connections(user_id):
            notify_user_disconnected(user_id)

    except Exception as e:
        # 未預期錯誤
        print(f"[WS][ROUTE][ERROR] user={user_id}: {e}")
        traceback.print_exc()
        try:
            await manager.force_recover(user_id, reason="error")
        except Exception as e2:
            print(f"[STREAM][force_recover][ERROR] user={user_id}: {e2}")
            traceback.print_exc()
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
        ws_manager.disconnect(user_id, websocket)
        if not ws_manager.get_user_connections(user_id):
            notify_user_disconnected(user_id)

    finally:
        # 雙保險清理
        try:
            await manager.force_recover(user_id, reason="finally")
        except Exception as e2:
            print(f"[STREAM][force_recover][ERROR][finally] user={user_id}: {e2}")
            traceback.print_exc()