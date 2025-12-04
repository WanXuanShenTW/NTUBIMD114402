# app/routes/pose_routes.py
import json
import traceback
import aiohttp
from datetime import datetime, timedelta
from collections import defaultdict
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

LOCATION = "客廳"
POSE_BEFORE_FALL = "站立"

# === [新增] 用來記錄每個使用者「上一次跌倒結束」的時間 ===
# 格式: user_id (str) -> datetime object
USER_LAST_FALL_END = defaultdict(lambda: None)

# 設定「跌倒後多久內的躺下」視為無效睡眠 (秒)
FALL_COOLDOWN_SECONDS = 30 

# ---------------------------
# 事件 Handler
# ---------------------------
async def on_fall_start(user_id: str, start_time: str, result: dict = None,
                        peak_score: float = None, prev_action_name: str | None = None,
                        curr_action_name: str | None = None, payload: dict = None,
                        clip: dict = None, location: str = None, **kwargs):
    """
    跌倒開始：寫 DB + 基本 log
    """
    elder_id = int(user_id)
    if clip is None:
        clip = kwargs.get("clip")
    _start = clip.get("start") if isinstance(clip, dict) else None
    _end = clip.get("end") if isinstance(clip, dict) else None

    pose_before = (prev_action_name or POSE_BEFORE_FALL)
    body = {
        "elder_id": elder_id,
        "start": _start,
        "end": _end
    }
    
    # 優先使用傳入的 location，如果沒有則使用預設值
    final_location = location if location else LOCATION

    try:
        print(f"[FALL_START] {user_id} {start_time}")
        if pose_before.lower() in ["lie", "liestill", "lying"]:
            pose_before = "躺"
        elif pose_before.lower() in ["sit", "sitstill", "sitting"]:
            pose_before = "坐"
        else:
            pose_before = "走路"
            
        record_id = await add_fall_event(
            user_id=int(user_id),
            location=final_location,
            pose_before_fall=pose_before,
            detected_time=start_time
        )
        print(f"[FALL EVENT] user_id={user_id} recorded to DB. id={record_id}")
        
        # Webhook
        url = "https://smartcare.southeastasia.cloudapp.azure.com/eric/webhook/elder"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=body, timeout=10) as response:
                    # print("[WEBHOOK] POST payload=", json.dumps(body, ensure_ascii=False))
                    pass
        except Exception as e:
            print(f"[Webhook][Warn] 無法發送跌倒通知: {e}")
            
    except Exception as e:
        print(f"[FALL_START][ERROR] user_id={user_id}: {e}")
        traceback.print_exc()

async def on_fall_recover(user_id: str, start_time: str, end_time: str,
                          peak_score: float = None, reason: str = "", payload: dict = None, **kwargs):
    """
    跌倒恢復：記錄恢復時間，防止後續誤判睡眠
    """
    try:
        print(f"[FALL_RECOVER] {user_id} {start_time} {end_time} {peak_score} reason={reason}")
        
        # [新增] 記錄跌倒結束時間，供後續 filter 使用
        try:
            dt_end = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
            USER_LAST_FALL_END[user_id] = dt_end
            print(f"[FILTER] User {user_id} fall ended at {dt_end}. Cooldown starts.")
        except Exception:
            pass
            
    except Exception as e:
        print(f"[FALL_RECOVER][ERROR] user={user_id}: {e}")
        traceback.print_exc()

async def on_state_event_start(user_id: str, event_name: str,
                               start_time: str, peak_score: float = None,
                               prev_action_name: str = None, curr_action_name: str = None,
                               payload: dict = None, clip: dict | None = None, **kwargs):
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
    多動作事件結束：判斷是否寫入 DB (坐姿、睡眠)
    """
    try:
        # --- 處理坐姿 ---
        if prev_action_name == "sitstill":
            # [修正] 參數可能也是 start_at/end_at，請確認 sit_event_service
            # 這裡假設 add_sit_event 用 start_at
            await add_sit_event(user_id=user_id, start_at=start_time, end_at=end_time)
            print(f"[SIT EVENT] user_id={user_id} recorded to DB.")

        # --- 處理睡眠 (liestill) ---
        elif prev_action_name == "liestill":
            # [新增] 檢查是否為跌倒後的連帶動作
            last_fall = USER_LAST_FALL_END[user_id]
            is_valid_sleep = True
            
            if last_fall:
                try:
                    dt_start = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
                    diff = (dt_start - last_fall).total_seconds()
                    
                    # 如果睡眠開始時間 - 上次跌倒結束時間 < 冷卻時間 (例如 30秒)
                    # 且 diff >= 0 (確保不是舊資料)
                    if 0 <= diff < FALL_COOLDOWN_SECONDS:
                        print(f"[FILTER] Ignored sleep record for user {user_id}. "
                              f"Too close to fall (gap={diff:.1f}s < {FALL_COOLDOWN_SECONDS}s).")
                        is_valid_sleep = False
                except Exception as e_time:
                    print(f"[FILTER] Time parsing error: {e_time}")

            if is_valid_sleep:
                # [修正] 參數名稱改為 start_at, end_at 以匹配 service 定義
                await add_sleep_record(user_id=user_id, start_at=start_time, end_at=end_time)
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

    # 3) 直接使用模組級單例
    manager = sim_mod.stream_infer_manager

    # 4) 綁定事件 handlers
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

    # 5) 主循環
    try:
        while True:
            text = await websocket.receive_text()
            await handle_ws_text(user_id, text)

    except WebSocketDisconnect as e:
        code = getattr(e, "code", None)
        print(f"[WS][DISCONNECT] user={user_id} code={code}")
        ws_manager.disconnect(user_id, websocket)
        try:
            await manager.force_recover(user_id, reason=f"disconnect(code={code})")
        except Exception:
            pass
        if not ws_manager.get_user_connections(user_id):
            notify_user_disconnected(user_id)

    except Exception as e:
        print(f"[WS][ROUTE][ERROR] user={user_id}: {e}")
        try:
            await manager.force_recover(user_id, reason="error")
        except Exception:
            pass
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
        ws_manager.disconnect(user_id, websocket)
        if not ws_manager.get_user_connections(user_id):
            notify_user_disconnected(user_id)

    finally:
        try:
            await manager.force_recover(user_id, reason="finally")
        except Exception:
            pass