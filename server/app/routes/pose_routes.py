from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketDisconnect
import traceback

from ..utils.ws_connection_manager import ws_manager
from ..utils.stream_infer_manager import stream_infer_manager
from ..utils.ws_message_dispatcher import handle_ws_text, notify_user_disconnected

from ..service.fall_event_service import add_fall_event
pose_router = APIRouter(tags=["姿態偵測與跌倒事件"])

async def on_fall_start(user_id: str, start_time: str, result: dict, clip: dict):
    print("[FALL_START]", user_id, start_time, result["probs"][result["pred_idx"]])
    # print(f"{clip20}")
    # 若你希望一開始就入庫，保留原本功能：
    await add_fall_event(
        user_id=user_id,
        detected_time=start_time,
        location="客廳",
        pose_before_fall="正常行走"
    )

async def on_fall_recover(user_id: str, start_time: str, end_time: str, peak_score: float, result: dict):
    print("[FALL_RECOVER]", user_id, start_time, end_time, peak_score)
    return

# 多事件（坐/躺）開始
async def on_state_event_start(user_id: str, event_name: str, start_time: str, peak_score: float, 
                               prev_action_name: str, curr_action_name: str, payload: dict):
    print("[STATE_START]", user_id, event_name, start_time, peak_score,
          "prev=", prev_action_name, "curr=", curr_action_name)
    # TODO: 寫 DB / 通知 / 排程（可記錄 prev/curr 便於分析連貫動作）

# 多事件（坐/躺）復原
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
