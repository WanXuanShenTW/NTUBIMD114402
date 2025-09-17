# app/routes/ws_pose_router.py
from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketDisconnect

from ..utils.ws_connection_manager import ws_manager
from ..utils.stream_infer_manager import stream_infer_manager
from ..utils.ws_message_dispatcher import handle_ws_text, notify_user_disconnected

from ..service.fall_event_service import add_fall_event
pose_router = APIRouter()

async def on_fall_start(user_id: str, start_time: str, start_frame: int, result: dict):
    """
    跌倒「開始」時觸發：這裡只做立即通知/記憶即可，不寫入 DB（你需求是恢復後才存）。
    如果你想先開一筆暫存事件（open），這裡也可以做，但依你需求可以先不寫庫。
    """
    # e.g. 發 LINE 通知 / 推播（選用）
    # await notify_svc.push_fall_detected(user_id, start_time, start_frame)
    print("[FALL_START]", user_id, start_time, result["probs"][result["pred_idx"]])
    await add_fall_event(
        user_id=user_id,
        detected_time=start_time,
        location="客廳",
        pose_before_fall="正常行走"
    )
    return

async def on_fall_recover(user_id: str, start_time: str, end_time: str,
                          start_frame: int, end_frame: int, peak_score: float, result: dict):
    """
    跌倒「恢復正常」時觸發：這裡才真正「寫入資料庫」。
    你可以在這裡組合你要存的欄位，再呼叫你的 DAO/Service。
    """
    # ★★★ 在這裡呼叫你的 DAO/Service ★★★
    # 下面是範例，請替換為你的方法與參數名稱
    # await inference_svc.insert_fall_event(
    #     user_id=user_id,
    #     start_time=start_time,     # 'YYYY-MM-DD HH:MM:SS'
    #     end_time=end_time,         # 'YYYY-MM-DD HH:MM:SS'
    #     start_frame=start_frame,
    #     end_frame=end_frame,
    #     peak_score=peak_score
    # )
    print("[FALL_RECOVER]", user_id, start_time, end_time, peak_score)
    return

# 模組載入時註冊 Hook（之後 manager 偵測到事件就會呼叫這兩個函式）
stream_infer_manager.set_handlers(
    on_fall_start=on_fall_start,
    on_fall_recover=on_fall_recover
)
print("[HOOKS] registered:", list(stream_infer_manager._handlers.keys()))

@pose_router.websocket("/ws/pose")
async def ws_pose(websocket: WebSocket):
    # 1) 強制要求 user_id
    user_id = websocket.query_params.get("user_id")
    if not user_id or not user_id.strip():
        await websocket.close(code=1008, reason="Missing required query param: user_id")
        return

    # 2) 建立連線（你的 ws_manager 版本）
    await ws_manager.connect(user_id, websocket)

    try:
        while True:
            text = await websocket.receive_text()
            # 3) 交給 utils 做解析與推論（依 user_id 分流到各自的 buffer）
            await handle_ws_text(user_id, text)

    except WebSocketDisconnect:
        # 4) 斷線：用「已知的 user_id」清理即可
        ws_manager.disconnect(user_id, websocket)
        print("disconnect recover")
        await stream_infer_manager.force_recover(user_id, reason="disconnect")

        # 若該 user 沒有其他連線，才釋放該 user 的推論 buffer
        if not ws_manager.get_user_connections(user_id):
            notify_user_disconnected(user_id)

    except Exception:
        print("force recover error")
        await stream_infer_manager.force_recover(user_id, reason="error")
        # 發生非預期錯誤時也關閉並清理
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
        ws_manager.disconnect(user_id, websocket)
        if not ws_manager.get_user_connections(user_id):
            notify_user_disconnected(user_id)
    finally:
        print("finally recover")
        await stream_infer_manager.force_recover(user_id, reason="finally")