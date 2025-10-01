from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketDisconnect
from ..utils.ws_connection_manager import ws_manager

from collections import deque
import time
import traceback

ws_test_router = APIRouter(tags=["WS 測試"])

@ws_test_router.websocket("/ws/test")
async def websocket_test(websocket: WebSocket):
    # 1) 確保 user_id 作為查詢參數傳遞
    user_id = websocket.query_params.get("user_id")
    if not user_id or not user_id.strip():
        await websocket.close(code=1008, reason="Missing required query param: user_id")
        return

    # 可調整參數：列印頻率與視窗秒數
    try:
        fps_log_every = float(websocket.query_params.get("fps_log_every", "1.0"))  # 每幾秒列印一次
        fps_window_sec = float(websocket.query_params.get("fps_window", "5.0"))    # 視窗平均(秒)
    except Exception:
        fps_log_every = 1.0
        fps_window_sec = 5.0

    # 2) 建立 WebSocket 連接
    await ws_manager.connect(user_id, websocket)
    print(f"User {user_id} connected to /ws/test. (fps_log_every={fps_log_every}s, fps_window={fps_window_sec}s)")

    # ---- FPS 統計變數 ----
    start_ts = time.time()
    last_log_ts = start_ts
    prev_ts = None
    total_msgs = 0

    # 1 秒視窗的即時 FPS（近 1s 收到幾個）
    win1s_ts = deque()

    # 可調整秒數視窗（預設 5s）的平均 FPS
    winXs_ts = deque()

    # 記錄最近 N 次到達間隔（用來看 jitter）
    inter_arrivals = deque(maxlen=200)

    try:
        while True:
            # 3) 接收來自客戶端的訊息（預期多為文字/JSON）
            message = await websocket.receive_text()
            print(f"Received from user {user_id}: {message}")
            
            now = time.time()
            total_msgs += 1

            # ---- 到達間隔（jitter）----
            if prev_ts is not None:
                dt = now - prev_ts
                inter_arrivals.append(dt)
            else:
                dt = 0.0
            prev_ts = now

            # ---- 1 秒即時 FPS ----
            win1s_ts.append(now)
            while win1s_ts and (now - win1s_ts[0] > 1.0):
                win1s_ts.popleft()
            inst_fps = float(len(win1s_ts))

            # ---- 視窗平均 FPS（預設 5 秒）----
            winXs_ts.append(now)
            while winXs_ts and (now - winXs_ts[0] > fps_window_sec):
                winXs_ts.popleft()
            if now - start_ts > 0:
                win_fps = len(winXs_ts) / min(fps_window_sec, now - start_ts)
                avg_fps = total_msgs / (now - start_ts)
            else:
                win_fps = 0.0
                avg_fps = 0.0

            # ---- 依照設定頻率列印 ----
            if now - last_log_ts >= fps_log_every:
                if inter_arrivals:
                    avg_dt = sum(inter_arrivals) / len(inter_arrivals)
                    min_dt = min(inter_arrivals)
                    max_dt = max(inter_arrivals)
                else:
                    avg_dt = min_dt = max_dt = 0.0

                print(
                    "[WS TEST][FPS] "
                    f"user={user_id} | inst={inst_fps:.1f} fps | win{fps_window_sec:.0f}s={win_fps:.2f} fps | "
                    f"avg={avg_fps:.2f} fps | "
                    f"dt={dt*1000:.1f} ms (avg {avg_dt*1000:.1f}, min {min_dt*1000:.1f}, max {max_dt*1000:.1f}) | "
                    f"total={total_msgs}"
                )
                last_log_ts = now

            # 4) 回傳確認消息給客戶端（保留原本行為）
            #    若你怕回傳太頻繁影響頻寬，可以改成每 N 則回一次
            response = {"ok": True, "received_len": len(message)}
            await websocket.send_json(response)

    except WebSocketDisconnect as e:
        code = getattr(e, "code", None)
        print(f"User {user_id} disconnected from /ws/test. code={code}")
        ws_manager.disconnect(user_id, websocket)

    except Exception as e:
        print(f"Error for user {user_id} in /ws/test: {e}")
        traceback.print_exc()
        ws_manager.disconnect(user_id, websocket)
        try:
            await websocket.close(code=1011, reason="Internal server error")
        except Exception:
            pass
