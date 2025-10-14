from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketDisconnect
from ..utils.ws_connection_manager import ws_manager

from collections import deque
import os, json, time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional
from fastapi import Body, Query

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

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _base_out_dir() -> str:
    # 優先使用你的專案固定路徑（若有 app/utils/paths.py ）
    try:
        from ..utils.paths import SC_ROOT
        return os.path.join(SC_ROOT, "data", "ws_frames")
    except Exception:
        return os.path.join(os.getcwd(), "tmp_ws_frames")

def _frame_to_video_item_exact(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    完全依照行動端傳入格式儲存，不誤合、多餘展開。
    每一幀只儲存:
      - boxes: [ [x1, y1, x2, y2], ... ] (每個人一筆)
      - keypoints: [ [ [x,y], [x,y], ... ], ... ] (每個人一組)
    """
    persons = payload.get("persons", [])
    boxes: List[List[float]] = []
    keypoints: List[List[List[float]]] = []

    for person in persons:
        # 正確地對應每位 person 的 bbox
        bbox = person.get("bbox", [])
        if len(bbox) == 4:
            boxes.append([float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])])

        # 為每位 person 建立一組 keypoints（僅取 x,y）
        kp_pairs = []
        for kp in person.get("keypoints", []):
            kp_pairs.append([float(kp.get("x", 0.0)), float(kp.get("y", 0.0))])
        keypoints.append(kp_pairs)

    # 完整結構：符合影片級 json 每幀一項格式
    return {
        "boxes": boxes,
        "keypoints": keypoints
    }

# --- 共用：追加一幀到「影片級 JSON」 ---
def _append_video_frame(user_id: str, ts_ms: int, payload: Dict[str, Any]) -> str:
    try:
        dt = datetime.fromtimestamp(float(ts_ms) / 1000.0)
    except Exception:
        dt = datetime.utcnow()
    date_str = dt.strftime("%Y%m%d")

    base_dir = _base_out_dir()
    out_dir = os.path.join(base_dir, f"user_{user_id}")
    _ensure_dir(out_dir)
    video_path = os.path.join(out_dir, f"{date_str}.json")

    # 轉一幀成 {boxes, keypoints}（完全照行動端輸入取值）
    item = _frame_to_video_item_exact(payload)

    # 讀舊檔 → append → 原子覆寫
    frames: List[Dict[str, Any]] = []
    if os.path.exists(video_path):
        try:
            with open(video_path, "r", encoding="utf-8") as fr:
                frames = json.load(fr)
                if not isinstance(frames, list):
                    frames = []
        except Exception:
            frames = []

    frames.append(item)

    tmp = f"{video_path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fw:
        json.dump(frames, fw, ensure_ascii=False, indent=2)
    os.replace(tmp, video_path)

    return video_path

# --- 新增：WebSocket 端點，持續接收並存成影片級 JSON ---
@ws_test_router.websocket("/ws/test/save")
async def ws_save_video_json(websocket: WebSocket):
    # 先用 query 的 user_id；payload 有 user_id 會覆蓋
    user_id = websocket.query_params.get("user_id", "unknown")

    # ★ 用 manager 來 accept + 記錄連線
    await ws_manager.connect(user_id, websocket)

    try:
        while True:
            # 文字 JSON（fallback bytes）
            try:
                text = await websocket.receive_text()
            except Exception:
                data = await websocket.receive_bytes()
                text = data.decode("utf-8", errors="ignore")

            payload = json.loads(text)

            # 以 payload 的 user_id 為準（若不同，可直接覆寫變數用於存檔）
            if "user_id" in payload:
                user_id = str(payload["user_id"])

            ts_ms = payload.get("ts_ms") or payload.get("timestamp_ms") or int(time.time() * 1000)

            # 這裡呼叫你原本的「追加一幀到影片級 JSON」方法
            video_path = _append_video_frame(user_id=user_id, ts_ms=ts_ms, payload=payload)

            # 回 ACK（你也可以改成 ws_manager.send_json(...) 廣播給同 user 的其它連線）
            await websocket.send_json({"ok": True, "saved": os.path.basename(video_path)})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"ok": False, "error": str(e)})
        except Exception:
            pass
    finally:
        # ★ 確保斷線後被清理
        ws_manager.disconnect(user_id, websocket)