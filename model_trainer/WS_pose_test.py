
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WS_mobile_sim.py
- 模擬行動端傳輸格式（type="frame"），把 pose 與 detect 資料合併成單包
- 缺少欄位以「現實時間」補齊（timestamp_ms / detect_ts / detect_age_ms / expire_ms）
- 支援 target_fps 與 detect_fps（偵測為較低頻率，非偵測幀沿用上次結果並更新 age）
- 以 WebSocket 連線送出，方便你在桌面端重現行動端資料流

Install:
  pip install websockets

Run:
  python WS_mobile_sim.py
"""

import asyncio
import json
import os
import time
import random
from typing import List, Dict, Any, Optional

import websockets

# ==========================
# 設定區（直接改這裡即可）
# ==========================
POSE_PATH   = "sources/test/Office_video (18)_back_front-pose.json"    # 骨架檔（每幀含 boxes + keypoints）
DETECT_PATH = "sources/test/Office_video (18)_back_front-detect.json"    # 物件檔（每幀含 objects[] 或 boxes[]）

# WebSocket 伺服器（擇一）
WS_URL  = "wss://331cdcbf68ce.ngrok-free.app/ws/pose?user_id=1"                                 # 若給完整 URL（含 ws/wss 與查詢參數），就用這個
HOST    = "localhost"                        # 否則用下列 4 個組 URI
PORT    = 8000
WS_PATH = "/ws/pose"                         # 例如行動端連的 /ws/pose
USER_ID = 1                                  # 會放在 frame.user_id 以及 ?user_id= 查詢參數

# 播放節奏
TARGET_FPS = 10                              # 目標送包 FPS（frame.target_fps）
DETECT_FPS = 10                               # 偵測結果更新頻率（frame.detect_fps）
EXPIRE_AHEAD_MS = 300                        # frame.expire_ms = timestamp_ms + 300ms（可改）
RANDOM_DET_LAT_MS = (60, 220)                # 模擬偵測延遲（detect_ts = timestamp_ms - U(a,b)）

# 畫面資訊（若 JSON 無此欄位，就用這裡作為 image_size）
IMG_W, IMG_H = 640, 480

# 回放與環狀控制
START_FRAME_ID = 1                           # 第一包的 frame_id
LOOP_FOREVER = False                         # 播完是否循環
SLEEP_ON_SEND_SEC = 0                        # 若要節流（例如實際 10fps 但想慢點），可加一點 sleep

# ==========================
# 讀檔輔助
# ==========================
def read_json_list(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    # 容忍 dict 包裝
    if isinstance(data, dict):
        for k in ("frames", "data", "results", "records", "items"):
            v = data.get(k)
            if isinstance(v, list):
                return v
        # 其他：嘗試展平成單元素
        return [data]
    return []

# ==========================
# 轉換：pose / detect 幀 → persons / detections
# ==========================
def build_persons_from_pose(frame: Dict[str, Any]) -> List[Dict[str, Any]]:
    boxes = frame.get("boxes", [])
    kps_groups = frame.get("keypoints", [])
    persons = []
    # 每幀可能多人，這裡逐一轉換
    n = max(len(boxes), len(kps_groups))
    for i in range(n):
        bbox = boxes[i] if i < len(boxes) else None
        key_xy = kps_groups[i] if i < len(kps_groups) else []
        # keypoints: [[x,y], ...] → [{x,y,conf}, ...]
        kps = []
        for pt in key_xy:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                x, y = float(pt[0]), float(pt[1])
                conf = 0.95 if (x is not None and y is not None) else 0.0
                kps.append({"x": x, "y": y, "conf": conf})
        if bbox is None and not kps:
            continue
        persons.append({
            "score": 0.9,             # 模擬分數
            "bbox": bbox,             # 若沒 bbox 也可不填（行動端可接受）
            "keypoints": kps,
        })
    return persons

def build_detections_from_detect(frame: Dict[str, Any]) -> List[Dict[str, Any]]:
    dets = []
    if "objects" in frame and isinstance(frame["objects"], list):
        for d in frame["objects"]:
            bbox = d.get("bbox") or d.get("xyxy") or d.get("box")
            if not (isinstance(bbox, list) and len(bbox) == 4):
                continue
            x1, y1, x2, y2 = bbox
            dets.append({
                "class_name": d.get("class_name") or d.get("label") or "obj",
                "confidence": float(d.get("confidence", 0.0)),
                "bbox": [x1, y1, x2, y2],
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })
        return dets

    # 後備：若沒有 objects，就把 boxes 畫成無類別方框
    for bbox in frame.get("boxes", []):
        if isinstance(bbox, list) and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            dets.append({
                "class_name": "box",
                "confidence": 1.0,
                "bbox": [x1, y1, x2, y2],
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })
    return dets

# ==========================
# WebSocket 客戶端
# ==========================
def build_ws_uri() -> str:
    if WS_URL:
        return WS_URL
    scheme = os.getenv("SCHEME", "ws")
    base = f"{scheme}://{HOST}:{PORT}{WS_PATH}"
    # 加上 user_id 查詢參數（行動端常這樣帶）
    sep = '&' if '?' in base else '?'
    return f"{base}{sep}user_id={USER_ID}"

async def stream_frames():
    pose_frames   = read_json_list(POSE_PATH)
    detect_frames = read_json_list(DETECT_PATH)

    total = min(len(pose_frames), len(detect_frames))
    if total == 0:
        print("[SIM] 沒有可用幀，請確認 POSE_PATH / DETECT_PATH")
        return

    # 偵測節奏（以 target_fps 為基準，間隔多少幀更新一次偵測）
    stride = max(1, round(TARGET_FPS / max(1, DETECT_FPS)))

    # 初始化偵測狀態（沿用上一筆資料以模擬低頻偵測）
    last_det = build_detections_from_detect(detect_frames[0])
    detect_seq = 1
    detect_ts  = int(time.time() * 1000)  # 初始偵測時間
    frames_sent = 0

    uri = build_ws_uri()
    print(f"[SIM] Connect to: {uri}")
    async with websockets.connect(uri, max_size=2**22, ping_interval=20) as ws:
        print("[SIM] connected.")
        i = 0
        frame_id = START_FRAME_ID

        while True:
            # 計算這一幀的時間
            now_ms = int(time.time() * 1000)
            # 是否更新偵測（依 stride）
            if frames_sent % stride == 0:
                # 取下一個 detect 幀（環狀）
                det_idx = i % len(detect_frames)
                last_det = build_detections_from_detect(detect_frames[det_idx])
                detect_seq += 1
                det_latency = random.randint(RANDOM_DET_LAT_MS[0], RANDOM_DET_LAT_MS[1])
                detect_ts = now_ms - det_latency

            # 構造 persons（每幀更新）
            pose_idx = i % len(pose_frames)
            persons = build_persons_from_pose(pose_frames[pose_idx])

            # 影像尺寸（若來源幀沒帶 image_size 就用設定值）
            img_w = pose_frames[pose_idx].get("image_size", {}).get("width", IMG_W)
            img_h = pose_frames[pose_idx].get("image_size", {}).get("height", IMG_H)

            # 合成行動端格式
            frame_msg = {
                "type": "frame",
                "frame_id": frame_id,
                "timestamp_ms": now_ms,
                "expire_ms": now_ms + EXPIRE_AHEAD_MS,
                "target_fps": TARGET_FPS,
                "detect_fps": DETECT_FPS,
                "user_id": USER_ID,
                "image_size": {"width": img_w, "height": img_h},
                "detect_seq": detect_seq,
                "detect_ts": detect_ts,
                "detect_age_ms": max(0, now_ms - detect_ts),
                "persons": persons,
                "detections": last_det,
            }

            await ws.send(json.dumps(frame_msg, ensure_ascii=False))
            frames_sent += 1
            i += 1
            frame_id += 1

            # 節奏控制（近似 target_fps）
            await asyncio.sleep(max(0.0, 1.0 / max(1, TARGET_FPS)))
            if SLEEP_ON_SEND_SEC > 0:
                await asyncio.sleep(SLEEP_ON_SEND_SEC)

            # 是否結束或循環
            if i >= total:
                if LOOP_FOREVER:
                    i = 0
                else:
                    break

        print(f"[SIM] done. sent frames: {frames_sent}")
        # 可選：接收伺服器回覆
        try:
            while True:
                msg = await ws.recv()
                print("[SERVER]", msg)
        except websockets.ConnectionClosed:
            print("[SIM] connection closed.")

if __name__ == "__main__":
    asyncio.run(stream_frames())
