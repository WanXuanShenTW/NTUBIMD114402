
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
display_dual_fps_realtimesync.py

目標：影片「原生 FPS + 牆鐘同步」播放（跟一般播放器一樣順），
      骨架/偵測依自訂 FPS 前進。若處理負荷過高，**會丟幀**以維持時間同步。

- 影片以 wall-clock 對齊：依 elapsed_time 計算應顯示的 frame_idx，必要時跳幀。
- 骨架索引同樣依 elapsed_time 計算（與影片互不干擾）。
- 影片貼到畫布採 letterbox，骨架/框經同一映射疊上去（不變形）。
- 保留 two-stage / binary-only 推論整合（透過 run_on_json）。

熱鍵：Space 暫停/續播（維持時間對齊）、q/ESC 離開
"""

import os
import sys
import json
import time
from typing import List, Dict, Tuple, Optional

# 避免 wayland 外掛問題
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import numpy as np

# ==============================
# 🔧 設定區
# ==============================
# POSE_JSON_PATH   = "outputs/skeletons/multi/YOLO-pose/liestill/780385481.887354_back.json"
# DETECT_JSON_PATH = "outputs/skeletons/multi/YOLO-detect/liestill/780385481.887354_back.json"
# POSE_JSON_PATH   = "outputs/skeletons/multi/YOLO-pose/sitstill/Sitting (20).json"
# DETECT_JSON_PATH = "outputs/skeletons/multi/YOLO-detect/sitstill/Sitting (20).json"
POSE_JSON_PATH   = "outputs/skeletons/multi/YOLO-pose/walk/Walking (11).json"
DETECT_JSON_PATH = "outputs/skeletons/multi/YOLO-detect/walk/Walking (11).json"

MODE             = "two-stage"    # "two-stage" | "binary-only" | "none"

ENABLE_VIDEO     = True
# VIDEO_PATH       = "medias/train_video/multi/liestill/780385481.887354_back.mp4"
# VIDEO_PATH       = "medias/train_video/multi/sitstill/Sitting (20).mp4"
VIDEO_PATH       = "medias/train_video/multi/walk/Walking (11).mp4"
VIDEO_FRAME_OFFSET = 0            # JSON 第 0 幀對應影片的第幾幀（對齊起點用）

SKELETON_FPS     = 10              # 骨架的節奏（與影片無關）
JSON_FRAME_OFFSET = 0             # JSON 幀起始偏移

CANVAS_WIDTH     = 640
CANVAS_HEIGHT    = 480
SHOW_TOPK        = 3
DEBUG_DRAW_MAPPING = False

# 顏色
COLOR_SKELETON   = (255, 255, 0)
COLOR_KP         = (0, 255, 0)
COLOR_POSE_BOX   = (60, 180, 255)
COLOR_DET_BOX    = (100, 100, 255)
COLOR_MAP_BORDER = (0, 200, 255)

EXTRA_PY_PATHS   = []

# ==============================
# YOLOv8(=COCO-17) 關節連線
# ==============================
YOLO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

# ==============================
# IO & 解析
# ==============================
def _json_to_list(data):
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in ('frames', 'data', 'records', 'annotations', 'items', 'results'):
            v = data.get(k)
            if isinstance(v, list):
                return v
        try:
            items = sorted([(int(k), v) for k, v in data.items() if str(k).isdigit()], key=lambda x: x[0])
            return [v for _, v in items] if items else [data]
        except Exception:
            return [data]
    return []

def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return _json_to_list(data)

def parse_pose_frame(frame: dict):
    boxes = frame.get("boxes", []) if isinstance(frame, dict) else []
    raw_kps = frame.get("keypoints", []) if isinstance(frame, dict) else []
    kps_pp = []
    for person in raw_kps:
        pts = []
        for pt in person[:17]:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                x, y = float(pt[0]), float(pt[1])
                pts.append((x, y))
        kps_pp.append(pts)
    return kps_pp, boxes

def parse_detect_frame(frame: dict) -> List[dict]:
    out = []
    if not isinstance(frame, dict):
        return out
    objs = frame.get("objects", None)
    if isinstance(objs, list):
        for o in objs:
            name = o.get("class_name") or o.get("name") or o.get("label") or "obj"
            bbox = o.get("bbox") or o.get("xyxy") or o.get("box") or o.get("bbox_xyxy")
            conf = float(o.get("confidence", 1.0))
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                out.append({"class_name": name, "bbox": [float(b) for b in bbox], "confidence": conf})
        return out
    boxes = frame.get("boxes", [])
    for b in boxes:
        if isinstance(b, (list, tuple)) and len(b) == 4:
            out.append({"class_name": "box", "bbox": [float(x) for x in b], "confidence": 1.0})
    return out

# ==============================
# Letterbox 對映
# ==============================
def letterbox_params(src_w: int, src_h: int, dst_w: int, dst_h: int):
    scale = min(dst_w / src_w, dst_h / src_h) if (src_w > 0 and src_h > 0) else 1.0
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))
    x0 = (dst_w - new_w) // 2
    y0 = (dst_h - new_h) // 2
    return scale, x0, y0

def map_point_to_canvas(x: float, y: float, scale: float, x0: int, y0: int):
    return int(round(x * scale + x0)), int(round(y * scale + y0))

def map_box_to_canvas(x1: float, y1: float, x2: float, y2: float, scale: float, x0: int, y0: int):
    p1 = map_point_to_canvas(x1, y1, scale, x0, y0)
    p2 = map_point_to_canvas(x2, y2, scale, x0, y0)
    return p1[0], p1[1], p2[0], p2[1]

# ==============================
# 畫圖（經對映）
# ==============================
def draw_skeletons_mapped(canvas, keypoints_pp, scale, x0, y0,
                          point_color=(0,255,0), line_color=(255,255,0), thickness=2):
    for pts in keypoints_pp:
        for (a, b) in YOLO_EDGES:
            if a < len(pts) and b < len(pts):
                pa = map_point_to_canvas(*pts[a], scale, x0, y0)
                pb = map_point_to_canvas(*pts[b], scale, x0, y0)
                cv2.line(canvas, pa, pb, line_color, thickness)
        for (x, y) in pts:
            px, py = map_point_to_canvas(x, y, scale, x0, y0)
            cv2.circle(canvas, (px, py), 3, point_color, -1)

def draw_detections_mapped(canvas, objects, scale, x0, y0, color=(100,100,255), thickness=1):
    for det in objects:
        x1, y1, x2, y2 = det["bbox"]
        X1, Y1, X2, Y2 = map_box_to_canvas(x1, y1, x2, y2, scale, x0, y0)
        cv2.rectangle(canvas, (X1, Y1), (X2, Y2), color, thickness)
        label = f"{det['class_name']} {det.get('confidence', 0):.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_top = max(0, Y1 - th - 6)
        cv2.rectangle(canvas, (X1, y_top), (X1 + tw + 4, y_top + th + 6), color, -1)
        cv2.putText(canvas, label, (X1 + 2, y_top + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

# ==============================
# Loader（同介面）
# ==============================
def _ensure_paths():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    for p in EXTRA_PY_PATHS:
        if p and (p not in sys.path):
            sys.path.insert(0, p)

def import_module_local(module_name: str):
    _ensure_paths()
    import importlib
    return importlib.import_module(module_name)

def run_loader_results(module_name: str, pose_json: str, detect_json: Optional[str]):
    try:
        m = import_module_local(module_name)
        if hasattr(m, "run_on_json"):
            return m.run_on_json(pose_json, detect_json)
    except Exception as e:
        print(f"[Warn] 無法執行 {module_name}.run_on_json(): {e}")
    return [], None

def build_frame_result_index(results: List[dict], total_frames: int) -> List[Optional[dict]]:
    idx = [None] * total_frames
    for r in results:
        s, e = int(r["start_frame"]), int(r["end_frame"])
        s = max(0, s); e = min(total_frames-1, e)
        for f in range(s, e+1):
            prev = idx[f]
            if (prev is None) or (int(prev["start_frame"]) <= s):
                idx[f] = r
    return idx

def binary_is_fall(r: dict, class_names: Optional[List[str]]) -> bool:
    if "is_rare" in r:
        return bool(r.get("is_rare"))
    pred = str(r.get("pred","")).lower()
    if "fall" in pred:
        return True
    probs = r.get("probs", None)
    if isinstance(class_names, list) and isinstance(probs, list) and len(class_names) == len(probs):
        try:
            import numpy as _np
            fall_idx = class_names.index("fall")
            return int(_np.argmax(_np.array(probs))) == fall_idx
        except ValueError:
            pass
    return False

def extract_pred_prob(r: dict, class_names: Optional[List[str]], prefer_rare: bool = True):
    if prefer_rare and ("rare_name" in r) and ("score_rare" in r):
        try:
            return f"{r['rare_name']}", float(r["score_rare"])
        except Exception:
            pass
    probs = r.get("probs", None)
    pred = r.get("pred", None)
    if isinstance(probs, list) and class_names and len(probs) == len(class_names):
        import numpy as _np
        idx = int(_np.argmax(_np.array(probs)))
        return class_names[idx], float(probs[idx])
    if isinstance(probs, list):
        import numpy as _np
        idx = int(_np.argmax(_np.array(probs)))
        label = pred if isinstance(pred, str) and pred else f"idx{idx}"
        return label, float(probs[idx])
    return (str(pred) if pred else "N/A"), None

# ==============================
# 主流程：影片 wall-clock 同步 + 骨架自訂 FPS
# ==============================
def main():
    pose_data = read_json(POSE_JSON_PATH)
    det_data  = read_json(DETECT_JSON_PATH)
    total_json = min(len(pose_data), len(det_data))

    cap = None
    video_fps = None
    src_w = src_h = 0
    if ENABLE_VIDEO:
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            print(f"[Warn] 無法開啟影片：{VIDEO_PATH}，改為黑底播放")
            cap = None
        else:
            video_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            if video_fps <= 1e-3:
                video_fps = 30.0
            src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            total_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            print(f"[Info] Video opened: fps={video_fps:.3f}, size=({src_w},{src_h}), frames={total_vid}")

    if cap is None:
        # fallback：無影片時，用畫布尺寸當來源大小
        src_w, src_h = CANVAS_WIDTH, CANVAS_HEIGHT

    scale, x0, y0 = letterbox_params(src_w, src_h, CANVAS_WIDTH, CANVAS_HEIGHT)
    print(f"[Info] letterbox: scale={scale:.6f}, offset=({x0},{y0})")

    # 推論索引（一次性）
    bin_index = [None] * total_json; bin_classes = None
    mul_index = [None] * total_json; mul_classes = None
    if MODE in ("binary-only", "two-stage"):
        bres, bin_classes = run_loader_results("binary_cnn_lstm_loader", POSE_JSON_PATH, DETECT_JSON_PATH)
        bin_index = build_frame_result_index(bres, total_json)
        print(f"[Info] Binary windows={len(bres)} classes={bin_classes}")
        if MODE == "two-stage":
            mres, mul_classes = run_loader_results("multi_cnn_lstm_loader", POSE_JSON_PATH, DETECT_JSON_PATH)
            mul_index = build_frame_result_index(mres, total_json)
            print(f"[Info] Multi  windows={len(mres)} classes={mul_classes}")

    # 牆鐘同步
    start_t = time.perf_counter()
    last_sk_idx = -1
    cache = {}

    # 初始化影片指標（避免反覆 set 影響效能）
    last_v_idx = -1
    if cap is not None and VIDEO_FRAME_OFFSET > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(VIDEO_FRAME_OFFSET))
        last_v_idx = VIDEO_FRAME_OFFSET - 1

    paused = False

    while True:
        now = time.perf_counter()
        elapsed = now - start_t

        # 影片索引 = floor(elapsed * video_fps) + 偏移
        if cap is not None:
            target_v_idx = int(elapsed * video_fps) + int(VIDEO_FRAME_OFFSET)
            if target_v_idx >= total_vid:
                break  # 播放完
            if target_v_idx != last_v_idx:
                if target_v_idx == last_v_idx + 1:
                    # 正常遞增：直接 read 下一幀（最快）
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break
                else:
                    # 跳幀：seek 到目標幀，再 read
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_v_idx)
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break
                last_v_idx = target_v_idx

            # 建立畫布並貼影片
            canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)
            new_w = int(round(src_w * scale)); new_h = int(round(src_h * scale))
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            canvas[y0:y0+new_h, x0:x0+new_w] = resized
        else:
            # 無影片：以骨架節奏刷新
            canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)

        # 骨架索引 = floor(elapsed * SKELETON_FPS) + 偏移
        sk_idx = JSON_FRAME_OFFSET + int(elapsed * SKELETON_FPS)
        if sk_idx >= total_json:
            sk_idx = total_json - 1  # clamp

        if sk_idx != last_sk_idx:
            kps_pp, pose_boxes = parse_pose_frame(pose_data[sk_idx])
            det_objs = parse_detect_frame(det_data[sk_idx])

            headers = []
            if MODE == "binary-only":
                r = bin_index[sk_idx]
                if r is not None:
                    lab, prob = extract_pred_prob(r, bin_classes, prefer_rare=True)
                    headers.append(f"BINARY: {lab} ({prob:.2f})" if prob is not None else f"BINARY: {lab}")
            elif MODE == "two-stage":
                rb = bin_index[sk_idx]
                if rb is not None:
                    lab_b, prob_b = extract_pred_prob(rb, bin_classes, prefer_rare=True)
                    isfall = binary_is_fall(rb, bin_classes)
                    headers.append(f"BINARY: {lab_b} ({prob_b:.2f})" if prob_b is not None else f"BINARY: {lab_b}")
                    if not isfall and mul_index[sk_idx] is not None:
                        rm = mul_index[sk_idx]
                        lab_m, prob_m = extract_pred_prob(rm, mul_classes, prefer_rare=False)
                        headers.append(f"MULTI: {lab_m} ({prob_m:.2f})" if prob_m is not None else f"MULTI: {lab_m}")

            cache = {"kps_pp": kps_pp, "pose_boxes": pose_boxes, "det_objs": det_objs, "headers": headers}
            last_sk_idx = sk_idx

        # 疊加骨架/框
        if cache:
            if cache["pose_boxes"]:
                pseudo = [{"class_name": "person", "bbox": b, "confidence": 1.0} for b in cache["pose_boxes"]]
                draw_detections_mapped(canvas, pseudo, scale, x0, y0, color=COLOR_POSE_BOX, thickness=1)
            draw_skeletons_mapped(canvas, cache["kps_pp"], scale, x0, y0, point_color=COLOR_KP, line_color=COLOR_SKELETON, thickness=2)
            if cache["det_objs"]:
                draw_detections_mapped(canvas, cache["det_objs"], scale, x0, y0, color=COLOR_DET_BOX, thickness=1)

        # 左上資訊
        if cap is not None:
            cv2.putText(canvas, f"V {last_v_idx+1}/{total_vid}  {video_fps:.2f}fps | SK {sk_idx+1}/{total_json}  {SKELETON_FPS}fps",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)
        else:
            cv2.putText(canvas, f"SK {sk_idx+1}/{total_json}  {SKELETON_FPS}fps  (no video)",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)

        if DEBUG_DRAW_MAPPING and cap is not None:
            cv2.rectangle(canvas, (x0, y0), (x0+int(round(src_w*scale)), y0+int(round(src_h*scale))), COLOR_MAP_BORDER, 1)

        # 右上：推論標頭
        if cache and cache["headers"]:
            pad = 10; x_margin = 20; y_margin = 20
            header = " | ".join(cache["headers"])
            (tw, th), _ = cv2.getTextSize(header, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            x0h = CANVAS_WIDTH - tw - x_margin - 2*pad
            y0h = y_margin
            cv2.rectangle(canvas, (x0h, y0h), (x0h + tw + 2*pad, y0h + th + 2*pad), (30,30,30), -1)
            cv2.putText(canvas, header, (x0h + pad, y0h + th + pad//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow("Detect+Pose Player (RealTime Sync)", canvas)

        # 極短的 GUI 事件延遲：不阻塞播放（若硬體吃不消會丟幀）
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord(' '):
            paused = not paused
            if paused:
                # 暫停：停在畫面，直到再按空白或 q
                while True:
                    k2 = cv2.waitKey(10) & 0xFF
                    if k2 in (27, ord('q')):
                        cap and cap.release()
                        cv2.destroyAllWindows()
                        return
                    elif k2 == ord(' '):
                        # 續播：重設基準時間，避免 elapsed 積累
                        # 令當前 elapsed 對應到現在的 v_idx / sk_idx
                        now2 = time.perf_counter()
                        # 將 start_t 往後推，讓 elapsed 維持原值
                        start_t += (now2 - now)
                        break

    cap and cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
