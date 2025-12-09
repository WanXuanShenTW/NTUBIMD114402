#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
display.py (Updated)
- ✅ 動態注入解析度：Display 預先讀取影片解析度，並強制 Loader 使用此解析度，避免 Loader 找不到影片導致座標抖動。
- ✅ 支援 Initial Pause
- ✅ 支援 Target Tracking
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
# POSE_JSON_PATH     = "outputs/skeletons/test/pose/Meet and Split (46).json"
# DETECT_JSON_PATH   = "outputs/skeletons/test/detect/Meet and Split (46).json"
# POSE_JSON_PATH   = "outputs/skeletons/binary-backup/YOLO-pose/fall/Home_video (2)_back.json"
# DETECT_JSON_PATH = "outputs/skeletons/binary-backup/YOLO-detect/fall/Home_video (2)_back.json"
POSE_JSON_PATH   = "outputs/skeletons/multi/YOLO-pose/walk/Walking (70).json"
DETECT_JSON_PATH = "outputs/skeletons/multi/YOLO-detect/walk/Walking (70).json"
# POSE_JSON_PATH   = "outputs/skeletons/multi/YOLO-pose/lie/780251760.975590_back.json"
# DETECT_JSON_PATH = "outputs/skeletons/multi/YOLO-detect/lie/780251760.975590_back.json"
# POSE_JSON_PATH   = "outputs/skeletons/multi/YOLO-pose/sitstill/Sitting (20).json"
# DETECT_JSON_PATH = "outputs/skeletons/multi/YOLO-detect/sitstill/Sitting (20).json"
# POSE_JSON_PATH   = "outputs/skeletons/binary/YOLO-pose/non_fall/Office_video (18)_back_front.json"
# DETECT_JSON_PATH = "outputs/skeletons/binary/YOLO-detect/non_fall/Office_video (18)_back_front.json"

MODE             = "two-stage"    # "two-stage" | "binary-only" | "none"

ENABLE_VIDEO     = True
# VIDEO_PATH       = "medias/test/Meet and Split (46).mp4"
# VIDEO_PATH       = "medias/raw_n_edited/fall_all/Home_video (2)_back.mp4"
VIDEO_PATH       = "medias/train_video/multi/walk/Walking (70).mp4"
# VIDEO_PATH       = "medias/train_video/multi/lie/780251760.975590_back.mp4"
# VIDEO_PATH       = "medias/train_video/multi/sitstill/Sitting (20).mp4"
# VIDEO_PATH       = "medias/train_video/binary/non_fall/Office_video (18)_back_front.mp4"

VIDEO_FRAME_OFFSET = 0            # JSON 第 0 幀對應影片的第幾幀（對齊起點用）

# SKELETON_FPS     = 10
# SKELETON_FPS     = 12.5  
SKELETON_FPS     = 10             
JSON_FRAME_OFFSET = 0             

CANVAS_WIDTH     = 640
CANVAS_HEIGHT    = 480
SHOW_TOPK        = 3
DEBUG_DRAW_MAPPING = False

INITIAL_PAUSE    = True           

# 顏色
COLOR_SKELETON   = (255, 255, 0)
COLOR_KP         = (0, 255, 0)
COLOR_POSE_BOX   = (60, 180, 255)
COLOR_DET_BOX    = (100, 100, 255)
COLOR_MAP_BORDER = (0, 200, 255)

EXTRA_PY_PATHS   = []

FALL_THRESHOLD = 0.7  

# ==============================
# Helper: 讀取影片解析度
# ==============================
def get_video_resolution(video_path):
    """讀取影片解析度 (W, H)"""
    if not video_path or not os.path.exists(video_path):
        return 0, 0
    try:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            w = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            return w, h
    except:
        pass
    return 0, 0

# ==============================
# Tracker & IO (保持不變)
# ==============================
def _bbox_area(bbox):
    w = max(0, bbox[2] - bbox[0])
    h = max(0, bbox[3] - bbox[1])
    return w * h

def _iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    b1_area = _bbox_area(box1); b2_area = _bbox_area(box2)
    union_area = b1_area + b2_area - inter_area
    return inter_area / (union_area + 1e-6)

class TargetTracker:
    def __init__(self, kp_conf_th=0.5, min_kps=5, iou_track_th=0.3):
        self.last_bbox = None
        self.kp_conf_th = kp_conf_th
        self.min_kps = min_kps
        self.iou_track_th = iou_track_th

    def _count_valid_kps(self, kps):
        cnt = 0
        for k in kps:
            conf = 1.0
            if isinstance(k, dict):
                conf = float(k.get('conf', k.get('confidence', 1.0)))
            elif isinstance(k, (list, tuple)) and len(k) > 2:
                conf = float(k[2])
            if conf >= self.kp_conf_th:
                cnt += 1
        return cnt

    def update(self, persons: List[dict]) -> Optional[dict]:
        if not persons:
            self.last_bbox = None
            return None
        if self.last_bbox is not None:
            best_p = None
            best_iou = -1.0
            for p in persons:
                bbox = p.get('bbox') or p.get('box')
                if not bbox: continue
                if len(bbox) == 4:
                    val = _iou(self.last_bbox, bbox)
                    if val > best_iou:
                        best_iou = val
                        best_p = p
            if best_iou >= self.iou_track_th and best_p:
                self.last_bbox = best_p.get('bbox') or best_p.get('box')
                return best_p
            else:
                self.last_bbox = None
        
        candidates = []
        for p in persons:
            kps = p.get('keypoints', [])
            bbox = p.get('bbox') or p.get('box')
            if not bbox: continue
            vk = self._count_valid_kps(kps)
            if vk >= self.min_kps:
                area = _bbox_area(bbox)
                candidates.append((area, p))
        
        if not candidates: return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        selected = candidates[0][1]
        self.last_bbox = selected.get('bbox') or selected.get('box')
        return selected

YOLO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

def _json_to_list(data):
    if isinstance(data, list): return data
    if isinstance(data, dict):
        for k in ('frames', 'data', 'records', 'annotations', 'items', 'results'):
            v = data.get(k)
            if isinstance(v, list): return v
        try:
            items = sorted([(int(k), v) for k, v in data.items() if str(k).isdigit()], key=lambda x: x[0])
            return [v for _, v in items] if items else [data]
        except Exception: return [data]
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

def parse_pose_and_track(frame_data, tracker: TargetTracker):
    persons = []
    if isinstance(frame_data, dict):
        if 'persons' in frame_data: persons = frame_data['persons']
        elif 'bodies' in frame_data: persons = frame_data['bodies']
        elif 'boxes' in frame_data and 'keypoints' in frame_data:
            boxes = frame_data['boxes']
            kps = frame_data['keypoints']
            persons = [{'bbox': boxes[i], 'keypoints': kps[i]} for i in range(min(len(boxes), len(kps)))]
        elif 'keypoints' in frame_data:
            kps = frame_data['keypoints']
            persons = [{'keypoints': k, 'bbox': None} for k in kps]
    elif isinstance(frame_data, list):
        persons = frame_data

    valid_persons = []
    for p in persons:
        if isinstance(p, dict): valid_persons.append(p)
        elif isinstance(p, list): valid_persons.append({'keypoints': p, 'bbox': None})
    
    target_person = tracker.update(valid_persons)
    target_res = ([], [])
    ignored_res = []

    for p in valid_persons:
        bbox = p.get('bbox') or p.get('box')
        if bbox: bbox = [float(x) for x in bbox]
        raw_kps = p.get('keypoints', [])
        pts = []
        if raw_kps:
            for k in raw_kps[:17]: 
                if isinstance(k, dict): pts.append((float(k.get('x',0)), float(k.get('y',0))))
                elif isinstance(k, (list, tuple)) and len(k) >= 2: pts.append((float(k[0]), float(k[1])))
                else: pts.append((0.0, 0.0))
        else:
            pts = [(0.0, 0.0)] * 17
        
        data_packet = ([pts], [bbox] if bbox else [])
        if p is target_person: target_res = data_packet
        else: ignored_res.append(data_packet)
            
    return target_res, ignored_res

def parse_detect_frame(frame: dict) -> List[dict]:
    out = []
    if not isinstance(frame, dict): return out
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
# Letterbox
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
# Draw
# ==============================
def draw_skeletons_mapped(canvas, keypoints_pp, scale, x0, y0, point_color=(0,255,0), line_color=(255,255,0), thickness=2):
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
# Loader Injection (核心修改)
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
        
        # === [核心修改] 動態注入：強制替換 try_get_video_resolution ===
        # 因為 display.py 知道正確的影片路徑，我們直接告訴 Loader 解析度是多少
        if hasattr(m, "try_get_video_resolution"):
            # 1. 取得本次影片的解析度
            w, h = get_video_resolution(VIDEO_PATH)
            
            # 2. 定義一個 Mock 函式，永遠回傳這個正確的值
            def mock_get_res(_ignored_path):
                print(f"[Display Injection] Forcing resolution: {int(w)}x{int(h)} for {module_name}")
                return w, h
            
            # 3. 替換掉 Loader 裡的函式
            m.try_get_video_resolution = mock_get_res
        # ============================================================

        if hasattr(m, "run_on_json"):
            return m.run_on_json(pose_json, detect_json)
    except Exception as e:
        print(f"[Warn] 無法執行 {module_name}.run_on_json(): {e}")
        import traceback
        traceback.print_exc()
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
            return probs[fall_idx] >= FALL_THRESHOLD 
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
# 主流程
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

    tracker = TargetTracker(kp_conf_th=0.5, min_kps=5, iou_track_th=0.3)
    
    start_t = None
    last_v_idx = -1
    last_sk_idx = -1

    if INITIAL_PAUSE:
        canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)
        if cap is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(VIDEO_FRAME_OFFSET))
            ok, frame = cap.read()
            if ok and frame is not None:
                new_w = int(round(src_w * scale)); new_h = int(round(src_h * scale))
                resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                canvas[y0:y0+new_h, x0:x0+new_w] = resized
        sk0 = int(JSON_FRAME_OFFSET)
        kps_pp, pose_boxes = parse_pose_frame(pose_data[min(max(sk0,0), total_json-1)])
        det_objs = parse_detect_frame(det_data[min(max(sk0,0), total_json-1)])
        if det_objs:
            det_to_draw = [d for d in det_objs if str(d.get("class_name","")).lower() != "person"]
            if det_to_draw:
                draw_detections_mapped(canvas, det_to_draw, scale, x0, y0, color=COLOR_DET_BOX, thickness=1)
        draw_skeletons_mapped(canvas, kps_pp, scale, x0, y0, point_color=COLOR_KP, line_color=COLOR_SKELETON, thickness=2)
        if det_objs:
            draw_detections_mapped(canvas, det_objs, scale, x0, y0, color=COLOR_DET_BOX, thickness=1)

        msg = "Paused — Press SPACE or CLICK to start"
        (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        x_center = (CANVAS_WIDTH - tw) // 2
        y_center = (CANVAS_HEIGHT) // 2
        cv2.rectangle(canvas, (x_center-12, y_center- th - 12), (x_center + tw + 12, y_center + 12), (30,30,30), -1)
        cv2.putText(canvas, msg, (x_center, y_center),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

        started = [False]
        def _on_mouse(event, x, y, flags, userdata):
            if event == cv2.EVENT_LBUTTONDOWN:
                started[0] = True

        cv2.namedWindow("Detect+Pose Player (RealTime Sync)", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("Detect+Pose Player (RealTime Sync)", _on_mouse)
        while True:
            cv2.imshow("Detect+Pose Player (RealTime Sync)", canvas)
            key = cv2.waitKey(10) & 0xFF
            if key in (27, ord('q')):
                if cap is not None:
                    cap.release()
                cv2.destroyAllWindows()
                return
            if key == ord(' ') or started[0]:
                break

        if cap is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(VIDEO_FRAME_OFFSET))
        start_t = time.perf_counter()
        last_v_idx = int(VIDEO_FRAME_OFFSET) - 1
        last_sk_idx = int(JSON_FRAME_OFFSET) - 1
    else:
        start_t = time.perf_counter()
        last_v_idx = int(VIDEO_FRAME_OFFSET) - 1
        last_sk_idx = int(JSON_FRAME_OFFSET) - 1

    cache = {}
    while True:
        now = time.perf_counter()
        elapsed = now - start_t

        if ENABLE_VIDEO and (cap is not None):
            target_v_idx = int(elapsed * video_fps) + int(VIDEO_FRAME_OFFSET)
            if target_v_idx >= total_vid:
                break 
            if target_v_idx != last_v_idx:
                if target_v_idx == last_v_idx + 1:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_v_idx)
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break
                last_v_idx = target_v_idx

            canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)
            new_w = int(round(src_w * scale)); new_h = int(round(src_h * scale))
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            canvas[y0:y0+new_h, x0:x0+new_w] = resized
        else:
            canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)

        sk_idx = int(JSON_FRAME_OFFSET) + int(elapsed * SKELETON_FPS)
        if sk_idx >= total_json:
            sk_idx = total_json - 1

        if sk_idx != last_sk_idx:
            target_bundle, ignored_list = parse_pose_and_track(pose_data[sk_idx], tracker)
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

            cache = {
                "target": target_bundle, 
                "ignored": ignored_list, 
                "det_objs": det_objs, 
                "headers": headers
            }
            last_sk_idx = sk_idx

        if cache:
            for (ign_kps, ign_boxes) in cache.get("ignored", []):
                pseudo = [{"class_name": "ignore", "bbox": b, "confidence": 0.5} for b in ign_boxes]
                draw_detections_mapped(canvas, pseudo, scale, x0, y0, color=(128, 128, 128), thickness=1)
                draw_skeletons_mapped(canvas, ign_kps, scale, x0, y0, point_color=(128,128,128), line_color=(100,100,100), thickness=1)

            tgt_kps, tgt_boxes = cache.get("target", ([], []))
            if tgt_boxes:
                pseudo = [{"class_name": "TARGET", "bbox": b, "confidence": 1.0} for b in tgt_boxes]
                draw_detections_mapped(canvas, pseudo, scale, x0, y0, color=(0, 255, 255), thickness=2)
                draw_skeletons_mapped(canvas, tgt_kps, scale, x0, y0, point_color=COLOR_KP, line_color=COLOR_SKELETON, thickness=2)

            det_objs = cache.get("det_objs", [])
            if det_objs:
                det_to_draw = [d for d in det_objs if str(d.get("class_name","")).lower() != "person"]
                if det_to_draw:
                    draw_detections_mapped(canvas, det_to_draw, scale, x0, y0, color=COLOR_DET_BOX, thickness=1)

        if ENABLE_VIDEO and (cap is not None):
            cv2.putText(canvas, f"V {last_v_idx+1}/{total_vid}  {video_fps:.2f}fps | SK {sk_idx+1}/{total_json}  {SKELETON_FPS}fps",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)
        else:
            cv2.putText(canvas, f"SK {sk_idx+1}/{total_json}  {SKELETON_FPS}fps  (no video)",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)

        if DEBUG_DRAW_MAPPING and ENABLE_VIDEO and (cap is not None):
            cv2.rectangle(canvas, (x0, y0), (x0+int(round(src_w*scale)), y0+int(round(src_h*scale))), COLOR_MAP_BORDER, 1)

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

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord(' '):
            while True:
                k2 = cv2.waitKey(10) & 0xFF
                if k2 in (27, ord('q')):
                    if cap is not None:
                        cap.release()
                    cv2.destroyAllWindows()
                    return
                elif k2 == ord(' '):
                    now2 = time.perf_counter()
                    paused_dt = now2 - now
                    start_t += paused_dt
                    break

    if ENABLE_VIDEO and (cap is not None):
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()