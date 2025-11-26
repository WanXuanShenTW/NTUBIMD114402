#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
display.py  (real-time video + mapped skeleton; initial pause support)

- 影片以「原生 FPS + 牆鐘同步」播放（與一般播放器一致；必要時自動丟幀）。
- 骨架/偵測依自訂 FPS 依牆鐘時間前進（與影片互不干擾）。
- 座標以 letterbox 對映，避免變形與偏移。
- 新增：INITIAL_PAUSE，起始先停住，按「Space」或「滑鼠左鍵」才開始。
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
# POSE_JSON_PATH   = "outputs/skeletons/binary/YOLO-pose/non_fall/Office_video (18)_back_front.json"
# DETECT_JSON_PATH = "outputs/skeletons/binary/YOLO-detect/non_fall/Office_video (18)_back_front.json"
# POSE_JSON_PATH   = "outputs/skeletons/binary-backup/YOLO-pose/fall/Home_video (2)_back.json"
# DETECT_JSON_PATH = "outputs/skeletons/binary-backup/YOLO-detect/fall/Home_video (2)_back.json"
POSE_JSON_PATH   = "outputs/skeletons/multi/YOLO-pose/lie/780251760.975590_back.json"
DETECT_JSON_PATH = "outputs/skeletons/multi/YOLO-detect/lie/780251760.975590_back.json"
# POSE_JSON_PATH   = "outputs/skeletons/multi/YOLO-pose/sitstill/Sitting (20).json"
# DETECT_JSON_PATH = "outputs/skeletons/multi/YOLO-detect/sitstill/Sitting (20).json"
# POSE_JSON_PATH   = "outputs/skeletons/multi/YOLO-pose/walk/Walking (70).json"
# DETECT_JSON_PATH = "outputs/skeletons/multi/YOLO-detect/walk/Walking (70).json"

MODE             = "two-stage"    # "two-stage" | "binary-only" | "none"

ENABLE_VIDEO     = True
# VIDEO_PATH       = "medias/test/Meet and Split (46).mp4"
# VIDEO_PATH       = "medias/train_video/binary/non_fall/Office_video (18)_back_front.mp4"
# VIDEO_PATH       = "medias/raw_n_edited/fall_all/Home_video (2)_back.mp4"
VIDEO_PATH       = "medias/train_video/multi/lie/780251760.975590_back.mp4"
# VIDEO_PATH       = "medias/train_video/multi/sitstill/Sitting (20).mp4"
# VIDEO_PATH       = "medias/train_video/multi/walk/Walking (70).mp4"
VIDEO_FRAME_OFFSET = 0            # JSON 第 0 幀對應影片的第幾幀（對齊起點用）

SKELETON_FPS     = 10             # 骨架的節奏（與影片無關）
JSON_FRAME_OFFSET = 0             # JSON 幀起始偏移

CANVAS_WIDTH     = 640
CANVAS_HEIGHT    = 480
SHOW_TOPK        = 3
DEBUG_DRAW_MAPPING = False

INITIAL_PAUSE    = True           # ★ 新增：起始先暫停，按 Space/滑鼠左鍵 才開始

# 顏色
COLOR_SKELETON   = (255, 255, 0)
COLOR_KP         = (0, 255, 0)
COLOR_POSE_BOX   = (60, 180, 255)
COLOR_DET_BOX    = (100, 100, 255)
COLOR_MAP_BORDER = (0, 200, 255)

EXTRA_PY_PATHS   = []

# Add a global variable for the threshold
FALL_THRESHOLD = 0.7  # Default threshold for fall detection

# ==============================
# 🆕 新增：單人目標追蹤邏輯 (Target Tracker)
# ==============================
def _bbox_area(bbox):
    # bbox: [x1, y1, x2, y2]
    w = max(0, bbox[2] - bbox[0])
    h = max(0, bbox[3] - bbox[1])
    return w * h

def _iou(box1, box2):
    # box: [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    b1_area = _bbox_area(box1)
    b2_area = _bbox_area(box2)
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
            # 兼容 dict 或 list 格式
            conf = 1.0
            if isinstance(k, dict):
                conf = float(k.get('conf', k.get('confidence', 1.0)))
            elif isinstance(k, (list, tuple)) and len(k) > 2:
                conf = float(k[2])
            if conf >= self.kp_conf_th:
                cnt += 1
        return cnt

    def update(self, persons: List[dict]) -> Optional[dict]:
        """
        輸入: 本幀所有 person list
        輸出: 被鎖定的那個 person dict，若無則回傳 None
        """
        if not persons:
            self.last_bbox = None
            return None

        # 1. Tracking (嘗試追蹤上一幀的人)
        if self.last_bbox is not None:
            best_p = None
            best_iou = -1.0
            for p in persons:
                bbox = p.get('bbox') or p.get('box')
                if not bbox: continue
                # 確保 bbox 格式為 list [x1,y1,x2,y2]
                if len(bbox) == 4:
                    val = _iou(self.last_bbox, bbox)
                    if val > best_iou:
                        best_iou = val
                        best_p = p
            
            # 若 IoU 足夠高，認定為同一人，更新 bbox 並回傳
            if best_iou >= self.iou_track_th and best_p:
                self.last_bbox = best_p.get('bbox') or best_p.get('box')
                return best_p
            else:
                # 追蹤失敗 (離開畫面或被遮擋)，清除記錄，進入重新選擇
                self.last_bbox = None
        
        # 2. Selection (重新鎖定策略：骨架完整度優先，其次選最近/最大的)
        candidates = []
        for p in persons:
            kps = p.get('keypoints', [])
            bbox = p.get('bbox') or p.get('box')
            if not bbox: continue
            
            vk = self._count_valid_kps(kps)
            # 必須滿足最小骨架點數
            if vk >= self.min_kps:
                area = _bbox_area(bbox)
                candidates.append((area, p))
        
        if not candidates:
            return None
            
        # 依面積排序 (由大到小 -> 離鏡頭最近)
        candidates.sort(key=lambda x: x[0], reverse=True)
        selected = candidates[0][1]
        self.last_bbox = selected.get('bbox') or selected.get('box')
        return selected

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

def parse_pose_and_track(frame_data, tracker: TargetTracker):
    """
    解析 JSON 並區分：
    - target_data: (kps_list, boxes_list) -> 被鎖定的人
    - ignored_data: list of (kps_list, boxes_list) -> 其他被忽略的人
    """
    persons = []
    
    # === 1. 資料正規化：將不同格式統一轉為 List of Dicts ===
    if isinstance(frame_data, dict):
        # A. 優先找標準格式 (List of Dicts)
        if 'persons' in frame_data:
            persons = frame_data['persons']
        elif 'bodies' in frame_data:
            persons = frame_data['bodies']
            
        # B. 處理 YOLO 分離格式 (Structure of Arrays) -> 這是報錯的主因
        # 如果 JSON 是 {"boxes": [...], "keypoints": [...]}
        elif 'boxes' in frame_data and 'keypoints' in frame_data:
            boxes = frame_data['boxes']
            kps = frame_data['keypoints']
            # 手動將它們合併成 person dict
            persons = []
            for i in range(min(len(boxes), len(kps))):
                persons.append({
                    'bbox': boxes[i],
                    'keypoints': kps[i]
                })
        
        # C. 只有 keypoints 的情況
        elif 'keypoints' in frame_data:
            kps = frame_data['keypoints']
            # 將每個 keypoint list 包裝成 dict
            persons = [{'keypoints': k, 'bbox': None} for k in kps]

    elif isinstance(frame_data, list):
        # 假設 list 裡面就是 person dicts
        persons = frame_data

    # === 2. 防呆檢查 ===
    # 確保進入 Tracker 的每個元素真的是 Dict，避免 list.get() 錯誤
    valid_persons = []
    for p in persons:
        if isinstance(p, dict):
            valid_persons.append(p)
        elif isinstance(p, list):
            # 若發現還有單純的 list (例如純座標)，幫它包一層
            valid_persons.append({'keypoints': p, 'bbox': None})
    
    # === 3. 呼叫追蹤器 ===
    target_person = tracker.update(valid_persons)
    
    target_res = ([], []) # kps, boxes
    ignored_res = []      # list of (kps, boxes)

    for p in valid_persons:
        # 提取 bbox
        bbox = p.get('bbox') or p.get('box')
        if bbox:
            bbox = [float(x) for x in bbox] # 確保是 float list
        
        # 提取 keypoints (統一轉成 list of (x,y))
        raw_kps = p.get('keypoints', [])
        pts = []
        # 防呆：有時 raw_kps 可能是 None
        if raw_kps:
            for k in raw_kps[:17]: 
                # 支援 {"x":, "y":} 或 [x, y, c]
                if isinstance(k, dict):
                    pts.append((float(k.get('x',0)), float(k.get('y',0))))
                elif isinstance(k, (list, tuple)) and len(k) >= 2:
                    pts.append((float(k[0]), float(k[1])))
                else:
                    pts.append((0.0, 0.0))
        else:
            # 補 17 個 0
            pts = [(0.0, 0.0)] * 17
        
        # 封裝結果
        data_packet = ([pts], [bbox] if bbox else [])
        
        # 分類
        if p is target_person:
            target_res = data_packet
        else:
            ignored_res.append(data_packet)
            
    return target_res, ignored_res

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
    # Modify the binary_is_fall function to use the global threshold
    probs = r.get("probs", None)
    if isinstance(class_names, list) and isinstance(probs, list) and len(class_names) == len(probs):
        try:
            import numpy as _np
            fall_idx = class_names.index("fall")
            return probs[fall_idx] >= FALL_THRESHOLD  # Use the global threshold
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
# 主流程：影片 wall-clock 同步 + 骨架自訂 FPS + 初始暫停
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

    # 🆕 初始化追蹤器
    # min_kps=5: 至少要有5個點才納入候選
    # iou_track_th=0.3: 前後幀重疊 30% 以上才持續追蹤
    tracker = TargetTracker(kp_conf_th=0.5, min_kps=5, iou_track_th=0.3)
    
    # ------- 初始暫停：顯示起始畫面，按 Space/滑鼠左鍵才開始 -------
    start_t = None
    last_v_idx = -1
    last_sk_idx = -1

    if INITIAL_PAUSE:
        # 準備畫布
        canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)
        # 顯示影片起始影格
        if cap is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(VIDEO_FRAME_OFFSET))
            ok, frame = cap.read()
            if ok and frame is not None:
                new_w = int(round(src_w * scale)); new_h = int(round(src_h * scale))
                resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                canvas[y0:y0+new_h, x0:x0+new_w] = resized
        # 顯示骨架/方框起始幀
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

        # 疊加提示文字
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
            if key in (27, ord('q')):  # 允許直接離開
                if cap is not None:
                    cap.release()
                cv2.destroyAllWindows()
                return
            if key == ord(' ') or started[0]:
                break

        # 釋放起始幀，重設影片指標到 offset，並準備 wall-clock 基準
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
    # ------- 播放主迴圈 -------
    while True:
        now = time.perf_counter()
        elapsed = now - start_t

        # 影片索引（牆鐘同步）
        if ENABLE_VIDEO and (cap is not None):
            target_v_idx = int(elapsed * video_fps) + int(VIDEO_FRAME_OFFSET)
            if target_v_idx >= total_vid:
                break  # 播放完
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

        # 骨架索引（牆鐘同步，自訂 FPS）
        sk_idx = int(JSON_FRAME_OFFSET) + int(elapsed * SKELETON_FPS)
        if sk_idx >= total_json:
            sk_idx = total_json - 1

        if sk_idx != last_sk_idx:
            # 🆕 改用追蹤邏輯解析
            # target_bundle: (kps_list, boxes_list) for target
            # ignored_list: list of (kps_list, boxes_list) for others
            target_bundle, ignored_list = parse_pose_and_track(pose_data[sk_idx], tracker)
            det_objs = parse_detect_frame(det_data[sk_idx])

            # 處理文字標頭 (維持原樣)
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

            # 更新 Cache 結構
            cache = {
                "target": target_bundle, 
                "ignored": ignored_list, 
                "det_objs": det_objs, 
                "headers": headers
            }
            last_sk_idx = sk_idx

        # 疊加骨架/框
        if cache:
            # 1. 先畫「被忽略」的人 (灰色/暗色) - 代表這些人不會被送去判斷
            for (ign_kps, ign_boxes) in cache.get("ignored", []):
                # 畫框 (灰色)
                pseudo = [{"class_name": "ignore", "bbox": b, "confidence": 0.5} for b in ign_boxes]
                draw_detections_mapped(canvas, pseudo, scale, x0, y0, color=(128, 128, 128), thickness=1)
                # 畫骨架 (灰色)
                draw_skeletons_mapped(canvas, ign_kps, scale, x0, y0, point_color=(128,128,128), line_color=(100,100,100), thickness=1)

            # 2. 再畫「被鎖定」的目標 (亮色/綠色) - 代表這是傳給後端的對象
            tgt_kps, tgt_boxes = cache.get("target", ([], []))
            if tgt_boxes:
                # 畫框 (亮青色)
                pseudo = [{"class_name": "TARGET", "bbox": b, "confidence": 1.0} for b in tgt_boxes]
                draw_detections_mapped(canvas, pseudo, scale, x0, y0, color=(0, 255, 255), thickness=2)
                # 畫骨架 (標準亮色: 點綠/線黃)
                draw_skeletons_mapped(canvas, tgt_kps, scale, x0, y0, point_color=COLOR_KP, line_color=COLOR_SKELETON, thickness=2)

            # 3. 畫環境物件 (Detect JSON 中的物品，排除 person)
            det_objs = cache.get("det_objs", [])
            if det_objs:
                det_to_draw = [d for d in det_objs if str(d.get("class_name","")).lower() != "person"]
                if det_to_draw:
                    draw_detections_mapped(canvas, det_to_draw, scale, x0, y0, color=COLOR_DET_BOX, thickness=1)

        # 左上資訊
        if ENABLE_VIDEO and (cap is not None):
            cv2.putText(canvas, f"V {last_v_idx+1}/{total_vid}  {video_fps:.2f}fps | SK {sk_idx+1}/{total_json}  {SKELETON_FPS}fps",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)
        else:
            cv2.putText(canvas, f"SK {sk_idx+1}/{total_json}  {SKELETON_FPS}fps  (no video)",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)

        if DEBUG_DRAW_MAPPING and ENABLE_VIDEO and (cap is not None):
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

        # 不阻塞 GUI（1ms），保持 wall-clock 對齊（不足時丟幀）
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord(' '):
            # 暫停：停住到再按空白或 q；續播時重設起始時間保持對齊
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
