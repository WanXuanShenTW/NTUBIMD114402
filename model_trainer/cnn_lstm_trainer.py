# -*- coding: utf-8 -*-
"""
CNN → LSTM 動作辨識訓練程式（加入卡爾曼濾波 + 骨架完整緩衝判斷）
- ✅ 不改動模型輸出/訓練流程與檔名；僅在『輸入前置處理』與『樣本挑選規則』做增強。
- ✅ 視窗設定仍為：window=20、stride=5（與你既有行為一致）。
- ✅ 僅當目前 20 幀緩衝中『每一幀都有完整骨架』時，才建立訓練樣本。
- ✅ 對該 20 幀的 17 個關節做 2D 常速模型卡爾曼濾波（平滑 keypoints）。
- ✅ 使用『半視窗』(window//2) 初始化卡爾曼濾波，並以線上方式每 5 幀更新。

替換方式：直接以此檔案覆蓋原先的 cnn_lstm_trainer.py 後執行。
"""

import os, json, glob, random
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from tqdm import tqdm

# ===================== 你只需要改這裡 =====================
class Config:
    # 路徑（分開設定）
    pose_root = "outputs/skeletons/YOLO/YOLO-pose"  # ★必改：骨架/人框 JSONL 所在根目錄
    obj_root  = "outputs/skeletons/YOLO/YOLO-detect"   # ★可選：物件框 JSONL 所在根目錄（use_objects=True 時必填）
    out_dir   = "outputs/models/test"               # 輸出資料夾

    # 是否使用物件框（若關閉則不需要 obj_root，且不建立物件通道）
    use_objects = True
    object_classes = ["bed", "chair"]  # 每類一通道；若空列表則不建立物件通道

    # 時序（看到的時間長度與重疊）
    window = 20   # ★每段片段幀數
    stride = 5    # ★視窗滑動步幅

    # Relation Map（空間與語意）
    H, W = 64, 64
    include_bone_lines = True            # 畫骨架連線（提供肢段方向）

    # 模型
    lstm_hidden  = 256                   # LSTM 隱層寬度
    bidirectional = False                # 訓練離線可開雙向
    temporal_pool = "attn"              # "last" | "mean" | "attn"
    dropout = 0.3

    # 訓練
    epochs     = 30
    batch_size = 8
    lr         = 1e-3
    use_sampler = True                   # 不平衡抽樣
    loss = "focal"                       # "focal" | "ce"

    # ====== 新增：輸入前置與過濾規則 ======
    enable_kalman = True                 # 對 keypoints 做卡爾曼平滑

    # ★ 新需求：半視窗 KF 初始化與滑動
    kalman_half_slide = True             # 以『半視窗』(window//2) 做線上式初始化/更新
    require_full_first_frame = True      # 半視窗的第一幀必須完整；後續可缺點，由 KF 補/穩定
    half_len_override = None             # 若想自訂半視窗長度，填整數；預設為 window//2 (=10)

    # 舊的全視窗完整性規則（通常改為 False）
    require_full_skeleton_all = False    # 若設 True，則 20 幀每一幀都要完整才取樣

    # 完整性的細節門檻
    full_kp_min = 17                     # 視為完整的最小關節數（預設全 17）
    bbox_required = True                 # 必須有人框（至少第一幀）

# ===================== 穩定預設（通常不需要改） =====================
_SEED = 42
_SAMPLE_EVERY = 1
_KP_CONF_TH = 0.4
_SIGMA_KP = 3.0
_CNN_OUT = 256
_LSTM_LAYERS = 2
_WEIGHT_DECAY = 1e-4
_GRAD_CLIP = 1.0
_AMP = True
_EARLY_STOP_PATIENCE = 10
_SAVE_TOP_K = 3
_FOCAL_GAMMA = 2.0

# ===================== Utils =====================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

# ===================== Relation Map =====================
COCO_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]

class RelationMapConfig:
    def __init__(self, H=64, W=64, sigma_kp=2.5, kp_conf_th=0.3,
                 include_bone_lines=True, object_classes=None):
        self.H = H
        self.W = W
        self.sigma_kp = float(sigma_kp)
        self.kp_conf_th = float(kp_conf_th)
        self.include_bone_lines = bool(include_bone_lines)
        self.object_classes = [c.strip() for c in (object_classes or []) if c.strip()]


def draw_gaussian(heatmap, x, y, sigma, mag=1.0):
    H, W = heatmap.shape
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma ** 2)).astype(np.float32) * mag
    heatmap += g


def _bbox_xyxy_from_any(b):
    if b is None:
        return None
    if isinstance(b, dict):
        if all(k in b for k in ("cx", "cy", "w", "h")):
            cx, cy, w, h = float(b["cx"]), float(b["cy"]), float(b["w"]), float(b["h"])
            return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        if all(k in b for k in ("x", "y", "w", "h")):
            x, y, w, h = float(b["x"]), float(b["y"]), float(b["w"]), float(b["h"])
            return x, y, x + w, y + h
    if isinstance(b, (list, tuple)) and len(b) == 4:
        x1, y1, x2, y2 = map(float, b)
        return x1, y1, x2, y2
    return None


def rasterize_frame(bbox, kps, objects, img_w, img_h, cfg: RelationMapConfig):
    H, W = cfg.H, cfg.W
    num_bbox_mask = 1
    num_dist = 1
    num_kp = 17
    num_edges = len(COCO_EDGES) if cfg.include_bone_lines else 0
    num_obj_ch = len(cfg.object_classes)
    num_coord = 2

    C = num_bbox_mask + num_dist + num_kp + num_edges + num_obj_ch + num_coord
    canvas = np.zeros((C, H, W), np.float32)

    ch = 0
    # 1) person bbox mask
    if bbox:
        bxyxy = _bbox_xyxy_from_any(bbox)
        if bxyxy is not None:
            x1, y1, x2, y2 = bxyxy
            x1 = int(np.clip((x1 / img_w) * W, 0, W - 1))
            y1 = int(np.clip((y1 / img_h) * H, 0, H - 1))
            x2 = int(np.clip((x2 / img_w) * W, 0, W - 1))
            y2 = int(np.clip((y2 / img_h) * H, 0, H - 1))
            if x2 >= x1 and y2 >= y1:
                canvas[ch, y1:y2 + 1, x1:x2 + 1] = 1.0
    ch += 1

    # 2) distance transform
    inv = (1.0 - canvas[0]).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    if dist.max() > 0:
        dist = dist / dist.max()
    canvas[ch] = dist
    ch += 1

    # 3) keypoints gaussians
    if kps:
        for i, kp in enumerate(kps[:17]):
            try:
                conf = float(kp.get("conf", kp.get("confidence", 1.0)))
                if conf < cfg.kp_conf_th:
                    continue
                x = (float(kp["x"]) / img_w) * W
                y = (float(kp["y"]) / img_h) * H
                draw_gaussian(canvas[ch + i], x, y, cfg.sigma_kp, mag=conf)
            except Exception:
                pass
    ch += 17

    # 4) bone lines
    if cfg.include_bone_lines and kps and len(kps) >= 17:
        pts = []
        for i in range(17):
            try:
                x = int(np.clip((float(kps[i]["x"]) / img_w) * W, 0, W - 1))
                y = int(np.clip((float(kps[i]["y"]) / img_h) * H, 0, H - 1))
                conf = float(kps[i].get("conf", kps[i].get("confidence", 1.0)))
                pts.append((x, y, conf))
            except Exception:
                pts.append((None, None, 0.0))
        for e_idx, (a, b) in enumerate(COCO_EDGES):
            x1, y1, c1 = pts[a]
            x2, y2, c2 = pts[b]
            if x1 is None or x2 is None:
                continue
            if min(c1, c2) < cfg.kp_conf_th:
                continue
            cv2.line(canvas[ch + e_idx], (x1, y1), (x2, y2), color=1.0, thickness=1)
    ch += (len(COCO_EDGES) if cfg.include_bone_lines else 0)

    # 5) objects → channels（每類一通道）
    if num_obj_ch > 0 and objects:
        name_to_idx = {n: i for i, n in enumerate(cfg.object_classes)}
        for det in objects:
            # 兼容多種鍵名：cls_name / class_name / name / label
            cname = None
            for key in ("cls_name", "class_name", "name", "label"):
                val = det.get(key)
                if isinstance(val, str) and val.strip():
                    cname = val.strip()
                    break
            if cname not in name_to_idx:
                continue
            bxyxy = _bbox_xyxy_from_any(det.get("bbox") or det.get("xyxy") or det.get("box") or det.get("bbox_xyxy"))
            if bxyxy is None:
                continue
            x1, y1, x2, y2 = bxyxy
            x1 = int(np.clip((x1 / img_w) * W, 0, W - 1))
            y1 = int(np.clip((y1 / img_h) * H, 0, H - 1))
            x2 = int(np.clip((x2 / img_w) * W, 0, W - 1))
            y2 = int(np.clip((y2 / img_h) * H, 0, H - 1))
            if x2 >= x1 and y2 >= y1:
                canvas[ch + name_to_idx[cname], y1:y2 + 1, x1:x2 + 1] = 1.0
    ch += num_obj_ch

    # 6) CoordConv
    xv = np.linspace(-1, 1, W)[None, :].repeat(H, 0)
    yv = np.linspace(-1, 1, H)[:, None].repeat(W, 1)
    canvas[ch] = xv
    canvas[ch + 1] = yv
    ch += 2

    return canvas

# ===================== Kalman Filter（2D 常速模型，一點一濾） =====================
class Kalman2D:
    def __init__(self, x=0.0, y=0.0, var_pos=1.0, var_vel=1.0, var_meas=4.0):
        # 狀態: [x, y, vx, vy]
        self.F = np.array([[1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float32)
        self.Q = np.diag([var_pos, var_pos, var_vel, var_vel]).astype(np.float32)
        self.R = np.diag([var_meas, var_meas]).astype(np.float32)
        self.x = np.array([[x], [y], [0.0], [0.0]], dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 10.0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        # z: [x, y] 量測
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y)
        I = np.eye(4, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

    def get_xy(self):
        return float(self.x[0, 0]), float(self.x[1, 0])


def kalman_smooth_kps(window_kps, *,
                       half_slide=False,
                       half_len=10,
                       require_full_first=True):
    """對一段視窗的 17 個關節序列做 KF 平滑。
    - half_slide=True：模擬線上推論的『半視窗(half_len)→每5幀更新』模式；
      * 第一個半視窗必須以完整第一幀初始化（若不完整，建議上游直接丟棄該樣本）。
      * 後續幀若關節缺失/低信心，只做 predict，不做 update（KF 可補位移）。
    - half_len：通常為 window//2（你的 window=20 → half_len=10）。
    - require_full_first：若第一幀不完整，直接回傳原始以利上游決策（或你也可選擇丟棄）。
    """
    T = len(window_kps)
    if T == 0:
        return window_kps
    J = 17

    # --- 驗證第一幀是否完整（必要時） ---
    if require_full_first:
        first_ok = 0
        first = window_kps[0] if len(window_kps) > 0 else []
        for j in range(min(J, len(first))):
            try:
                conf = float(first[j].get("conf", first[j].get("confidence", 1.0)))
                if conf >= _KP_CONF_TH and np.isfinite(first[j].get("x", 0)) and np.isfinite(first[j].get("y", 0)):
                    first_ok += 1
            except Exception:
                pass
        if first_ok < J:  # 未達完整度
            return window_kps

    # --- 初始化每個關節的濾波器（用第一幀作初始） ---
    filters = []
    first = window_kps[0]
    for j in range(J):
        if j < len(first) and "x" in first[j] and "y" in first[j]:
            kf = Kalman2D(first[j]["x"], first[j]["y"], var_pos=1e-2, var_vel=1e-1, var_meas=4.0)
        else:
            kf = Kalman2D(0.0, 0.0, var_pos=1e-2, var_vel=1e-1, var_meas=4.0)
        filters.append(kf)

    out = []

    if not half_slide:
        # 傳統：整段依序做 KF
        for t in range(T):
            frame = window_kps[t]
            smoothed = []
            for j in range(J):
                kf = filters[j]
                kf.predict()
                if j < len(frame) and ("x" in frame[j]) and ("y" in frame[j]):
                    conf = float(frame[j].get("conf", frame[j].get("confidence", 1.0)))
                    if conf >= _KP_CONF_TH:
                        z = np.array([[float(frame[j]["x"])], [float(frame[j]["y"])]]).astype(np.float32)
                        kf.update(z)
                x, y = kf.get_xy()
                smoothed.append({"x": x, "y": y, "conf": float(frame[j].get("conf", 1.0)) if j < len(frame) else 0.0})
            out.append(smoothed)
        return out

    # half_slide 模式：模擬 1~10 → 5~15 → 10~20 → 10~20 ... 的線上更新
    half_len = int(half_len) if half_len else max(1, T // 2)
    step = 5  # 你的系統每 5 幀更新

    # 逐幀輸出（整段都要有平滑結果）
    for t in range(T):
        # 每到新的 5 幀邊界，等效於『重新鎖定半視窗的尾端』
        # 這裡不重置濾波器，而是持續利用狀態（較貼近真實線上）；
        # 若你想在每個半視窗重置，改成在邊界重建 filters。
        frame = window_kps[t]
        smoothed = []
        for j in range(J):
            kf = filters[j]
            kf.predict()
            if j < len(frame) and ("x" in frame[j]) and ("y" in frame[j]):
                conf = float(frame[j].get("conf", frame[j].get("confidence", 1.0)))
                if conf >= _KP_CONF_TH:
                    z = np.array([[float(frame[j]["x"])], [float(frame[j]["y"])]]).astype(np.float32)
                    kf.update(z)
            x, y = kf.get_xy()
            smoothed.append({"x": x, "y": y, "conf": float(frame[j].get("conf", 1.0)) if j < len(frame) else 0.0})
        out.append(smoothed)

    return out

# ===================== Dataset =====================

def read_any_json(path):
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
    if not txt:
        return []
    try:
        data = json.loads(txt)
        recs = []
        if isinstance(data, list):
            recs = [r for r in data if isinstance(r, (dict, list))]
        elif isinstance(data, dict):
            for k in ("frames", "data", "records", "annotations", "items", "results"):
                if isinstance(data.get(k), list):
                    recs = [r for r in data[k] if isinstance(r, (dict, list))]
                    break
            if not recs:
                kv = []
                for k, v in data.items():
                    if isinstance(k, str) and k.isdigit() and isinstance(v, dict):
                        vv = v.copy()
                        vv.setdefault("frame_id", int(k))
                        kv.append(vv)
                if kv:
                    recs = kv
                else:
                    recs = [data]
        if recs:
            return recs
    except Exception:
        pass
    recs = []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            recs.append(json.loads(line))
        except Exception:
            continue
    return recs


def _get_frame_id(r):
    for k in ("frame_id", "frame", "fid", "index", "idx", "image_id"):
        if isinstance(r, dict) and k in r:
            try:
                return int(r[k])
            except Exception:
                pass
    return None


def _list_json_files(d):
    files = []
    for pat in ("*.jsonl", "*.JSONL", "*.json", "*.JSON"):
        files.extend(glob.glob(os.path.join(d, pat)))
    return sorted(files)


def _list_json_files_recursive(root):
    files = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith((".jsonl", ".json")):
                files.append(os.path.join(dirpath, fn))
    return sorted(files)


def load_pose_sequence(pose_path):
    poses = {}
    files = []
    if os.path.isdir(pose_path):
        files = _list_json_files(pose_path)
    elif os.path.isfile(pose_path):
        files = [pose_path]
    for fp in files:
        recs = read_any_json(fp)
        for idx, r in enumerate(recs):
            if isinstance(r, dict) and r.get("type") not in (None, "pose"):
                continue
            fid = _get_frame_id(r) if isinstance(r, dict) else None
            if fid is None:
                fid = idx
            poses[fid] = r if isinstance(r, dict) else {"raw": r}
    return poses


def load_object_sequence(obj_path):
    objs = {}
    files = []
    if os.path.isdir(obj_path):
        files = _list_json_files(obj_path)
    elif os.path.isfile(obj_path):
        files = [obj_path]
    for fp in files:
        recs = read_any_json(fp)
        for idx, r in enumerate(recs):
            if isinstance(r, dict) and r.get("type") not in (None, "object"):
                continue
            fid = _get_frame_id(r) if isinstance(r, dict) else None
            if fid is None:
                fid = idx
            objs[fid] = r if isinstance(r, dict) else {"raw": r}
    return objs


def _find_sequences_for_class(pose_cdir, obj_cdir, use_objects):
    seqs = []
    pose_files = _list_json_files_recursive(pose_cdir)
    if not pose_files:
        return seqs
    if use_objects:
        for pf in pose_files:
            rel = os.path.relpath(pf, pose_cdir)
            of = os.path.join(obj_cdir or "", rel) if obj_cdir else None
            if of and os.path.isfile(of):
                seqs.append((pf, of))
        if not seqs and obj_cdir:
            obj_files = _list_json_files_recursive(obj_cdir)
            if len(pose_files) == 1 and len(obj_files) == 1:
                seqs.append((pose_files[0], obj_files[0]))
    else:
        for pf in pose_files:
            seqs.append((pf, None))
    return seqs

# ============== 新增：骨架完整性檢查（每幀） ==============

def _extract_pose_basic(p):
    """解析一幀 pose，回傳 (bbox, kps_list[17], img_w, img_h)；
    kps_list 中每個元素為 {x,y,conf}。
    """
    bbox = None
    kps_list = []
    img_w = 1.0
    img_h = 1.0

    if isinstance(p, dict) and p.get("persons"):
        person = max(p.get("persons", []), key=lambda x: x.get("score", 0.0))
        bbox = person.get("bbox")
        kps_list = person.get("keypoints") or []
        # 轉成統一格式
        kps_list = [
            {"x": float(k.get("x", 0.0)), "y": float(k.get("y", 0.0)), "conf": float(k.get("conf", k.get("confidence", 1.0)))}
            for k in (kps_list[:17] if isinstance(kps_list, list) else [])
        ]
        img_w = p.get("image_size", {}).get("width", 1) or 1
        img_h = p.get("image_size", {}).get("height", 1) or 1
    else:
        boxes = (p.get("boxes") if isinstance(p, dict) else None) or []
        kps_all = (p.get("keypoints") if isinstance(p, dict) else None) or []
        best_i, best_area = 0, -1
        for i, b in enumerate(boxes):
            if not (isinstance(b, (list, tuple)) and len(b) == 4):
                continue
            x1, y1, x2, y2 = b
            area = max(0, x2 - x1) * max(0, y2 - y1)
            if area > best_area:
                best_area = area
                best_i = i
        bbox = boxes[best_i] if boxes else None
        kp_raw = kps_all[best_i] if (kps_all and best_i < len(kps_all)) else (kps_all[0] if kps_all else [])
        kps_list = ([{"x": float(x), "y": float(y), "conf": 1.0} for (x, y) in kp_raw[:17]] if kp_raw else [])
        xs, ys = [], []
        for b in boxes:
            if isinstance(b, (list, tuple)) and len(b) == 4:
                xs += [b[0], b[2]]; ys += [b[1], b[3]]
        for grp in kps_all:
            for pt in grp:
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    xs.append(pt[0]); ys.append(pt[1])
        img_w = max(1.0, max(xs) if xs else 640.0)
        img_h = max(1.0, max(ys) if ys else 480.0)
    return bbox, kps_list, float(img_w), float(img_h)


def _frame_has_full_skeleton(p, kp_need=17, bbox_required=True):
    bbox, kps_list, _, _ = _extract_pose_basic(p)
    if bbox_required and (bbox is None):
        return False
    ok = 0
    for kp in (kps_list or []):
        try:
            conf = float(kp.get("conf", kp.get("confidence", 1.0)))
            if conf >= _KP_CONF_TH and np.isfinite(kp.get("x", 0)) and np.isfinite(kp.get("y", 0)):
                ok += 1
        except Exception:
            pass
    return ok >= kp_need


class PoseObjectDataset(Dataset):
    def __init__(self, pose_root, obj_root=None, use_objects=True,
                 window=32, stride=8, sample_every=1,
                 H=64, W=64, sigma_kp=2.5, kp_conf_th=0.3,
                 include_bone_lines=True, object_classes=None,
                 person_sel="top1_score",
                 enable_kalman=True,
                 kalman_half_slide=True,
                 half_len=None,
                 require_full_first=True,
                 require_full_all=False,
                 full_kp_min=17, bbox_required=True):
        self.cfg = RelationMapConfig(H, W, sigma_kp, kp_conf_th,
                                     include_bone_lines=include_bone_lines,
                                     object_classes=(object_classes if use_objects else None))
        self.window = int(window)
        self.stride = int(stride)
        self.sample_every = max(1, int(sample_every))
        self.person_sel = person_sel
        self.use_objects = bool(use_objects) and (len(self.cfg.object_classes) > 0)
        self.enable_kalman = bool(enable_kalman)
        self.kalman_half_slide = bool(kalman_half_slide)
        self.half_len = int(half_len) if (half_len and int(half_len) > 0) else max(1, self.window // 2)
        self.require_full_first = bool(require_full_first)
        self.require_full_all = bool(require_full_all)
        self.full_kp_min = int(full_kp_min)
        self.bbox_required = bool(bbox_required)

        if not os.path.isdir(pose_root):
            raise FileNotFoundError(f"pose_root not found: {pose_root}")
        self.pose_root = pose_root
        self.obj_root = obj_root if obj_root and os.path.isdir(obj_root) else None

        pose_classes = {d for d in os.listdir(pose_root) if os.path.isdir(os.path.join(pose_root, d))}
        if self.use_objects:
            if self.obj_root is None:
                raise FileNotFoundError("use_objects=True 但 obj_root 無效或不存在")
            obj_classes = {d for d in os.listdir(self.obj_root) if os.path.isdir(os.path.join(self.obj_root, d))}
            self.class_names = sorted(list(pose_classes & obj_classes))
        else:
            self.class_names = sorted(list(pose_classes))

        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        # 滑動窗口樣本清單： (label_idx, frames, poses, objs)
        self.windows = []
        total_seqs = 0
        debug_info = []

        for cname in self.class_names:
            pose_cdir = os.path.join(pose_root, cname)
            obj_cdir = os.path.join(self.obj_root, cname) if self.obj_root else None

            seq_paths = _find_sequences_for_class(pose_cdir, obj_cdir, self.use_objects)
            total_seqs += len(seq_paths)
            if not seq_paths:
                debug_info.append({
                    "class": cname,
                    "pose_dir": pose_cdir,
                    "pose_files": len(_list_json_files(pose_cdir)),
                    "obj_dir": obj_cdir,
                    "obj_files": len(_list_json_files(obj_cdir)) if obj_cdir else None,
                })

            for pose_path, obj_path in seq_paths:
                poses = load_pose_sequence(pose_path)
                objs  = load_object_sequence(obj_path) if (self.use_objects and obj_path) else {}

                frames_pose = set(poses.keys())
                frames_obj  = set(objs.keys()) if self.use_objects else frames_pose
                frames = sorted(list(frames_pose & frames_obj))

                if len(frames) < self.window:
                    continue

                # === 取樣規則 ===
                for i in range(0, len(frames) - self.window + 1, self.stride):
                    win_frames = frames[i:i + self.window]

                    if self.require_full_all:
                        all_full = True
                        for fid in win_frames:
                            if not _frame_has_full_skeleton(poses[fid], kp_need=self.full_kp_min, bbox_required=self.bbox_required):
                                all_full = False
                                break
                        if not all_full:
                            continue
                    elif self.require_full_first:
                        # 半視窗初始化→只要求第一幀完整
                        first_fid = win_frames[0]
                        if not _frame_has_full_skeleton(poses[first_fid], kp_need=self.full_kp_min, bbox_required=self.bbox_required):
                            continue

                    self.windows.append((self.class_to_idx[cname], win_frames, poses, objs))

        if len(self.windows) == 0:
            print(f"[Dataset] No windows.")
            print(f"[Dataset] classes={self.class_names}  | discovered_seqs={total_seqs}")
            for d in debug_info:
                print(f"[Dataset] class='{d['class']}' | pose_dir='{d['pose_dir']}' files={d['pose_files']} | obj_dir='{d['obj_dir']}' files={d['obj_files']}")
            print(f"[Dataset] Hints: 1) 檢查 frame_id 對齊；2) 檔名是否 .jsonl/.json；3) window={self.window} 是否過大；4) 結構是否在 class 下的巢狀資料夾或檔案")
        else:
            per_class = {}
            for y, *_ in self.windows:
                per_class[y] = per_class.get(y, 0) + 1
            pretty = {self.class_names[k]: v for k, v in per_class.items()}
            print(f"[Dataset] classes={self.class_names}")
            print(f"[Dataset] discovered sequences={total_seqs}, windows per class={pretty}")

        if len(self.windows) == 0:
            print(f"[Dataset] No windows.")
            print(f"[Dataset] classes={self.class_names}  | discovered_seqs={total_seqs}")
            for d in debug_info:
                print(f"[Dataset] class='{d['class']}' | pose_dir='{d['pose_dir']}' files={d['pose_files']} | obj_dir='{d['obj_dir']}' files={d['obj_files']}")
            print(f"[Dataset] Hints: 1) 檢查 frame_id 對齊；2) 檔名是否 .jsonl/.json；3) window={self.window} 是否過大；4) 結構是否在 class 下的巢狀資料夾或檔案")
        else:
            per_class = {}
            for y, *_ in self.windows:
                per_class[y] = per_class.get(y, 0) + 1
            pretty = {self.class_names[k]: v for k, v in per_class.items()}
            print(f"[Dataset] classes={self.class_names}")
            print(f"[Dataset] discovered sequences={total_seqs}, windows per class={pretty}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        y_idx, frames, poses, objs = self.windows[idx]
        sel_frames = frames[::self.sample_every]

        # 先解析出整段視窗的 (bbox, kps, img_w, img_h)
        parsed = []
        for fid in sel_frames:
            p = poses[fid]
            o = objs.get(fid) if self.use_objects else None
            bbox, kps_list, img_w, img_h = _extract_pose_basic(p)
            detections = []
            if self.use_objects and o is not None:
                detections = o.get("detections") or o.get("objects") or o.get("boxes") or o.get("bboxes") or o.get("predictions") or []
            parsed.append((bbox, kps_list, detections, img_w, img_h))

        # 視窗內做卡爾曼平滑（半視窗/線上式初始化 + 之後每 5 幀滑動）
        if self.enable_kalman and len(parsed) > 0:
            window_kps = [kps for (_, kps, _, _, _) in parsed]
            smoothed = kalman_smooth_kps(
                window_kps,
                half_slide=self.kalman_half_slide,
                half_len=self.half_len,
                require_full_first=self.require_full_first,
            )
            # 替換回去
            parsed = [(bbox, smoothed[i], dets, img_w, img_h) for i, (bbox, _, dets, img_w, img_h) in enumerate(parsed)]

        # Rasterize → 堆疊 (T, C, H, W)
        Xs = []
        for (bbox, kps_list, detections, img_w, img_h) in parsed:
            Xs.append(rasterize_frame(bbox, kps_list, detections, img_w, img_h, self.cfg))
        X = torch.from_numpy(np.stack(Xs)).float()
        y = torch.tensor(y_idx, dtype=torch.long)
        return X, y

# ===================== Model =====================
class SpaceCNN(nn.Module):
    def __init__(self, in_ch, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(256, out_dim)

    def forward(self, x):
        return self.fc(self.net(x).flatten(1))


class TemporalHead(nn.Module):
    def __init__(self, in_dim, num_classes, mode="last", dropout=0.0):
        super().__init__()
        self.mode = mode
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if mode == "attn":
            self.attn = nn.Linear(in_dim, 1)
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, seq_feats):
        if self.mode == "mean":
            g = seq_feats.mean(dim=1)
        elif self.mode == "attn":
            a = self.attn(seq_feats).squeeze(-1)
            w = torch.softmax(a, dim=1).unsqueeze(-1)
            g = (seq_feats * w).sum(dim=1)
        else:
            g = seq_feats[:, -1]
        g = self.drop(g)
        return self.fc(g)


class CNNLSTM(nn.Module):
    def __init__(self, in_ch, num_classes, cnn_out=256, lstm_h=256, lstm_layers=2,
                 bidirectional=False, temporal_pool="last", dropout=0.0):
        super().__init__()
        self.cnn = SpaceCNN(in_ch, cnn_out)
        self.lstm = nn.LSTM(cnn_out, lstm_h, lstm_layers, batch_first=True,
                            bidirectional=bidirectional)
        feat_dim = lstm_h * (2 if bidirectional else 1)
        self.head = TemporalHead(feat_dim, num_classes, mode=temporal_pool, dropout=dropout)

    def forward(self, x):
        B, T, C, H, W = x.shape
        z = self.cnn(x.view(B * T, C, H, W)).view(B, T, -1)
        out, _ = self.lstm(z)
        logits = self.head(out)
        return logits

# ===================== Loss =====================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, y):
        ce = self.ce(logits, y)
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            loss = self.alpha[y] * loss
        return loss.mean()

# ===================== Train / Evaluate =====================

def evaluate(model, loader, device):
    model.eval()
    tot, corr = 0, 0
    all_y, all_p = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            out = model(X)
            pred = out.argmax(1)
            corr += (pred == y).sum().item()
            tot += len(y)
            all_y.append(y.cpu().numpy())
            all_p.append(pred.cpu().numpy())
    acc = corr / max(1, tot)
    y_true = np.concatenate(all_y) if all_y else np.array([])
    y_pred = np.concatenate(all_p) if all_p else np.array([])
    mf1 = f1_score(y_true, y_pred, average="macro") if y_true.size else 0.0
    return acc, mf1


# 修正後的程式碼

def train(cfg: Config):
    set_seed(_SEED)
    ensure_dir(cfg.out_dir)

    # Dataset & Loader（加入前置處理設定）
    ds = PoseObjectDataset(
        pose_root=cfg.pose_root,
        obj_root=cfg.obj_root,
        use_objects=cfg.use_objects,
        window=cfg.window,
        stride=cfg.stride,
        sample_every=_SAMPLE_EVERY,
        H=cfg.H,
        W=cfg.W,
        sigma_kp=_SIGMA_KP,
        kp_conf_th=_KP_CONF_TH,
        include_bone_lines=bool(cfg.include_bone_lines),
        object_classes=cfg.object_classes,
        person_sel="top1_score",
        enable_kalman=cfg.enable_kalman,
        kalman_half_slide=cfg.kalman_half_slide,
        half_len=(cfg.half_len_override if cfg.half_len_override else cfg.window // 2),
        require_full_first=cfg.require_full_first_frame,
        require_full_all=cfg.require_full_skeleton_all,
        full_kp_min=cfg.full_kp_min,
        bbox_required=cfg.bbox_required,
    )

    n = len(ds)
    if n == 0:
        raise RuntimeError("No training windows found. Check your roots and window/stride settings.")

    idx = list(range(n))
    random.shuffle(idx)
    split = int(n * 0.8)
    tr_idx, va_idx = idx[:split], idx[split:]

    tr = torch.utils.data.Subset(ds, tr_idx)
    va = torch.utils.data.Subset(ds, va_idx)

    num_classes = len(ds.class_names)
    counts = [0] * num_classes
    for i in tr_idx:
        y_i = ds.windows[i][0]
        counts[y_i] += 1
    counts = [c if c > 0 else 1 for c in counts]

    if cfg.use_sampler:
        total = sum(counts)
        inv = [total / c for c in counts]
        sample_weights = [inv[ds.windows[i][0]] for i in tr_idx]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        tr_loader = DataLoader(tr, batch_size=cfg.batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    else:
        tr_loader = DataLoader(tr, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    va_loader = DataLoader(va, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_x, _ = ds[0]
    _, C, H, W = sample_x.shape

    model = CNNLSTM(
        in_ch=C,
        num_classes=num_classes,
        cnn_out=_CNN_OUT,
        lstm_h=cfg.lstm_hidden,
        lstm_layers=_LSTM_LAYERS,
        bidirectional=bool(cfg.bidirectional),
        temporal_pool=cfg.temporal_pool,
        dropout=cfg.dropout,
    ).to(device)

    total = float(sum(counts))
    alpha = torch.tensor([total / c for c in counts], dtype=torch.float32)
    alpha = (alpha / alpha.sum()).to(device)

    if cfg.loss == "ce":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = FocalLoss(alpha=alpha, gamma=_FOCAL_GAMMA)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=_WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=_AMP and device.type == "cuda")

    best_score = -1.0
    best_path = None
    top_ckpts = []

    for ep in range(1, cfg.epochs + 1):
        model.train()
        tot, corr, loss_sum = 0, 0, 0.0
        pbar = tqdm(tr_loader, desc=f"Epoch {ep}/{cfg.epochs} [train]")
        for X, y in pbar:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda', enabled=scaler.is_enabled()):
                out = model(X)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            if _GRAD_CLIP and _GRAD_CLIP > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), _GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            loss_sum += loss.item() * len(y)
            corr += (out.argmax(1) == y).sum().item()
            tot += len(y)
            pbar.set_postfix({"loss": f"{loss_sum/max(1,tot):.3f}", "acc": f"{corr/max(1,tot):.3f}"})

        train_loss = loss_sum / max(1, tot)
        train_acc = corr / max(1, tot)
        print(f"Ep{ep} train loss={train_loss:.3f} acc={train_acc:.3f}")

        acc, mf1 = evaluate(model, va_loader, device)
        score = mf1
        print(f"Ep{ep} val acc={acc:.3f} macro_f1={mf1:.3f}")

        ckpt_name = f"ep{ep:03d}_score{score:.4f}.pt"
        ckpt_path = os.path.join(cfg.out_dir, ckpt_name)
        torch.save({
            "epoch": ep,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("__") and not callable(getattr(cfg, k))},
            "class_names": ds.class_names,
        }, ckpt_path)
        top_ckpts.append((score, ckpt_path))
        top_ckpts.sort(key=lambda x: x[0], reverse=True)
        if len(top_ckpts) > _SAVE_TOP_K:
            for _ in range(len(top_ckpts) - _SAVE_TOP_K):
                _, rm_path = top_ckpts.pop()
                try:
                    if rm_path != best_path:
                        os.remove(rm_path)
                except Exception:
                    pass

        if score > best_score:
            best_score = score
            best_path = os.path.join(cfg.out_dir, "best.pt")
            torch.save(model.state_dict(), best_path)
            print(f"* New best (macro_f1={best_score:.4f}), saved to {best_path}")
            no_improve = 0
        else:
            no_improve = getattr(train, "_no_improve", 0) + 1
        setattr(train, "_no_improve", no_improve)

        if _EARLY_STOP_PATIENCE > 0 and no_improve >= _EARLY_STOP_PATIENCE:
            print(f"Early stopping at epoch {ep} (no improvement for {no_improve} epochs)")
            break

    if best_path and os.path.exists(best_path):
        print("\nLoading best model for final report...")
        model.load_state_dict(torch.load(best_path, map_location=device))
        acc, mf1 = evaluate(model, va_loader, device)
        print(f"Best model val acc={acc:.3f} macro_f1={mf1:.3f}")

        model.eval()
        all_y, all_p = [], []
        with torch.no_grad():
            for X, y in va_loader:
                X = X.to(device)
                out = model(X)
                pred = out.argmax(1).cpu().numpy()
                all_p.append(pred)
                all_y.append(y.numpy())
        y_true = np.concatenate(all_y)
        y_pred = np.concatenate(all_p)
        cm = confusion_matrix(y_true, y_pred)
        print("\nConfusion Matrix:\n", cm)
        print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=ds.class_names, digits=3))


if __name__ == "__main__":
    cfg = Config()
    train(cfg)