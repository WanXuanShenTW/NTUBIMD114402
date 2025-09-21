# -*- coding: utf-8 -*-
"""
以你最新上傳的『基準版 cnn_lstm_trainer.py』為底，補上此前要求但不破壞原有介面：
- ✅ tqdm 進度條（train / valid）與每 Ep 摘要列印（保留）
- ✅ Checkpoint：每回合 `last.pt`；`best.pt`；並**維護 Top-3** → `best_epXXX.pt` + `topk.json`
- ✅ KF 半視窗（不改 raw、只在 Dataset 前處理）
- ✅ motion(9 維) + 有效幀 mask，與**TemporalHead 支援 mask**（mean/attn 忽略無效幀）
- ✅ motion 正規化（dataset 估計 mean/std → 存入 ckpt）
- ✅ RelationMap 通道與 inference 對齊：bbox(1)+dist(1)+kp(17)+bone(12 可選)+object(N)+coord(2)
- ✅ 參數型別**沿用基準**（例如 `use_objects: bool`）。

使用方式：與你原本相同（若你沒有 object，就設 `use_objects=False`；其他參數不必新增）。
"""

import os, json, glob, random, math
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from tqdm import tqdm
from typing import Optional, List, Tuple, Dict

# ===================== Config（沿用基準類別＋布林 use_objects） =====================
class Config:
    # 路徑（分開設定）
    pose_root = "outputs/skeletons/YOLO/YOLO-pose"
    obj_root  = "outputs/skeletons/YOLO/YOLO-detect"
    out_dir   = "outputs/models/test"

    # 是否使用物件框
    use_objects = True            # ← 保持為 bool，只用於是否使用 objects
    object_classes = ["bed", "chair"]  # 空則不建立物件通道

    # 時序
    window = 20
    stride = 5

    # Relation Map
    H, W = 64, 64
    include_bone_lines = True

    # 模型
    lstm_hidden  = 256
    bidirectional = False
    temporal_pool = "attn"       # "last" | "mean" | "attn"
    dropout = 0.3

    # 訓練
    epochs     = 40
    batch_size = 8
    lr         = 1e-3
    use_sampler = True
    loss = "focal"               # "focal" | "ce"

    # 輸入前處理 / 過濾規則（沿用基準）
    enable_kalman = True
    kalman_half_slide = True
    require_full_first_frame = True
    half_len_override = None
    require_full_skeleton_all = False
    full_kp_min = 17
    bbox_required = True

# ===================== 穩定預設 =====================
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

COCO_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]

# ===================== Utils =====================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

# ===================== Relation Map（對齊推論側） =====================
class RelationMapConfig:
    def __init__(self, H=64, W=64, sigma_kp=3.0, kp_conf_th=0.4,
                 include_bone_lines=True, object_classes=None):
        self.H = int(H); self.W = int(W)
        self.sigma_kp = float(sigma_kp)
        self.kp_conf_th = float(kp_conf_th)
        self.include_bone_lines = bool(include_bone_lines)
        self.object_classes = [c.strip() for c in (object_classes or []) if c and c.strip()]

def gaussian2d(shape, sigma):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y)/(2*sigma*sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_gaussian(canvas, cx, cy, sigma, mag=1.0):
    if cx is None or cy is None:
        return
    Hc, Wc = canvas.shape
    x = int(cx); y = int(cy)
    size = int(6 * sigma + 3)
    g = gaussian2d((size, size), sigma)
    x0 = x - size // 2; y0 = y - size // 2
    x1 = min(Wc, x0 + size); y1 = min(Hc, y0 + size)
    g_x0 = max(0, -x0); g_y0 = max(0, -y0)
    gx1 = g_x0 + (x1 - max(0, x0)); gy1 = g_y0 + (y1 - max(0, y0))
    if x0 < Wc and y0 < Hc and x1 > 0 and y1 > 0:
        canvas[max(0,y0):y1, max(0,x0):x1] = np.maximum(
            canvas[max(0,y0):y1, max(0,x0):x1],
            g[g_y0:gy1, g_x0:gx1] * float(mag)
        )

def _bbox_xyxy_from_any(b):
    if b is None:
        return None
    if isinstance(b, dict):
        if all(k in b for k in ("cx","cy","w","h")):
            cx,cy,w,h = float(b['cx']),float(b['cy']),float(b['w']),float(b['h'])
            return cx-w/2, cy-h/2, cx+w/2, cy+h/2
        if all(k in b for k in ("x","y","w","h")):
            x,y,w,h = float(b['x']),float(b['y']),float(b['w']),float(b['h'])
            return x, y, x+w, y+h
    if isinstance(b,(list,tuple)) and len(b)==4:
        x1,y1,x2,y2 = map(float,b)
        return x1,y1,x2,y2
    return None

def rasterize_frame(bbox, kps_list, dets, img_w, img_h, cfg: RelationMapConfig):
    Hc, Wc = cfg.H, cfg.W
    num_kp = 17
    num_edges = len(COCO_EDGES) if cfg.include_bone_lines else 0
    num_obj_ch = len(cfg.object_classes)
    num_coord = 2
    C = 2 + num_kp + num_edges + num_obj_ch + num_coord
    canvas = np.zeros((C, Hc, Wc), dtype=np.float32)
    ch = 0
    # 1) bbox mask
    if bbox:
        bxyxy = _bbox_xyxy_from_any(bbox)
        if bxyxy is not None:
            x1,y1,x2,y2 = bxyxy
            x1 = int(np.clip((x1/img_w)*Wc, 0, Wc-1)); x2 = int(np.clip((x2/img_w)*Wc, 0, Wc-1))
            y1 = int(np.clip((y1/img_h)*Hc, 0, Hc-1)); y2 = int(np.clip((y2/img_h)*Hc, 0, Hc-1))
            if x2>=x1 and y2>=y1:
                canvas[ch, y1:y2+1, x1:x2+1] = 1.0
    ch += 1
    # 2) distance transform
    inv = (1.0 - canvas[ch-1]).astype(np.uint8)*255
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    if dist.max() > 0:
        dist = dist / dist.max()
    canvas[ch] = dist; ch += 1
    # 3) keypoints heatmaps
    if kps_list:
        for i, kp in enumerate(kps_list[:num_kp]):
            try:
                conf = float(kp.get('conf', kp.get('confidence',1.0)))
                if conf < _KP_CONF_TH:
                    continue
                x = (float(kp['x'])/img_w)*Wc; y = (float(kp['y'])/img_h)*Hc
                draw_gaussian(canvas[ch+i], x, y, _SIGMA_KP, mag=conf)
            except Exception:
                pass
    ch += num_kp
    # 4) bone lines
    if num_edges > 0 and kps_list and len(kps_list) >= num_kp:
        pts = []
        for i in range(num_kp):
            try:
                x = int(np.clip((float(kps_list[i]['x'])/img_w)*Wc, 0, Wc-1))
                y = int(np.clip((float(kps_list[i]['y'])/img_h)*Hc, 0, Hc-1))
                c = float(kps_list[i].get('conf', kps_list[i].get('confidence',1.0)))
                pts.append((x,y,c))
            except Exception:
                pts.append((None,None,0.0))
        for e_idx, (a,b) in enumerate(COCO_EDGES):
            x1,y1,c1 = pts[a]; x2,y2,c2 = pts[b]
            if x1 is None or x2 is None: continue
            if min(c1,c2) < _KP_CONF_TH: continue
            cv2.line(canvas[ch+e_idx], (x1,y1), (x2,y2), 1.0, 1)
    ch += num_edges
    # 5) objects
    if num_obj_ch > 0 and dets:
        cls2ch = {name: i for i, name in enumerate(cfg.object_classes)}
        for d in dets or []:
            name = d.get("cls_name") or d.get("class_name") or d.get("label") or d.get("name")
            if name not in cls2ch:
                continue
            bb = d.get("bbox") or d.get("xyxy") or {}
            bxyxy = _bbox_xyxy_from_any(bb)
            if bxyxy is None: continue
            x1,y1,x2,y2 = bxyxy
            x1 = int(np.clip((x1/img_w)*Wc, 0, Wc-1)); x2 = int(np.clip((x2/img_w)*Wc, 0, Wc-1))
            y1 = int(np.clip((y1/img_h)*Hc, 0, Hc-1)); y2 = int(np.clip((y2/img_h)*Hc, 0, Hc-1))
            if x2>=x1 and y2>=y1:
                canvas[ch+cls2ch[name], y1:y2+1, x1:x2+1] = 1.0
    ch += num_obj_ch
    # 6) CoordConv（x/y）
    xv = np.linspace(-1, 1, Wc)[None, :].repeat(Hc, 0)
    yv = np.linspace(-1, 1, Hc)[:, None].repeat(Wc, 1)
    canvas[ch] = xv; canvas[ch+1] = yv; ch += 2
    return canvas

# ===================== 骨架完整性（第一幀用） =====================
def frame_has_full_skeleton(bbox, kps, *, kp_need=17, require_bbox=True):
    if require_bbox and not bbox:
        return False
    if not kps:
        return False
    cnt = 0
    for j in range(min(17, len(kps))):
        if ('x' in kps[j]) and ('y' in kps[j]):
            conf = float(kps[j].get('conf', kps[j].get('confidence', 1.0)))
            if conf >= _KP_CONF_TH:
                cnt += 1
    return cnt >= kp_need

# ===================== Kalman（半視窗） =====================
class Kalman2D:
    def __init__(self, x=0.0, y=0.0, var_pos=1e-2, var_vel=1e-1, var_meas=4.0):
        self.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=np.float32)
        self.H = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float32)
        self.Q = np.diag([var_pos, var_pos, var_vel, var_vel]).astype(np.float32)
        self.R = np.diag([var_meas, var_meas]).astype(np.float32)
        self.x = np.array([[x], [y], [0.0], [0.0]], dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 10.0
    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
    def update(self, z):
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y)
        I = np.eye(4, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P
    def get_xy(self):
        return float(self.x[0,0]), float(self.x[1,0])

def _kf_run_on_segment(kps_seq, require_full_first=True):
    L = len(kps_seq)
    if L == 0:
        return []
    if require_full_first:
        bbox_dummy = [0,0,1,1]
        if not frame_has_full_skeleton(bbox_dummy, kps_seq[0], kp_need=17, require_bbox=False):
            return None
    J = 17
    first = kps_seq[0]
    filters = []
    for j in range(J):
        if j < len(first) and ('x' in first[j]) and ('y' in first[j]):
            kf = Kalman2D(first[j]['x'], first[j]['y'])
        else:
            kf = Kalman2D(0.0, 0.0)
        filters.append(kf)
    out = []
    for t in range(L):
        frame = kps_seq[t]
        smoothed = []
        for j in range(J):
            kf = filters[j]
            kf.predict()
            if j < len(frame) and ('x' in frame[j]) and ('y' in frame[j]):
                conf = float(frame[j].get('conf', frame[j].get('confidence', 1.0)))
                if conf >= _KP_CONF_TH:
                    z = np.array([[float(frame[j]['x'])],[float(frame[j]['y'])]], dtype=np.float32)
                    kf.update(z)
            x,y = kf.get_xy()
            smoothed.append({'x': x, 'y': y, 'conf': float(frame[j].get('conf',1.0)) if j < len(frame) else 0.0})
        out.append(smoothed)
    return out

def kalman_smooth_kps(window_kps, *, half_slide=True, half_len=10, require_full_first=True, step=5):
    T = len(window_kps)
    if T == 0:
        return window_kps
    if not half_slide:
        seq = _kf_run_on_segment(window_kps, require_full_first=require_full_first)
        return seq if seq is not None else window_kps
    half_len = int(half_len) if half_len else max(1, T//2)
    step = int(step)
    out = [None]*T
    seg0 = _kf_run_on_segment(window_kps[0:half_len], require_full_first=require_full_first)
    if seg0 is None:
        return window_kps
    for t in range(min(half_len, T)):
        out[t] = seg0[t]
    s = step
    while s + half_len <= T:
        seg = _kf_run_on_segment(window_kps[s:s+half_len], require_full_first=require_full_first)
        if seg is not None:
            a = s + half_len - step
            b = s + half_len
            for t_rel, t_abs in enumerate(range(a, b)):
                if 0 <= t_abs < T:
                    out[t_abs] = seg[half_len - step + t_rel]
        s += step
    if any(x is None for x in out):
        fb = _kf_run_on_segment(window_kps, require_full_first=False) or window_kps
        for i in range(T):
            if out[i] is None:
                out[i] = fb[i]
    return out

# ===================== Motion + Mask（與推論一致） =====================
def _safe_mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals)/len(vals) if vals else None

def _angle(a,b,c):
    if (a is None) or (b is None) or (c is None): return None
    ba = (a[0]-b[0], a[1]-b[1]); bc = (c[0]-b[0], c[1]-b[1])
    nba = math.hypot(*ba); nbc = math.hypot(*bc)
    if nba<1e-6 or nbc<1e-6: return None
    cosv = (ba[0]*bc[0] + ba[1]*bc[1])/(nba*nbc)
    cosv = max(-1.0, min(1.0, cosv))
    return math.acos(cosv)

def _kp_xy(kps, i, img_w, img_h, conf_th):
    if i < len(kps):
        d = kps[i]
        conf = float(d.get("conf", d.get("confidence", 1.0)))
        if conf >= conf_th and ("x" in d) and ("y" in d):
            return (float(d["x"]) / img_w, float(d["y"]) / img_h)
    return None

def compute_motion_feats_with_mask_from_parsed(parsed, conf_th=_KP_CONF_TH):
    """parsed: List[(bbox, kps, dets, img_w, img_h)] -> (T,9), (T,)"""
    T = len(parsed)
    if T == 0:
        return np.zeros((0,9), np.float32), np.zeros((0,), np.float32)
    ycom, hgt, area, trunk, kneeL, kneeR = [], [], [], [], [], []
    for (bbox, kps, _d, img_w, img_h) in parsed:
        hips = [_kp_xy(kps,11,img_w,img_h,conf_th), _kp_xy(kps,12,img_w,img_h,conf_th)]
        hs = [p for p in hips if p is not None]
        if hs:
            y_c = _safe_mean([p[1] for p in hs])
        else:
            shs = [_kp_xy(kps,5,img_w,img_h,conf_th), _kp_xy(kps,6,img_w,img_h,conf_th)]
            ss = [p for p in shs if p is not None]
            if ss:
                y_c = _safe_mean([p[1] for p in ss])
            else:
                ys = [float(p["y"]) / img_h for p in kps if float(p.get("conf", p.get("confidence",1.0))) >= conf_th and ("y" in p)]
                y_c = _safe_mean(ys)
        ycom.append(y_c)
        if bbox is not None:
            x1,y1,x2,y2 = _bbox_xyxy_from_any(bbox)
            bw = max(1e-6, (x2-x1)/img_w); bh = max(1e-6, (y2-y1)/img_h)
            h = bh; w = bw
        else:
            ys = [float(p["y"]) / img_h for p in kps if float(p.get("conf", p.get("confidence",1.0))) >= conf_th and ("y" in p)]
            if len(ys) >= 2:
                h = max(ys)-min(ys); w = 0.4*h
            else:
                h=None; w=None
        hgt.append(h)
        area.append((w*h) if (w is not None and h is not None) else None)
        shL=_kp_xy(kps,5,img_w,img_h,conf_th); shR=_kp_xy(kps,6,img_w,img_h,conf_th)
        hpL=_kp_xy(kps,11,img_w,img_h,conf_th); hpR=_kp_xy(kps,12,img_w,img_h,conf_th)
        knL=_kp_xy(kps,13,img_w,img_h,conf_th); anL=_kp_xy(kps,15,img_w,img_h,conf_th)
        knR=_kp_xy(kps,14,img_w,img_h,conf_th); anR=_kp_xy(kps,16,img_w,img_h,conf_th)
        if shL and shR and hpL and hpR:
            sh=((shL[0]+shR[0])/2,(shL[1]+shR[1])/2)
            hp=((hpL[0]+hpR[0])/2,(hpL[1]+hpR[1])/2)
            vec=(hp[0]-sh[0], hp[1]-sh[1])
            ang=abs(math.atan2(vec[0], vec[1]))
        else:
            ang=None
        trunk.append(ang)
        kneeL.append(_angle(hpL, knL, anL))
        kneeR.append(_angle(hpR, knR, anR))
    def fill_small(arr):
        arr=list(arr); n=len(arr); i=0
        while i<n:
            if arr[i] is None:
                j=i
                while j<n and arr[j] is None: j+=1
                gap=j-i
                if gap<=3 and i>0 and j<n and arr[i-1] is not None and arr[j] is not None:
                    for k in range(gap):
                        w=(k+1)/(gap+1); arr[i+k]=arr[i-1]*(1-w)+arr[j]*w
                i=j
            else:
                i+=1
        return [0.0 if v is None else v for v in arr]
    ycom=fill_small(ycom); hgt=fill_small(hgt); area=fill_small(area)
    trunk=fill_small(trunk); kneeL=fill_small(kneeL); kneeR=fill_small(kneeR)
    ycom=np.array(ycom,np.float32); hgt=np.array(hgt,np.float32); area=np.array(area,np.float32)
    trunk=np.array(trunk,np.float32); kneeL=np.array(kneeL,np.float32); kneeR=np.array(kneeR,np.float32)
    def diff1(x):
        v=np.zeros_like(x); v[1:]=x[1:]-x[:-1]; return v
    def diff2(v):
        a=np.zeros_like(v); a[1:]=v[1:]-v[:-1]; return a
    v_y, a_y = diff1(ycom), diff2(diff1(ycom))
    v_h, a_h = diff1(hgt),  diff2(diff1(hgt))
    v_A, a_A = diff1(area), diff2(diff1(area))
    dtrunk   = diff1(trunk)
    dkneeL   = diff1(kneeL); dkneeR = diff1(kneeR)
    feats = np.stack([
        np.clip(v_y, -3.0, 3.0),
        np.clip(a_y, -9.0, 9.0),
        np.clip(v_h, -3.0, 3.0),
        np.clip(a_h, -9.0, 9.0),
        np.clip(v_A, -3.0, 3.0),
        np.clip(a_A, -9.0, 9.0),
        np.clip(dtrunk, -3.0, 3.0),
        np.clip(dkneeL, -3.0, 3.0),
        np.clip(dkneeR, -3.0, 3.0),
    ], axis=1).astype(np.float32)
    valid=[]
    for (bbox, kps, _d, _iw, _ih) in parsed:
        cnt=0
        for j in (11,12,5,6,13,14):
            if j < len(kps):
                conf=float(kps[j].get("conf", kps[j].get("confidence",1.0)))
                if conf>=_KP_CONF_TH: cnt+=1
        ok=(cnt>=2) or (bbox is not None)
        valid.append(1.0 if ok else 0.0)
    valid=np.array(valid,np.float32)
    return feats, valid

# ===================== 資料讀取 =====================
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
                        vv = v.copy(); vv.setdefault("frame_id", int(k)); kv.append(vv)
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
        for dirpath, _, filenames in os.walk(pose_path):
            for fn in filenames:
                if fn.lower().endswith((".jsonl", ".json")):
                    files.append(os.path.join(dirpath, fn))
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
    if obj_path and os.path.isdir(obj_path):
        for dirpath, _, filenames in os.walk(obj_path):
            for fn in filenames:
                if fn.lower().endswith((".jsonl", ".json")):
                    files.append(os.path.join(dirpath, fn))
    elif obj_path and os.path.isfile(obj_path):
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
        if (not seqs) and obj_cdir:
            obj_files = _list_json_files_recursive(obj_cdir)
            if len(pose_files) == 1 and len(obj_files) == 1:
                seqs.append((pose_files[0], obj_files[0]))
    else:
        for pf in pose_files:
            seqs.append((pf, None))
    return seqs

# ============== 解析一幀 pose ==============

def _extract_pose_basic(p):
    bbox = None
    kps_list = []
    img_w = 1.0
    img_h = 1.0
    if isinstance(p, dict) and p.get("persons"):
        person = max(p.get("persons", []), key=lambda x: x.get("score", 0.0))
        bbox = person.get("bbox")
        kps_list = person.get("keypoints") or []
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
                best_area = area; best_i = i
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

# ===================== Dataset =====================
class PoseObjectDataset(Dataset):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rm_cfg = RelationMapConfig(H=cfg.H, W=cfg.W, sigma_kp=_SIGMA_KP, kp_conf_th=_KP_CONF_TH,
                                        include_bone_lines=bool(cfg.include_bone_lines),
                                        object_classes=(cfg.object_classes if cfg.use_objects else []))
        self.window = int(cfg.window)
        self.stride = int(cfg.stride)
        self.use_objects = bool(cfg.use_objects) and len(self.rm_cfg.object_classes)>0 and cfg.obj_root and os.path.isdir(cfg.obj_root)
        self.enable_kf = bool(cfg.enable_kalman)
        self.kf_half = bool(cfg.kalman_half_slide)
        self.half_len = int(cfg.half_len_override) if cfg.half_len_override else max(1,self.window//2)
        self.require_full_first = bool(cfg.require_full_first_frame)
        self.require_full_all = bool(cfg.require_full_skeleton_all)
        self.full_kp_min = int(cfg.full_kp_min)
        self.bbox_required = bool(cfg.bbox_required)
        if not os.path.isdir(cfg.pose_root):
            raise FileNotFoundError(cfg.pose_root)
        self.pose_root = cfg.pose_root
        self.obj_root = (cfg.obj_root if self.use_objects else None)
        pose_classes = {d for d in os.listdir(self.pose_root) if os.path.isdir(os.path.join(self.pose_root, d))}
        if self.use_objects:
            obj_classes = {d for d in os.listdir(self.obj_root) if os.path.isdir(os.path.join(self.obj_root, d))}
            self.class_names = sorted(list(pose_classes & obj_classes))
        else:
            self.class_names = sorted(list(pose_classes))
        self.class_to_idx = {c:i for i,c in enumerate(self.class_names)}
        # 建立 windows
        self.windows = []
        for cname in self.class_names:
            pose_cdir = os.path.join(self.pose_root, cname)
            obj_cdir  = os.path.join(self.obj_root, cname) if self.obj_root else None
            seq_paths = _find_sequences_for_class(pose_cdir, obj_cdir, self.use_objects)
            for pose_path, obj_path in seq_paths:
                poses = load_pose_sequence(pose_path)
                objs  = load_object_sequence(obj_path) if (self.use_objects and obj_path) else {}
                frames_pose = set(poses.keys())
                frames_obj  = set(objs.keys()) if self.use_objects else frames_pose
                frames = sorted(list(frames_pose & frames_obj))
                if len(frames) < self.window:
                    continue
                for i in range(0, len(frames) - self.window + 1, self.stride):
                    win_frames = frames[i:i+self.window]
                    if self.require_full_all:
                        if not all(frame_has_full_skeleton(_extract_pose_basic(poses[f])[0], _extract_pose_basic(poses[f])[1],
                                                           kp_need=self.full_kp_min, require_bbox=self.bbox_required)
                                   for f in win_frames):
                            continue
                    elif self.require_full_first:
                        f0 = win_frames[0]
                        bbox0, kps0, _, _ = _extract_pose_basic(poses[f0])
                        if not frame_has_full_skeleton(bbox0, kps0, kp_need=self.full_kp_min, require_bbox=self.bbox_required):
                            continue
                    self.windows.append((self.class_to_idx[cname], win_frames, poses, objs))
        if not self.windows:
            raise RuntimeError("No training windows built. Check data roots.")
        # motion mean/std
        self.motion_stats = self._estimate_motion_stats(n_limit=2000)

    def _estimate_motion_stats(self, n_limit=2000):
        ms = []
        step = max(1, len(self.windows)//max(1, n_limit))
        for i in range(0, len(self.windows), step):
            y, frames, poses, objs = self.windows[i]
            parsed = []
            for fid in frames:
                p = poses[fid]; o = objs.get(fid) if self.use_objects else None
                bbox,kps,iw,ih = _extract_pose_basic(p)
                dets = (o.get("detections") or o.get("objects") or o.get("boxes") or o.get("bboxes") or o.get("predictions") or []) if o else []
                parsed.append((bbox,kps,dets,iw,ih))
            if self.enable_kf:
                window_kps=[kps for (_,kps,_,_,_) in parsed]
                smoothed=kalman_smooth_kps(window_kps, half_slide=self.kf_half, half_len=self.half_len, require_full_first=True, step=self.stride)
                parsed=[(bbox,smoothed[j],d,iw,ih) for j,(bbox,_,d,iw,ih) in enumerate(parsed)]
            M,_=compute_motion_feats_with_mask_from_parsed(parsed)
            if M.shape[0]>0:
                ms.append(M)
        if not ms:
            return {"mean":[0.0]*9, "std":[1.0]*9}
        M=np.concatenate(ms,axis=0)
        mean=M.mean(axis=0); std=M.std(axis=0)+1e-6
        return {"mean": mean.tolist(), "std": std.tolist()}

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        y_idx, frames, poses, objs = self.windows[idx]
        parsed = []
        for fid in frames:
            p = poses[fid]; o = objs.get(fid) if self.use_objects else None
            bbox,kps,iw,ih = _extract_pose_basic(p)
            dets = (o.get("detections") or o.get("objects") or o.get("boxes") or o.get("bboxes") or o.get("predictions") or []) if o else []
            parsed.append((bbox,kps,dets,iw,ih))
        if self.enable_kf:
            window_kps=[kps for (_,kps,_,_,_) in parsed]
            smoothed=kalman_smooth_kps(window_kps, half_slide=self.kf_half, half_len=self.half_len, require_full_first=True, step=self.stride)
            parsed=[(bbox,smoothed[j],d,iw,ih) for j,(bbox,_,d,iw,ih) in enumerate(parsed)]
        # motion + mask（並做 z-score）
        M,mask = compute_motion_feats_with_mask_from_parsed(parsed)
        mm=np.array(self.motion_stats['mean'],np.float32); ss=np.array(self.motion_stats['std'],np.float32)
        if M.size>0: M=(M-mm)/ss
        # rasterize
        Xs=[]
        for (bbox,kps,dets,iw,ih) in parsed:
            Xs.append(rasterize_frame(bbox,kps,dets,iw,ih,self.rm_cfg))
        X = torch.from_numpy(np.stack(Xs)).float()
        M = torch.from_numpy(M).float()
        mask = torch.from_numpy(mask).float()
        y = torch.tensor(y_idx, dtype=torch.long)
        return (X,M,mask), y

# ===================== Model（支援 motion_dim 與 mask） =====================
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
    def __init__(self, in_dim, num_classes, mode="attn", dropout=0.0):
        super().__init__()
        self.mode = mode
        self.drop = nn.Dropout(dropout) if dropout>0 else nn.Identity()
        if mode == "attn":
            self.attn = nn.Linear(in_dim, 1)
        self.fc = nn.Linear(in_dim, num_classes)
    def forward(self, seq_feats: torch.Tensor, mask: Optional[torch.Tensor]=None):
        # seq_feats: (B,T,D); mask: (B,T) in {0,1}
        if (mask is not None) and (self.mode in ("mean","attn")):
            if self.mode == "mean":
                m = mask.unsqueeze(-1)
                den = m.sum(dim=1).clamp_min(1e-6)
                g = (seq_feats * m).sum(dim=1) / den
            else:
                a = self.attn(seq_feats).squeeze(-1)
                a = a.masked_fill((mask<=0), float("-inf"))
                w = torch.softmax(a, dim=1).unsqueeze(-1)
                g = (seq_feats * w).sum(dim=1)
        else:
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
                 bidirectional=False, temporal_pool="attn", dropout=0.3, motion_dim:int=9):
        super().__init__()
        self.cnn = SpaceCNN(in_ch, cnn_out)
        self.motion_dim = int(motion_dim) if motion_dim else 0
        lstm_in = cnn_out + self.motion_dim
        self.lstm = nn.LSTM(lstm_in, lstm_h, lstm_layers, batch_first=True, bidirectional=bidirectional)
        feat_dim = lstm_h * (2 if bidirectional else 1)
        self.head = TemporalHead(feat_dim, num_classes, mode=temporal_pool, dropout=dropout)
    def forward(self, x, motion: Optional[torch.Tensor]=None, mask: Optional[torch.Tensor]=None):
        B,T,C,H,W = x.shape
        z = self.cnn(x.view(B*T, C, H, W)).view(B, T, -1)
        if self.motion_dim>0 and motion is not None:
            z = torch.cat([z, motion], dim=-1)
        out,_ = self.lstm(z)
        return self.head(out, mask=mask)

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

# ===================== Evaluate =====================
def evaluate(model, loader, device, *, use_motion=True):
    model.eval()
    tot, corr = 0, 0
    all_y, all_p = [], []
    with torch.no_grad():
        for (X,M,mask), y in loader:
            X = X.to(device); M = M.to(device); mask = mask.to(device); y = y.to(device)
            out = model(X, motion=(M if use_motion else None), mask=mask)
            pred = out.argmax(1)
            corr += (pred == y).sum().item()
            tot += len(y)
            all_y.append(y.cpu().numpy()); all_p.append(pred.cpu().numpy())
    acc = corr / max(1, tot)
    y_true = np.concatenate(all_y) if all_y else np.array([])
    y_pred = np.concatenate(all_p) if all_p else np.array([])
    mf1 = f1_score(y_true, y_pred, average="macro") if y_true.size else 0.0
    return acc, mf1

# ===================== Train =====================
def build_loaders(cfg: Config, ds: PoseObjectDataset, idx_tr, idx_va):
    tr_subset = Subset(ds, idx_tr)
    va_subset = Subset(ds, idx_va)
    # Weighted sampler（類別平衡）
    labels_tr = [ds.windows[i][0] for i in idx_tr]
    from collections import Counter
    cnt = Counter(labels_tr)
    total_tr = float(len(labels_tr))
    weight_per_class = {c: (total_tr / n) if n>0 else 0.0 for c,n in cnt.items()}
    sample_w = [weight_per_class[y] for y in labels_tr]
    sampler = WeightedRandomSampler(torch.as_tensor(sample_w, dtype=torch.double), num_samples=len(idx_tr), replacement=True) if cfg.use_sampler else None
    dl_tr = DataLoader(tr_subset, batch_size=cfg.batch_size, sampler=sampler, shuffle=(sampler is None), num_workers=4, pin_memory=True)
    dl_va = DataLoader(va_subset, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return dl_tr, dl_va

def train(cfg: Config):
    set_seed(_SEED)
    ensure_dir(cfg.out_dir)
    # Dataset
    ds = PoseObjectDataset(cfg)
    n = len(ds)
    idx = list(range(n)); random.shuffle(idx)
    split = int(n * 0.8)
    idx_tr, idx_va = idx[:split], idx[split:]
    dl_tr, dl_va = build_loaders(cfg, ds, idx_tr, idx_va)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # in_ch 對齊 rasterizer
    in_ch = 2 + 17 + (len(COCO_EDGES) if cfg.include_bone_lines else 0) + (len(ds.rm_cfg.object_classes) if cfg.use_objects else 0) + 2
    model = CNNLSTM(in_ch=in_ch, num_classes=len(ds.class_names), cnn_out=_CNN_OUT,
                    lstm_h=cfg.lstm_hidden, lstm_layers=_LSTM_LAYERS, bidirectional=bool(cfg.bidirectional),
                    temporal_pool=cfg.temporal_pool, dropout=cfg.dropout, motion_dim=9).to(device)

    # Loss
    counts = [0]*len(ds.class_names)
    for i in idx_tr:
        counts[ds.windows[i][0]] += 1
    total = float(sum(max(1,c) for c in counts))
    alpha = torch.tensor([total/max(1,c) for c in counts], dtype=torch.float32)
    alpha = (alpha/alpha.sum()).to(device)
    criterion = nn.CrossEntropyLoss() if cfg.loss=="ce" else FocalLoss(alpha=alpha, gamma=_FOCAL_GAMMA)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=_WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=_AMP and device.type=="cuda")

    # ---- Top-3 索引檔 ----
    TOPK = _SAVE_TOP_K
    topk_list = []
    _topk_index_path = os.path.join(cfg.out_dir, 'topk.json')

    best_score = -1.0
    best_path = None

    for ep in range(1, cfg.epochs+1):
        # ========= Train =========
        model.train(); tot=0; corr=0; loss_sum=0.0
        pbar = tqdm(dl_tr, desc=f"Epoch {ep}/{cfg.epochs} [train]")
        for (X,M,mask), y in pbar:
            X=X.to(device); M=M.to(device); mask=mask.to(device); y=y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda', enabled=scaler.is_enabled()):
                out = model(X, motion=M, mask=mask)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            if _GRAD_CLIP and _GRAD_CLIP>0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), _GRAD_CLIP)
            scaler.step(opt); scaler.update()
            # stats
            loss_sum += float(loss.item()) * len(y)
            corr += (out.argmax(1) == y).sum().item(); tot += len(y)
            pbar.set_postfix({"loss": f"{loss_sum/max(1,tot):.4f}", "acc": f"{corr/max(1,tot):.3f}"})
        tr_loss = loss_sum/max(1,tot); tr_acc = corr/max(1,tot)

        # ========= Valid =========
        acc, mf1 = evaluate(model, dl_va, device, use_motion=True)
        score = mf1  # 仍用 macro-f1 當排序標準
        print(f"[Ep {ep:03d}] tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f} | va_acc={acc:.3f} va_mf1={mf1:.3f}")

        # ========= Save last / best（含 ep 與 motion_norm） =========
        ck = {
            'epoch': ep,
            'model_state': model.state_dict(),
            'optimizer_state': opt.state_dict(),
            'class_names': ds.class_names,
            'motion_norm': ds.motion_stats,
            'config': {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith('__') and not callable(getattr(cfg,k))},
        }
        torch.save(ck, os.path.join(cfg.out_dir, 'last.pt'))

        if score > best_score:
            best_score = score
            best_path = os.path.join(cfg.out_dir, 'best.pt')
            torch.save(ck, best_path)
            # 同步寫入 classes/motion_norm 方便推論端讀取
            with open(os.path.join(cfg.out_dir,'classes.json'),'w',encoding='utf-8') as f:
                json.dump(ds.class_names,f,ensure_ascii=False)
            with open(os.path.join(cfg.out_dir,'motion_norm.json'),'w',encoding='utf-8') as f:
                json.dump(ds.motion_stats,f,ensure_ascii=False)
            print(f"  ↳ saved new best: {best_path}")

        # ========= Top-3（保留前三名 ckpt + 索引） =========
        ep_ck_name = f"ep{ep:03d}_score{score:.4f}.pt"
        ep_path = os.path.join(cfg.out_dir, ep_ck_name)
        try:
            torch.save(ck, ep_path)
        except Exception:
            pass
        topk_list.append({'epoch': int(ep), 'score': float(score), 'path': ep_ck_name})
        topk_list = sorted(topk_list, key=lambda d: d['score'], reverse=True)
        # 移除多餘 ckpt
        for extra in topk_list[TOPK:]:
            try:
                os.remove(os.path.join(cfg.out_dir, extra['path']))
            except Exception:
                pass
        topk_list = topk_list[:TOPK]
        try:
            with open(_topk_index_path, 'w', encoding='utf-8') as f:
                json.dump(topk_list, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        print('  ↳ top3:', ', '.join([f"ep{d['epoch']}:{d['score']:.3f}" for d in topk_list]))

        # ========= Early stop =========
        no_improve = getattr(train, "_no_improve", 0)
        if score > getattr(train, "_best_score_track", -1):
            setattr(train, "_best_score_track", score)
            no_improve = 0
        else:
            no_improve += 1
        setattr(train, "_no_improve", no_improve)
        if _EARLY_STOP_PATIENCE > 0 and no_improve >= _EARLY_STOP_PATIENCE:
            print(f"Early stopping at epoch {ep} (no improvement {no_improve} epochs)")
            break

    # ========= Final report =========
    if best_path and os.path.exists(best_path):
        print("\nLoading best model for final report...")
        ck = torch.load(best_path, map_location='cpu')
        model.load_state_dict(ck['model_state'])
        acc, mf1 = evaluate(model, dl_va, device, use_motion=True)
        print(f"Best model val acc={acc:.3f} macro_f1={mf1:.3f}")
        # 詳細報告
        model.eval(); all_y, all_p = [], []
        with torch.no_grad():
            for (X,M,mask), y in dl_va:
                X=X.to(device); M=M.to(device); mask=mask.to(device)
                out = model(X, motion=M, mask=mask)
                all_p.append(out.argmax(1).cpu().numpy()); all_y.append(y.numpy())
        y_true = np.concatenate(all_y); y_pred = np.concatenate(all_p)
        print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))
        print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=ds.class_names, digits=3))

if __name__ == "__main__":
    cfg = Config()
    train(cfg)
