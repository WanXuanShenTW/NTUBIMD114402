import os, json
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import cv2

from .ws_connection_manager import ws_manager
from .paths import SC_MODELS
from .time_utils import now_str

try:
    from .cnn_lstm_utils import CNNLSTM  # ← 已升級支援 motion + mask
except Exception:
    CNNLSTM = None

# === 動作判斷設定 ===
ABNORMAL_LABELS = {"fall"}
TH_FALL = 0.60
RECOVER_CONSEC = 2
TOTAL_FRAMES: Optional[int] = 60

# === 與 loader 對齊 ===
H, W = 64, 64
WINDOW = 20
STRIDE = 5
INCLUDE_BONE_LINES = True
OBJECT_CLASSES: List[str] = ["bed", "chair"]
COORDCONV_2 = True
LSTM_HIDDEN = 256
BIDIRECTIONAL = False
TEMPORAL_POOL = "attn"
DROPOUT = 0.3
ENABLE_KALMAN = True
KALMAN_HALF_SLIDE = True
HALF_LEN_OVERRIDE = None
REQUIRE_FULL_FIRST = False         # ← 開啟第一幀完整檢查（bbox+17kp）
KP_CONF_TH = 0.2                  # ← 與 trainer 對齊（原本 2.0 太高）
SIGMA_KP = 3.0

COCO_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 目標 FPS 補幀設定 ===
TARGET_FPS = float(os.getenv("SC_TARGET_FPS", "0"))  # 0 = 不補幀
UPSAMPLE_MAX_MULT = int(os.getenv("SC_UPSAMPLE_MAX_MULT", "4"))

# === 二階段推論設定（可用環境變數覆蓋） ===
USE_TWO_STAGE = True  # 關掉就回到舊單模型流程

# Binary（跌倒/非跌倒）
BINARY_MODEL_PATH   = os.getenv("SC_BINARY_MODEL_PATH", f"{SC_MODELS}/binary/best.pt")
BINARY_CLASSES_PATH = os.getenv("SC_BINARY_CLASSES_PATH", f"{SC_MODELS}/binary/classes.json")
BINARY_POS_NAME     = os.getenv("SC_BINARY_POS_NAME", "fall")
BINARY_THR          = float(os.getenv("SC_BINARY_THR", "0.50"))  # binary 判定 fall 的 gate

# Multi（多類別動作）
MULTI_MODEL_PATH    = os.getenv("SC_MULTI_MODEL_PATH",  f"{SC_MODELS}/multi/best.pt")
MULTI_CLASSES_PATH  = os.getenv("SC_MULTI_CLASSES_PATH",f"{SC_MODELS}/multi/classes.json")

# 舊單模型相容
MODEL_PATH   = os.getenv("SC_MODEL_PATH",   f"{SC_MODELS}/best.pt")
CLASSES_PATH = os.getenv("SC_CLASSES_PATH", f"{SC_MODELS}/classes.json")

# === 跌倒事件門檻 ===
FALL_TRIGGER_THR = float(os.getenv("SC_FALL_TRIGGER_THR", "0.60"))
FALL_RECOVER_THR = float(os.getenv("SC_FALL_RECOVER_THR", "0.40"))
TRIGGER_CONSEC   = int(os.getenv("SC_FALL_TRIGGER_CONSEC", "2"))
RECOVER_CONSEC   = int(os.getenv("SC_FALL_RECOVER_CONSEC", "2"))

# === 坐/躺事件規則（內嵌預設；可用 SC_ACTION_EVENTS_JSON 整包覆蓋） ===
ACTION_EVENTS_DEFAULT = {
    "sit": {
        "pos_labels":     ["lie","locomotion"],
        "recover_labels": ["lie","locomotion"],
        "trigger_thr":    float(os.getenv("SC_SIT_TRIGGER_THR", "0.60")),
        "recover_thr":    float(os.getenv("SC_SIT_RECOVER_THR", "0.40")),
        "trigger_consec": int(os.getenv("SC_SIT_TRIGGER_CONSEC", "2")),
        "recover_consec": int(os.getenv("SC_SIT_RECOVER_CONSEC", "2"))
    },
    "lie": {
        "pos_labels":     ["sit", "locomotion"],
        "recover_labels": ["sit", "locomotion"],
        "trigger_thr":    float(os.getenv("SC_LIE_TRIGGER_THR", "0.60")),
        "recover_thr":    float(os.getenv("SC_LIE_RECOVER_THR", "0.40")),
        "trigger_consec": int(os.getenv("SC_LIE_TRIGGER_CONSEC", "2")),
        "recover_consec": int(os.getenv("SC_LIE_RECOVER_CONSEC", "2"))
    }
}
ACTION_EVENTS = ACTION_EVENTS_DEFAULT

# ===== Relation Map & 工具 =====
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

def _bbox_xyxy_from_cxcywh(bb):
    if not bb: return None
    cx, cy, w, h = bb.get("cx"), bb.get("cy"), bb.get("w"), bb.get("h")
    if None in (cx, cy, w, h): return None
    x1 = cx - w/2.0; x2 = cx + w/2.0
    y1 = cy - h/2.0; y2 = cy + h/2.0
    return float(x1), float(y1), float(x2), float(y2)

def rasterize_frame(bbox, kps_list, dets, img_w, img_h, cfg: RelationMapConfig):
    Hc, Wc = cfg.H, cfg.W
    num_kp = 17
    num_edges = len(COCO_EDGES) if cfg.include_bone_lines else 0
    num_obj_ch = len(cfg.object_classes)
    num_coord = 2 if COORDCONV_2 else 0
    C = 2 + num_kp + num_edges + num_obj_ch + num_coord
    canvas = np.zeros((C, Hc, Wc), dtype=np.float32)
    ch = 0
    # 1) bbox mask
    if bbox:
        x1,y1,x2,y2 = bbox
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
                if conf < KP_CONF_TH:
                    continue
                x = (float(kp['x'])/img_w)*Wc; y = (float(kp['y'])/img_h)*Hc
                draw_gaussian(canvas[ch+i], x, y, SIGMA_KP, mag=conf)
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
            if min(c1,c2) < KP_CONF_TH: continue
            cv2.line(canvas[ch+e_idx], (x1,y1), (x2,y2), 1.0, 1)
    ch += num_edges
    # 5) objects
    if num_obj_ch > 0 and dets:
        cls2ch = {name: i for i, name in enumerate(cfg.object_classes)}
        for d in dets:
            name = d.get("cls_name") or ""
            if name not in cls2ch: 
                continue
            bb = d.get("bbox") or {}
            cx, cy, bw, bh = bb.get("cx"), bb.get("cy"), bb.get("w"), bb.get("h")
            if None in (cx,cy,bw,bh): 
                continue
            x1 = (cx - bw/2.0); x2 = (cx + bw/2.0)
            y1 = (cy - bh/2.0); y2 = (cy + bh/2.0)
            x1 = int(np.clip((x1/img_w)*Wc, 0, Wc-1)); x2 = int(np.clip((x2/img_w)*Wc, 0, Wc-1))
            y1 = int(np.clip((y1/img_h)*Hc, 0, Hc-1)); y2 = int(np.clip((y2/img_h)*Hc, 0, Hc-1))
            if x2>=x1 and y2>=y1:
                canvas[ch+cls2ch[name], y1:y2+1, x1:x2+1] = 1.0
    ch += num_obj_ch
    # 6) CoordConv（x/y）
    if COORDCONV_2:
        xv = np.linspace(-1, 1, Wc)[None, :].repeat(Hc, 0)
        yv = np.linspace(-1, 1, Hc)[:, None].repeat(Wc, 1)
        canvas[ch] = xv; canvas[ch+1] = yv; ch += 2
    return canvas

# ===== 骨架完整性（第一幀用） =====

def frame_has_full_skeleton(bbox, kps, *, kp_need=17, require_bbox=True):
    if require_bbox and not bbox:
        return False
    if not kps:
        return False
    cnt = 0
    for j in range(min(17, len(kps))):
        if ('x' in kps[j]) and ('y' in kps[j]):
            conf = float(kps[j].get('conf', kps[j].get('confidence', 1.0)))
            if conf >= KP_CONF_TH:
                cnt += 1
    return cnt >= kp_need

# ===== Kalman（半視窗） =====
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
                if conf >= KP_CONF_TH:
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

def _estimate_fps(ts_list_ms: List[int]) -> float:
    """用相鄰 timestamp 中位數估計輸入 FPS（較抗噪）"""
    if not ts_list_ms or len(ts_list_ms) < 2:
        return 0.0
    dts = [max(1.0, float(ts_list_ms[i] - ts_list_ms[i-1])) for i in range(1, len(ts_list_ms))]
    dts.sort()
    med = dts[len(dts)//2]
    return 1000.0 / med if med > 0 else 0.0

def _normalize_in_fps(fps: float) -> float:
    """你指定的規則：4.x fps 一律視為 5 fps"""
    if 4.0 <= fps < 5.0:
        return 5.0
    return fps

def _interp_val(a: Optional[float], b: Optional[float], w: float) -> Optional[float]:
    if a is None and b is None: return None
    if a is None: return b
    if b is None: return a
    return a*(1.0-w) + b*w

def _interp_kp(kpa: dict, kpb: dict, w: float) -> dict:
    ax = kpa.get('x'); ay = kpa.get('y'); ac = float(kpa.get('conf', kpa.get('confidence', 1.0)))
    bx = kpb.get('x'); by = kpb.get('y'); bc = float(kpb.get('conf', kpb.get('confidence', 1.0)))
    x = _interp_val(ax, bx, w); y = _interp_val(ay, by, w)
    c = ac*(1.0-w) + bc*w
    out = {}
    if x is not None: out['x'] = float(x)
    if y is not None: out['y'] = float(y)
    out['conf'] = float(c)
    return out

def _interp_bbox_xyxy(bba: Optional[Tuple[float,float,float,float]],
                      bbb: Optional[Tuple[float,float,float,float]],
                      w: float) -> Optional[Tuple[float,float,float,float]]:
    """FrameRecord.bbox 是 xyxy tuple，線性插值後回傳 xyxy"""
    if not bba and not bbb:
        return None
    if not bba: return bbb
    if not bbb: return bba
    x1a,y1a,x2a,y2a = bba
    x1b,y1b,x2b,y2b = bbb
    return (
        _interp_val(x1a, x1b, w),
        _interp_val(y1a, y1b, w),
        _interp_val(x2a, x2b, w),
        _interp_val(y2a, y2b, w),
    )

def _densify_sequences(kps_seq: List[List[dict]],
                       bbox_seq: List[Optional[Tuple[float,float,float,float]]],
                       dets_seq: List[List[dict]],
                       mult: int):
    """
    針對相鄰幀做等距插值，mult=1 表示不變；mult=2 表示每對幀中插 1 幀，以此類推
    dets 使用左值持有（物件可慢到）
    回傳：kps_dense, bbox_dense, dets_dense
    """
    if mult <= 1 or len(kps_seq) <= 1:
        return kps_seq, bbox_seq, dets_seq

    J = max(len(kps_seq[0]) if kps_seq and kps_seq[0] else 17, 17)
    kd, bd, dd = [], [], []
    N = len(kps_seq)
    for i in range(N-1):
        A_kps, B_kps = kps_seq[i] or [], kps_seq[i+1] or []
        A_bbox, B_bbox = bbox_seq[i], bbox_seq[i+1]
        A_dets = dets_seq[i] if dets_seq[i] is not None else []
        kd.append(A_kps)
        bd.append(A_bbox)
        dd.append(A_dets)
        # 插入中間幀
        for k in range(1, mult):
            w = k / float(mult)  # 0<w<1
            inter_kps = []
            for j in range(J):
                a = A_kps[j] if j < len(A_kps) else {}
                b = B_kps[j] if j < len(B_kps) else {}
                inter_kps.append(_interp_kp(a, b, w))
            kd.append(inter_kps)
            bd.append(_interp_bbox_xyxy(A_bbox, B_bbox, w))
            dd.append(A_dets)  # dets：零階保持

    # 末幀
    kd.append(kps_seq[-1] or [])
    bd.append(bbox_seq[-1])
    dd.append(dets_seq[-1] if dets_seq[-1] is not None else [])
    return kd, bd, dd

def _virtual_window_from(bbox_seq: List[Optional[Tuple[float,float,float,float]]],
                         dets_seq: List[List[dict]],
                         template_fr: 'FrameRecord') -> List['FrameRecord']:
    """產生與 bbox_seq 同長度的虛擬 window（img_w/img_h/bbox/dets 來自參數；frame_id/ts 用樣板）"""
    vwin = []
    for i in range(len(bbox_seq)):
        vwin.append(FrameRecord(
            frame_id=template_fr.frame_id,
            ts_ms=template_fr.ts_ms,
            img_w=template_fr.img_w,
            img_h=template_fr.img_h,
            kps=None,
            bbox=bbox_seq[i],
            dets=dets_seq[i] or []
        ))
    return vwin

def _resample_sequences_to_fps(window, use_kps, use_bbox, use_dets, target_fps):
    """
    以「時間軸重採樣」的方式，回傳剛好 WINDOW 長度、等間距(=1000/target_fps ms)的序列。
    - keypoints / bbox：線性插值
    - dets：左值保持（較慢沒關係）
    - 虛擬 window：只帶 img_w/img_h/bbox/dets 給後續 motion/mask / rasterize 使用
    """
    if not window or len(window) < 2:
        # 樣本太少，直接回傳原序列
        return use_kps, use_bbox, use_dets, window

    # 目標時間格點：以最後一幀時間為對齊點
    dt = 1000.0 / float(target_fps)
    t_end = float(window[-1].ts_ms)
    t_grid = [t_end - dt * (WINDOW - 1 - i) for i in range(WINDOW)]  # 長度=WINDOW

    # 原時間序列
    ts = [float(fr.ts_ms) for fr in window]
    n = len(ts)

    res_kps, res_bbox, res_dets = [], [], []

    j = 0  # 游標（單調前進）
    for t in t_grid:
        # 往前找到 ts[j] <= t <= ts[j+1]
        while j + 1 < n and ts[j + 1] < t:
            j += 1

        if t <= ts[0]:
            # 落在最前，直接取第 0 幀
            kps_a, bbox_a, dets_a = use_kps[0], use_bbox[0], (use_dets[0] or [])
            res_kps.append(kps_a)
            res_bbox.append(bbox_a)
            res_dets.append(dets_a)
            continue

        if t >= ts[-1]:
            # 落在最後，直接取最後一幀
            kps_b, bbox_b, dets_b = use_kps[-1], use_bbox[-1], (use_dets[-1] or [])
            res_kps.append(kps_b)
            res_bbox.append(bbox_b)
            res_dets.append(dets_b)
            continue

        # 正常情況：介於 j 與 j+1 之間
        t0, t1 = ts[j], ts[j + 1]
        # 保險起見
        if t1 <= t0:
            w = 0.0
        else:
            w = (t - t0) / (t1 - t0)

        # keypoints 插值
        J = max(len(use_kps[j]) if use_kps[j] else 17, 17)
        inter_kps = []
        for idx in range(J):
            a = use_kps[j][idx] if idx < len(use_kps[j]) else {}
            b = use_kps[j + 1][idx] if idx < len(use_kps[j + 1]) else {}
            inter_kps.append(_interp_kp(a, b, w))
        res_kps.append(inter_kps)

        # bbox 插值（xyxy）
        res_bbox.append(_interp_bbox_xyxy(use_bbox[j], use_bbox[j + 1], w))

        # dets：採左值保持
        res_dets.append(use_dets[j] or [])

    # 產生對齊長度的虛擬 window（img_w/img_h/bbox/dets）
    vwin = _virtual_window_from(res_bbox, res_dets, template_fr=window[-1])
    return res_kps, res_bbox, res_dets, vwin

# ===== Motion + Mask =====
import math

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
            return (float(d["x"])/img_w, float(d["y"])/img_h)
    return None

def compute_motion_feats_with_mask_from_window(window, kps_seq, conf_th=KP_CONF_TH):
    """
    window: List[FrameRecord]
    kps_seq: 平滑後的 kps 對應 window 順序
    回傳 (motion: (T,9), mask: (T,))
    """
    T = len(window)
    if T == 0:
        return np.zeros((0,9), np.float32), np.zeros((0,), np.float32)
    ycom, hgt, area, trunk, kneeL, kneeR = [], [], [], [], [], []
    for i, fr in enumerate(window):
        bbox = fr.bbox
        kps  = kps_seq[i]
        img_w, img_h = fr.img_w, fr.img_h
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
                ys = [float(p["y"])/img_h for p in kps if float(p.get("conf", p.get("confidence",1.0)))>=conf_th and ("y" in p)]
                y_c = _safe_mean(ys)
        ycom.append(y_c)
        if bbox is not None:
            x1,y1,x2,y2 = bbox
            bw = max(1e-6, (x2-x1)/img_w); bh = max(1e-6, (y2-y1)/img_h)
            h = bh; w = bw
        else:
            ys = [float(p["y"])/img_h for p in kps if float(p.get("conf", p.get("confidence",1.0)))>=conf_th and ("y" in p)]
            if len(ys)>=2:
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
    for fr, kps in zip(window, kps_seq):
        cnt=0
        for j in (11,12,5,6,13,14):
            if j < len(kps):
                conf=float(kps[j].get("conf", kps[j].get("confidence",1.0)))
                if conf>=KP_CONF_TH: cnt+=1
        ok=(cnt>=2) or (fr.bbox is not None)
        valid.append(1.0 if ok else 0.0)
    valid=np.array(valid,np.float32)
    return feats, valid

# ===== Buffer / Manager =====
@dataclass
class FrameRecord:
    frame_id: int
    ts_ms: int
    img_w: float
    img_h: float
    kps: Optional[List[dict]] = None
    bbox: Optional[Tuple[float,float,float,float]] = None
    dets: List[dict] = field(default_factory=list)

class UserBuffer:
    def __init__(self, total_frames: Optional[int] = 60):
        self.frames: List[FrameRecord] = []
        self.total_frames = int(total_frames) if total_frames else None
        self._wrap_anchor: Optional[int] = None
    def _maybe_update_anchor(self, fid: int):
        if not self.total_frames or self._wrap_anchor is not None:
            return
        last_third_start = (2 * self.total_frames) // 3
        if fid > last_third_start:
            self._wrap_anchor = fid
    def _order_key(self, fid: int) -> int:
        if not self.total_frames or self._wrap_anchor is None:
            return fid
        n = self.total_frames
        return (fid - self._wrap_anchor) % n
    def upsert_pose(self, frame_id: int, ts_ms: int, img_w: float, img_h: float, persons: List[dict]):
        person = max(persons, key=lambda x: x.get("score",0.0)) if persons else None
        bbox = None; kps = None
        if person:
            bbox_xyxy = _bbox_xyxy_from_cxcywh(person.get("bbox") or {})
            if bbox_xyxy is not None:
                bbox = bbox_xyxy
            raw = person.get("keypoints") or []
            kps = []
            for kp in raw[:17]:
                if isinstance(kp, dict):
                    kps.append({"x": float(kp.get("x",0.0)), "y": float(kp.get("y",0.0)), "conf": float(kp.get("conf", kp.get("confidence",1.0)))})
                elif isinstance(kp, (list,tuple)) and len(kp)>=2:
                    kps.append({"x": float(kp[0]), "y": float(kp[1]), "conf": 1.0})
        self._merge(frame_id, ts_ms, img_w, img_h, bbox=bbox, kps=kps)
    def upsert_objects(self, frame_id: int, ts_ms: int, img_w: float, img_h: float, detections: List[dict]):
        dets = []
        for d in detections or []:
            bb = d.get("bbox") or {}
            dets.append({
                "cls_name": d.get("cls_name"),
                "score": float(d.get("score", 1.0)),
                "bbox": {"cx": bb.get("cx"), "cy": bb.get("cy"), "w": bb.get("w"), "h": bb.get("h")}
            })
        self._merge(frame_id, ts_ms, img_w, img_h, dets=dets)
    def _merge(self, frame_id: int, ts_ms: int, img_w: float, img_h: float,
               bbox: Optional[Tuple[float,float,float,float]]=None,
               kps: Optional[List[dict]] = None,
               dets: Optional[List[dict]] = None):
        self._maybe_update_anchor(frame_id)
        for fr in self.frames:
            if fr.frame_id == frame_id:
                fr.ts_ms = max(fr.ts_ms, ts_ms)
                fr.img_w = img_w; fr.img_h = img_h
                if bbox is not None: fr.bbox = bbox
                if kps  is not None: fr.kps  = kps
                if dets is not None and len(dets)>0: fr.dets = dets
                break
        else:
            self.frames.append(FrameRecord(frame_id=frame_id, ts_ms=ts_ms, img_w=img_w, img_h=img_h,
                                           kps=kps, bbox=bbox, dets=dets or []))
            self.frames.sort(key=lambda x: self._order_key(x.frame_id))
    def pop_left(self, n: int):
        self.frames = self.frames[n:] if n > 0 else self.frames
        if not self.frames:
            self._wrap_anchor = None
    def ready_window(self, require_full_first: bool = True) -> Optional[List[FrameRecord]]:
        if len(self.frames) < WINDOW:
            return None
        window = self.frames[:WINDOW]
        if require_full_first and REQUIRE_FULL_FIRST and not frame_has_full_skeleton(window[0].bbox, window[0].kps, require_bbox=True):
            self.pop_left(1)
            print(f"[InferManager] WARN: First frame incomplete, discard and wait for next.")
            return None
        return window

class StreamInferManager:
    """改良版：
    - 仍使用『獨立快取 smooth_cache』儲存回填，不覆寫 raw。
    - 使用『最終定稿(finalized)』概念：只對已完成回填(定稿)的幀使用平滑值，否則用 raw。
    - 『中心幀決策』：事件觸發/恢復只看每窗中心幀；且只有當中心幀已定稿才會寫入決策時間線，避免滑窗擴散延長。
    - 仍保留原本 per-window 的推論訊息（方便 UI 顯示熱度），但事件判定用 centerline 更穩。
    """
    def __init__(self):
        coord = 2 if COORDCONV_2 else 0
        self.in_ch = 2 + 17 + (len(COCO_EDGES) if INCLUDE_BONE_LINES else 0) + (len(OBJECT_CLASSES) if OBJECT_CLASSES else 0) + coord
        
        # 讀 classes
        self.class_names_bin   = ["non-fall", "fall"]
        self.class_names_multi = ["class_0", "class_1"]
        try:
            with open(BINARY_CLASSES_PATH, "r", encoding="utf-8") as f:
                self.class_names_bin = json.load(f)
        except Exception:
            pass
        try:
            with open(MULTI_CLASSES_PATH, "r", encoding="utf-8") as f:
                self.class_names_multi = json.load(f)
        except Exception:
            pass

        # （保留你原本 self.in_ch 的計算）

        if USE_TWO_STAGE:
            self.model_bin   = self._build_model(num_classes=len(self.class_names_bin))
            self.model_multi = self._build_model(num_classes=len(self.class_names_multi))
            try:
                self._load_weights(self.model_bin,   BINARY_MODEL_PATH)
                self._load_weights(self.model_multi, MULTI_MODEL_PATH)
                print(f"[InferManager] Two-stage loaded. bin={len(self.class_names_bin)}, multi={len(self.class_names_multi)}")
            except Exception as e:
                print(f"[InferManager] WARN two-stage load failed: {e}")
            self.model_bin.eval().to(DEVICE)
            self.model_multi.eval().to(DEVICE)
        else:
            try:
                with open(CLASSES_PATH, "r", encoding="utf-8") as f:
                    self.class_names_multi = json.load(f)
            except Exception:
                self.class_names_multi = ["class_0", "class_1"]

            self.num_classes = len(self.class_names_multi)  

            self.model_multi = self._load_model()
            self.model_multi.eval().to(DEVICE)

        # 多事件（坐/躺）狀態表 + 依 multi 類別編譯事件規則
        self._multi_event_state = {}  # { user_id: { event_name: {...} } }
        self._event_defs = self._compile_action_events(ACTION_EVENTS, self.class_names_multi)
        self._last_action_name: Dict[str, Optional[str]] = {}  # {user_id: 上一次 multi 的動作名稱}
        self.cfg = RelationMapConfig(H=H, W=W, sigma_kp=SIGMA_KP, kp_conf_th=KP_CONF_TH,
                                     include_bone_lines=INCLUDE_BONE_LINES, object_classes=OBJECT_CLASSES)
        # user 狀態
        self._buffers: Dict[str, UserBuffer] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._handlers = {}
        self._fall_state = {}
        # 平滑資料：獨立快取 + 定稿集合
        self._smooth_cache: Dict[str, Dict[int, List[dict]]] = {}      # user_id -> {frame_id: kps_smoothed}
        self._smooth_finalized: Dict[str, set] = {}                    # user_id -> {frame_id}
        # 中心幀時間線（事件判定只看中心幀）
        self._centerline: Dict[str, Dict[int, float]] = {}             # user_id -> {center_frame_id: p_fall}

    def _load_model(self) -> nn.Module:
        try:
            state = torch.load(MODEL_PATH, map_location=DEVICE)
            if CNNLSTM is not None:
                m = CNNLSTM(
                    in_ch=self.in_ch,
                    num_classes=self.num_classes,
                    cnn_out=256,
                    lstm_h=LSTM_HIDDEN,
                    lstm_layers=2,
                    bidirectional=BIDIRECTIONAL,
                    temporal_pool=TEMPORAL_POOL,
                    dropout=DROPOUT,
                    motion_dim=9,
                )
            else:
                class _LiteCNNLSTM(nn.Module):
                    def __init__(self, in_ch, num_classes):
                        super().__init__()
                        self.c1 = nn.Sequential(
                            nn.Conv2d(in_ch,64,3,padding=1), nn.ReLU(),
                            nn.Conv2d(64,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
                            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
                            nn.Conv2d(128,256,3,padding=1), nn.ReLU(),
                            nn.AdaptiveAvgPool2d(1)
                        )
                        self.p = nn.Linear(256,256)
                        self.lstm = nn.LSTM(256, LSTM_HIDDEN, num_layers=2, batch_first=True, bidirectional=BIDIRECTIONAL)
                        feat = LSTM_HIDDEN * (2 if BIDIRECTIONAL else 1)
                        self.fc = nn.Linear(feat, num_classes)
                    def forward(self, x):
                        B,T,C,H,W = x.shape
                        z = self.p(self.c1(x.view(B*T,C,H,W)).flatten(1)).view(B,T,-1)
                        z,_ = self.lstm(z)
                        return self.fc(z[:,-1])
                m = _LiteCNNLSTM(self.in_ch, self.num_classes)
            if isinstance(state, dict) and "state_dict" in state:
                m.load_state_dict(state["state_dict"], strict=False)
            elif isinstance(state, dict):
                m.load_state_dict(state, strict=False)
            else:
                m = state
            print(f"[InferManager] Loaded weights: in_ch={self.in_ch}, classes={self.num_classes}")
            return m
        except Exception as e:
            print(f"[InferManager] WARN: Load model failed: {e}")
            class _SimpleSpaceCNN(nn.Module):
                def __init__(self, in_ch=32, out_dim=256):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Conv2d(in_ch, 64, 3, 1, 1), nn.ReLU(inplace=True),
                        nn.MaxPool2d(2),
                        nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(inplace=True),
                        nn.MaxPool2d(2),
                        nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(inplace=True),
                        nn.AdaptiveAvgPool2d((1,1))
                    )
                    self.proj = nn.Linear(256, out_dim)
                def forward(self, x):
                    z = self.net(x).flatten(1)
                    return self.proj(z)
            class _SimpleCNNLSTM(nn.Module):
                def __init__(self, in_ch, num_classes, feat=256, hid=256):
                    super().__init__()
                    self.cnn = _SimpleSpaceCNN(in_ch, feat)
                    self.lstm = nn.LSTM(feat, hid, num_layers=1, batch_first=True)
                    self.fc = nn.Linear(hid, num_classes)
                def forward(self, x):
                    B,T,C,Hh,Ww = x.shape
                    z = self.cnn(x.view(B*T, C, Hh, Ww)).view(B, T, -1)
                    z,_ = self.lstm(z)
                    return self.fc(z[:, -1])
            return _SimpleCNNLSTM(in_ch=self.in_ch, num_classes=self.num_classes)

    def _buf(self, user_id: str) -> 'UserBuffer':
        return self._buffers.setdefault(user_id, UserBuffer(total_frames=TOTAL_FRAMES))

    def _lock(self, user_id: str) -> asyncio.Lock:
        if user_id not in self._locks:
            self._locks[user_id] = asyncio.Lock()
        return self._locks[user_id]

    def set_handlers(self, **handlers):
        self._handlers.update({k: v for k, v in handlers.items() if v})

    def _emit(self, name: str, *args, **kwargs):
        h = self._handlers.get(name)
        if h:
            asyncio.create_task(h(*args, **kwargs))

    def _build_model(self, num_classes: int) -> nn.Module:
        if CNNLSTM is not None:
            m = CNNLSTM(
                in_ch=self.in_ch,
                num_classes=num_classes,
                cnn_out=256,
                lstm_h=LSTM_HIDDEN,
                lstm_layers=2,
                bidirectional=BIDIRECTIONAL,
                temporal_pool=TEMPORAL_POOL,
                dropout=DROPOUT,
                motion_dim=9,
            )
        else:
            class _LiteCNNLSTM(nn.Module):
                def __init__(self, in_ch, num_classes):
                    super().__init__()
                    self.c1 = nn.Sequential(
                        nn.Conv2d(in_ch,64,3,padding=1), nn.ReLU(),
                        nn.Conv2d(64,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
                        nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
                        nn.Conv2d(128,256,3,padding=1), nn.ReLU(),
                        nn.AdaptiveAvgPool2d(1)
                    )
                    self.p = nn.Linear(256,256)
                    self.lstm = nn.LSTM(256, LSTM_HIDDEN, num_layers=2, batch_first=True, bidirectional=BIDIRECTIONAL)
                    feat = LSTM_HIDDEN * (2 if BIDIRECTIONAL else 1)
                    self.fc = nn.Linear(feat, num_classes)
                def forward(self, x):
                    B,T,C,H,W = x.shape
                    z = self.p(self.c1(x.view(B*T,C,H,W)).flatten(1)).view(B,T,-1)
                    z,_ = self.lstm(z)
                    return self.fc(z[:,-1])
            m = _LiteCNNLSTM(self.in_ch, num_classes)
        return m

    def _load_weights(self, model: nn.Module, weight_path: str):
        state = torch.load(weight_path, map_location=DEVICE)
        if isinstance(state, dict) and "state_dict" in state:
            model.load_state_dict(state["state_dict"], strict=False)
        elif isinstance(state, dict):
            model.load_state_dict(state, strict=False)

    def _compile_action_events(self, cfg: dict, class_names: list):
        name_to_idx = {n: i for i, n in enumerate(class_names)}
        event_defs = {}
        for ev, econf in cfg.items():
            pos_idx = [name_to_idx[x] for x in econf.get("pos_labels", []) if x in name_to_idx]
            rec_idx = [name_to_idx[x] for x in econf.get("recover_labels", []) if x in name_to_idx]
            event_defs[ev] = {
                "pos_idx": pos_idx,
                "rec_idx": rec_idx,
                "trigger_thr":    float(econf.get("trigger_thr", 0.80)),
                "recover_thr":    float(econf.get("recover_thr", 0.60)),
                "trigger_consec": int(econf.get("trigger_consec", 2)),
                "recover_consec": int(econf.get("recover_consec", 2)),
            }
            if not pos_idx:
                print(f"[InferManager] WARN: event '{ev}' has no valid pos_labels.")
            if not rec_idx:
                print(f"[InferManager] WARN: event '{ev}' has no valid recover_labels.")
        return event_defs

    @staticmethod
    def _group_prob(indices: list, probs: list) -> float:
        if not indices or not probs: return 0.0
        return float(max((probs[i] for i in indices if 0 <= i < len(probs)), default=0.0))

    def _update_multi_events(self, user_id: str, center_fid: int, probs: list, result_payload: dict,
                         prev_action_name: Optional[str] = None, curr_action_name: Optional[str] = None):
        user_map = self._multi_event_state.setdefault(user_id, {})
        now = now_str()

        for ev_name, evdef in self._event_defs.items():
            st = user_map.setdefault(ev_name, {
                "active": False, "start_time": None, "start_frame": None,
                "peak_score": 0.0, "pos_streak": 0, "rec_streak": 0
            })
            pos_p = self._group_prob(evdef["pos_idx"], probs)
            rec_p = self._group_prob(evdef["rec_idx"], probs)

            if not st["active"]:
                if pos_p >= evdef["trigger_thr"]:
                    st["pos_streak"] += 1
                    st["peak_score"] = max(st["peak_score"], pos_p)
                else:
                    st["pos_streak"] = 0
                    st["rec_streak"] = 0
                if st["pos_streak"] >= evdef["trigger_consec"]:
                    st.update({"active": True, "start_time": now, "start_frame": center_fid})
                    self._emit("on_state_event_start",
                            user_id, ev_name, st["start_time"], float(st["peak_score"]),
                            prev_action_name, curr_action_name, result_payload)
            else:
                if rec_p >= evdef["recover_thr"]:
                    st["rec_streak"] += 1
                else:
                    st["rec_streak"] = 0
                if st["rec_streak"] >= evdef["recover_consec"]:
                    end_time = now; end_frame = center_fid; peak = float(st["peak_score"])
                    self._emit("on_state_event_recover",
                            user_id, ev_name, st["start_time"], end_time, peak,
                            prev_action_name, curr_action_name, result_payload)
                    st.update({"active": False, "start_time": None, "start_frame": None,
                            "peak_score": 0.0, "pos_streak": 0, "rec_streak": 0})


    # ---- 事件決策（只看中心幀） ----
    @staticmethod
    def _consecutive(xs: List[bool], need: int) -> bool:
        if need <= 1:
            return any(xs)
        c = 0
        for v in xs:
            c = c + 1 if v else 0
            if c >= need:
                return True
        return False

    async def force_recover(self, user_id: str, reason: str = "manual"):
        st = self._fall_state.get(user_id)
        if not st or not st.get("active"):
            return
        start_time: Optional[str] = st.get("start_time")
        start_frame: Optional[int] = st.get("start_frame")
        end_frame: Optional[int] = st.get("last_end_frame")
        peak_score: float = float(st.get("peak_score", 0.0))
        end_time = now_str()
        result_stub = {"forced_reason": reason, "pred": "fall"}
        if self._handlers.get("on_fall_recover"):
            await self._handlers["on_fall_recover"](
                user_id, start_time, end_time, start_frame, end_frame, peak_score, result_stub
            )
        st.update({"active": False, "start_time": None, "start_frame": None, "peak_score": 0.0, "normal_streak": 0, "last_end_frame": None})

    async def force_recover_all(self, reason: str = "shutdown"):
        tasks = []
        for uid, st in list(self._fall_state.items()):
            if st and st.get("active"):
                tasks.append(self.force_recover(uid, reason=reason))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def ingest(self, user_id: str, msg: dict):
        async with self._lock(user_id):
            t = (msg.get("type") or "frame").lower()  # 預設當 frame（容錯）
            fid = int(msg.get("frame_id", -1))
            ts = int(msg.get("timestamp_ms", 0))
            img_w = float(((msg.get("image_size") or {}).get("width") or 640))
            img_h = float(((msg.get("image_size") or {}).get("height") or 480))
            buf = self._buf(user_id)

            # 新鍵名優先（pose / detect），保留舊鍵名相容（persons / detections）
            pose   = msg.get("pose")    or msg.get("persons")
            detect = msg.get("detect")  or msg.get("detections")

            # 合併包：同一訊息可能帶 pose 與/或 detect（detect 可省略）
            if t == "frame" or (t not in ("pose","object","detect") and (pose is not None or detect is not None)):
                if pose   is not None: buf.upsert_pose(fid, ts, img_w, img_h, pose)
                if detect is not None: buf.upsert_objects(fid, ts, img_w, img_h, detect)
            # 分開包仍相容
            elif t == "pose":
                buf.upsert_pose(fid, ts, img_w, img_h, pose or [])
            elif t in ("object", "detect"):
                buf.upsert_objects(fid, ts, img_w, img_h, detect or [])
            else:
                return

            window = buf.ready_window()
            if window is None:
                return

            # 1) 取得該 user 的平滑快取與定稿集合
            user_cache = self._smooth_cache.setdefault(user_id, {})
            finalized = self._smooth_finalized.setdefault(user_id, set())

            # 2) 本窗 raw kps 與『以定稿優先』的 kps
            kps_seq_raw = [fr.kps for fr in window]
            kps_seq = [ (user_cache[fr.frame_id] if fr.frame_id in finalized else fr.kps) for fr in window ]

            # 3) 執行半視窗 KF 但『不覆寫 buffer』，僅寫入 cache，且只寫一次
            if ENABLE_KALMAN:
                half_len = int(HALF_LEN_OVERRIDE) if HALF_LEN_OVERRIDE else max(1, WINDOW//2)
                smoothed_full = kalman_smooth_kps(
                    kps_seq_raw,
                    half_slide=KALMAN_HALF_SLIDE,
                    half_len=half_len,
                    require_full_first=True,
                    step=STRIDE
                )
                # 只寫本窗尾端 STRIDE 幀到 cache，且只寫一次（定稿）
                write_range = range(WINDOW - STRIDE, WINDOW)
                for idx in write_range:
                    fid_tail = window[idx].frame_id
                    if fid_tail not in finalized:
                        user_cache[fid_tail] = smoothed_full[idx]
                        finalized.add(fid_tail)
                # 更新使用序列（定稿優先）
                kps_seq = [ (user_cache[fr.frame_id] if fr.frame_id in finalized else fr.kps) for fr in window ]

            # === (新) 依目標 FPS 做上采樣（只用 pose；dets 用左值持有） ===
            use_kps  = kps_seq
            use_bbox = [fr.bbox for fr in window]   # fr.bbox 是 xyxy tuple
            use_dets = [fr.dets for fr in window]
            win_for_feats = window                  # 給 motion/mask 用的對齊 window

            if TARGET_FPS > 0:
                in_fps = _estimate_fps([fr.ts_ms for fr in window])
                in_fps = _normalize_in_fps(in_fps)

                # 只在「目標 fps 明顯高於輸入 fps」時才重採樣，避免沒必要的運算
                if in_fps > 0 and TARGET_FPS > in_fps * 1.02:
                    # 安全門：若倍率太大可視需要限制，但時間軸重採樣不會增加序列長度，通常不必限制
                    # ratio = TARGET_FPS / in_fps
                    # if ratio > UPSAMPLE_MAX_MULT + 1e-6:  # 真的要限制可放開
                    #     target = in_fps * UPSAMPLE_MAX_MULT
                    # else:
                    #     target = TARGET_FPS
                    target = TARGET_FPS

                    use_kps, use_bbox, use_dets, win_for_feats = _resample_sequences_to_fps(
                        window, use_kps, use_bbox, use_dets, target
                    )


            # 4) relation map + motion + mask（改用補幀後的序列）
            clips = []
            for i in range(len(use_kps)):
                fr_like = win_for_feats[i]
                canvas = rasterize_frame(use_bbox[i], use_kps[i], use_dets[i],
                                        fr_like.img_w, fr_like.img_h, self.cfg)
                clips.append(canvas)

            x = torch.from_numpy(np.stack(clips)).unsqueeze(0).float().to(DEVICE)
            motion_feats, valid_mask = compute_motion_feats_with_mask_from_window(
                win_for_feats, use_kps, conf_th=KP_CONF_TH
            )
            M = torch.from_numpy(motion_feats).unsqueeze(0).float().to(DEVICE)
            mask = torch.from_numpy(valid_mask).unsqueeze(0).float().to(DEVICE)

            # 5) 二階段模型推論
            with torch.no_grad():
                if USE_TWO_STAGE and getattr(self, "model_bin", None) is not None:
                    # --- (1) 先 binary ---
                    try:
                        logits_bin = self.model_bin(x, motion=M, mask=mask)
                    except TypeError:
                        logits_bin = self.model_bin(x)
                    if isinstance(logits_bin, (list, tuple)):
                        logits_bin = logits_bin[0]
                    if logits_bin.ndim > 2:
                        logits_bin = logits_bin.view(1, -1)
                    prob_bin = torch.softmax(logits_bin, dim=1).detach().cpu().numpy()[0].tolist()
                    try:
                        bin_fall_idx = self.class_names_bin.index(BINARY_POS_NAME)
                    except ValueError:
                        bin_fall_idx = min(1, len(self.class_names_bin)-1)  # 安全 fallback
                    p_fall = float(prob_bin[bin_fall_idx])
                    is_fall = (p_fall >= BINARY_THR)

                    multi_block = None
                    if (not is_fall) and getattr(self, "model_multi", None) is not None:
                        # --- (2) non-fall 才跑 multi ---
                        try:
                            logits_multi = self.model_multi(x, motion=M, mask=mask)
                        except TypeError:
                            logits_multi = self.model_multi(x)
                        if isinstance(logits_multi, (list, tuple)):
                            logits_multi = logits_multi[0]
                        if logits_multi.ndim > 2:
                            logits_multi = logits_multi.view(1, -1)
                        prob_multi = torch.softmax(logits_multi, dim=1).detach().cpu().numpy()[0].tolist()
                        pred_multi = int(np.argmax(prob_multi))
                        multi_block = {
                            "pred_idx": pred_multi,
                            "pred": self.class_names_multi[pred_multi] if 0 <= pred_multi < len(self.class_names_multi) else f"class_{pred_multi}",
                            "probs": [float(p) for p in prob_multi],
                        }

                    if is_fall:
                        final_pred = BINARY_POS_NAME
                        final_idx  = bin_fall_idx
                        final_probs = prob_bin     # 傳回的是 binary 的 2 維
                        stage = "binary"
                    else:
                        final_pred = multi_block["pred"]
                        final_idx  = multi_block["pred_idx"]
                        final_probs = multi_block["probs"]
                        stage = "multi"

                    result = {
                        "type": "inference",
                        "window": {"start_frame": window[0].frame_id, "end_frame": window[-1].frame_id},
                        # 舊欄位（相容前端）：以最終結果為準
                        "pred_idx": final_idx,
                        "pred": final_pred,
                        "probs": final_probs,
                        # 新增說明：二階段細節
                        "stage": stage,
                        "binary": {
                            "class_names": self.class_names_bin,
                            "pred_idx": bin_fall_idx if is_fall else int(np.argmax(prob_bin)),
                            "pred": BINARY_POS_NAME if is_fall else self.class_names_bin[int(np.argmax(prob_bin))],
                            "probs": [float(p) for p in prob_bin],
                            "thr": BINARY_THR,
                        },
                        "multi": (multi_block if multi_block is not None else None)
                    }
                    await ws_manager.send_json(result, user_id)

                else:
                    # 單模型 fallback（沿用你原來的邏輯）
                    try:
                        logits = self.model_multi(x, motion=M, mask=mask)
                    except TypeError:
                        logits = self.model_multi(x)
                    if isinstance(logits, (list, tuple)):
                        logits = logits[0]
                    if logits.ndim > 2:
                        logits = logits.view(1, -1)
                    prob = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
                    pred_idx = int(np.argmax(prob))
                    result = {
                        "type": "inference",
                        "window": {"start_frame": window[0].frame_id, "end_frame": window[-1].frame_id},
                        "pred_idx": pred_idx,
                        "pred": self.class_names_multi[pred_idx] if 0 <= pred_idx < len(self.class_names_multi) else f"class_{pred_idx}",
                        "probs": [float(p) for p in prob],
                        "stage": "single",
                    }
                    await ws_manager.send_json(result, user_id)

                    # 給事件判斷用的 p_fall（單模型時若有 'fall' 類就拿那一維，否則拿最大機率）
                    try:
                        p_fall_idx = self.class_names_multi.index("fall")
                        p_fall = float(prob[p_fall_idx])
                    except ValueError:
                        p_fall = float(np.max(prob))

            # 6) 中心幀決策：只有當『中心幀已定稿』才寫入 centerline，避免滑窗延長
            center_fid = window[0].frame_id + WINDOW // 2
            commit_center = (center_fid in finalized)
            st = self._fall_state.setdefault(user_id, {
                "active": False,
                "start_time": None,
                "peak_score": 0.0,
                "normal_streak": 0,
                "started": False
            })
            if commit_center:
                line = self._centerline.setdefault(user_id, {})
                # 已在上面的推論區塊算出 p_fall（binary 的 fall 機率）
                # 這裡直接用 p_fall，不再從 multi 的 prob 推估
                # （注意：若走單模型 fallback，p_fall 已在上面單模型分支給好）
                line[center_fid] = p_fall
                # 事件判定（僅看最近幾個中心點）
                recent_centers = sorted([k for k in line.keys() if k <= center_fid])
                def get_last_vals(n):
                    return [line[f] for f in recent_centers[-n:]] if n>0 else []

                if (not st["active"]):
                    # 先判斷「跌倒觸發」：連續 TRIGGER_CONSEC 個中心幀 ≥ FALL_TRIGGER_THR
                    vals = get_last_vals(TRIGGER_CONSEC)
                    if len(vals) == TRIGGER_CONSEC and all(v >= FALL_TRIGGER_THR for v in vals):
                        st.update({
                            "active": True,
                            "start_time": now_str(),
                            "start_frame": recent_centers[-TRIGGER_CONSEC],
                            "peak_score": max(vals)
                        })

                        # 準備 clip20（當前視窗的 20 幀原始資料）
                        clip20 = {
                            "frames": [
                                {
                                    "frame_id": fr.frame_id,
                                    "ts_ms": fr.ts_ms,
                                    "img_w": fr.img_w, "img_h": fr.img_h,
                                    "bbox": fr.bbox,
                                    "kps": fr.kps,
                                    "dets": fr.dets,
                                } for fr in window
                            ]
                        }

                        # 送出跌倒觸發（帶 clip20）
                        self._emit("on_fall_start", user_id, st["start_time"], result, clip20)

                    else:
                        # 沒觸發跌倒 → 才進行 multi 的坐/躺事件偵測（保留你原本這段）
                        multi_info = None
                        if isinstance(result, dict) and result.get("stage") == "multi":
                            multi_info = result.get("multi") or {
                                "pred_idx": result.get("pred_idx"),
                                "pred": result.get("pred"),
                                "probs": result.get("probs"),
                            }
                        if multi_info and multi_info.get("probs"):
                            prev_action = self._last_action_name.get(user_id)  # 可能為 None
                            curr_action = multi_info.get("pred")
                            self._update_multi_events(
                                user_id, center_fid, multi_info["probs"], result,
                                prev_action_name=prev_action, curr_action_name=curr_action
                            )
                            # 更新「上一個動作」
                            if isinstance(curr_action, str) and curr_action:
                                self._last_action_name[user_id] = curr_action
                else:
                    vals = get_last_vals(RECOVER_CONSEC)
                    if len(vals) == RECOVER_CONSEC and all(v <= FALL_RECOVER_THR for v in vals):
                        end_time = now_str(); end_frame = center_fid
                        self._emit("on_fall_recover", user_id, st["start_time"], end_time, float(st["peak_score"]), result)
                        st.update({"active": False, "start_time": None, "start_frame": None,
                                "peak_score": 0.0, "normal_streak": 0, "last_end_frame": end_frame})

            # 7) 彈出 STRIDE 並清理對應的快取
            to_pop_ids = [fr.frame_id for fr in window[:STRIDE]]
            buf.pop_left(STRIDE)
            user_cache = self._smooth_cache.get(user_id, {})
            finalized = self._smooth_finalized.get(user_id, set())
            for fid_rm in to_pop_ids:
                user_cache.pop(fid_rm, None)
                finalized.discard(fid_rm)

    def drop_user(self, user_id: str):
        print(f"[InferManager] Drop user '{user_id}' and clear all related data.")
        self._buffers.pop(user_id, None)
        self._locks.pop(user_id, None)
        self._fall_state.pop(user_id, None)
        self._centerline.pop(user_id, None)
        self._smooth_cache.pop(user_id, None)
        self._smooth_finalized.pop(user_id, None)
        self._last_action_name.pop(user_id, None)

stream_infer_manager = StreamInferManager()