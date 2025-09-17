# app/utils/stream_infer_manager.py
import asyncio
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import cv2

from .ws_connection_manager import ws_manager
from .paths import SC_MODELS

from .time_utils import now_str

# 若有你自己的 CNNLSTM，這裡直接用；否則 fallback
try:
    from .cnn_lstm_utils import CNNLSTM
except Exception:
    CNNLSTM = None

# === 動作判斷設定 ===
ABNORMAL_LABELS = {"fall"}   # 需要額外作業的類別；之後可加 "lying"、"kneel" ...
TH_FALL = 0.60       # 跌倒分數門檻
RECOVER_CONSEC = 2           # 連續多少個視窗才算「回復正常」

# ===================== 與 loader 對齊的參數 =====================
H, W = 64, 64
WINDOW = 20
STRIDE = 5

INCLUDE_BONE_LINES = True
OBJECT_CLASSES: List[str] = ["bed", "chair"]   # 與訓練一致 :contentReference[oaicite:1]{index=1}
COORDCONV_2 = True                             # 加入 2 個座標通道（x/y） :contentReference[oaicite:2]{index=2}

LSTM_HIDDEN = 256
BIDIRECTIONAL = False
TEMPORAL_POOL = "attn"
DROPOUT = 0.3                                   # 與訓練一致 :contentReference[oaicite:3]{index=3}

# 卡爾曼（半視窗）/第一幀完整檢查（需 bbox + 17kp）
ENABLE_KALMAN = True
KALMAN_HALF_SLIDE = True
HALF_LEN_OVERRIDE = None
REQUIRE_FULL_FIRST = True                      # 與 loader 一致：第一幀需完整且含 bbox :contentReference[oaicite:4]{index=4}

KP_CONF_TH = 0.0
SIGMA_KP = 3.0

COCO_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = f"{SC_MODELS}/best.pt"
CLASSES_PATH = f"{SC_MODELS}/classes.json"

# ===================== Relation Map & 小工具 =====================
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

    # 通道與 loader 對齊：1(bbox)+1(dist)+17(kp)+edges+obj+2(coord) :contentReference[oaicite:5]{index=5}
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

    # 5) objects（每類一通道，與訓練一致）
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

    # 防呆：通道數對齊
    # assert ch == C, f"channel mismatch {ch} vs {C}"
    return canvas  # (C,H,W)

def frame_has_full_skeleton(bbox, kps, *, kp_need=17, require_bbox=True):
    """與 loader 相同：第一幀需有 bbox 且 17 個關節（達門檻）:contentReference[oaicite:6]{index=6}"""
    if require_bbox and not bbox:
        return False
    if not kps:
        return False
    cnt = 0
    for j in range(min(17, len(kps))):
        x = kps[j].get('x'); y = kps[j].get('y')
        if x is None or y is None: 
            continue
        conf = float(kps[j].get('conf', kps[j].get('confidence', 1.0)))
        if conf < KP_CONF_TH: 
            continue
        cnt += 1
    return cnt >= kp_need

# ===================== Kalman（半視窗）與平滑 =====================
class Kalman2D:
    def __init__(self, x=0.0, y=0.0, var_pos=1e-2, var_vel=1e-1, var_meas=4.0):
        self.F = np.array([[1,0,1,0],
                           [0,1,0,1],
                           [0,0,1,0],
                           [0,0,0,1]], dtype=np.float32)
        self.H = np.array([[1,0,0,0],
                           [0,1,0,0]], dtype=np.float32)
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
    """半視窗 KF：0~half覆蓋，之後每 step 覆蓋尾段（與 loader 一致）:contentReference[oaicite:7]{index=7}"""
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

# ===================== Fallback 模型（載入失敗時） =====================
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
    def forward(self, x):            # x: (B,T,C,H,W)
        B,T,C,Hh,Ww = x.shape
        z = self.cnn(x.view(B*T, C, Hh, Ww)).view(B, T, -1)
        z,_ = self.lstm(z)
        return self.fc(z[:, -1])

# ===================== 使用者 Buffer 與整體 Manager =====================
@dataclass
class FrameRecord:
    frame_id: int
    ts_ms: int
    img_w: float
    img_h: float
    kps: Optional[List[dict]] = None
    bbox: Optional[Tuple[float,float,float,float]] = None  # (x1,y1,x2,y2)
    dets: List[dict] = field(default_factory=list)

class UserBuffer:
    def __init__(self):
        self.frames: List[FrameRecord] = []

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
            self.frames.sort(key=lambda x: x.frame_id)

    def pop_left(self, n: int):
        self.frames = self.frames[n:] if n>0 else self.frames

    def ready_window(self, require_full_first: bool = True) -> Optional[List[FrameRecord]]:
        if len(self.frames) < WINDOW:
            return None
        window = self.frames[:WINDOW]
        # 與 loader 一致：第一幀需 bbox+17kp
        if require_full_first and REQUIRE_FULL_FIRST and not frame_has_full_skeleton(window[0].bbox, window[0].kps, require_bbox=True):
            self.pop_left(1)
            print(f"[InferManager] WARN: First frame incomplete, discard and wait for next.")
            return None
        return window

class StreamInferManager:
    def __init__(self):
        coord = 2 if COORDCONV_2 else 0
        self.in_ch = 2 + 17 + (len(COCO_EDGES) if INCLUDE_BONE_LINES else 0) + (len(OBJECT_CLASSES) if OBJECT_CLASSES else 0) + coord
        # 類別
        try:
            with open(CLASSES_PATH, "r", encoding="utf-8") as f:
                self.class_names = json.load(f)
        except Exception:
            self.class_names = ["class_0", "class_1"]
        self.num_classes = len(self.class_names)
        # 模型（用與訓練一致的超參建模後再載 state_dict）:contentReference[oaicite:8]{index=8}
        self.model = self._load_model()
        self.model.eval().to(DEVICE)
        # 其他
        self.cfg = RelationMapConfig(H=H, W=W, sigma_kp=SIGMA_KP, kp_conf_th=KP_CONF_TH,
                                     include_bone_lines=INCLUDE_BONE_LINES, object_classes=OBJECT_CLASSES)
        self._buffers: Dict[str, UserBuffer] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._handlers = {}                 # 事件 hooks：外部注入
        self._fall_state = {}               # user_id -> {"active":bool, "start_time":str, "start_frame":int, "peak_score":float, "normal_streak":int}

    def _load_model(self) -> nn.Module:
        try:
            state = torch.load(MODEL_PATH, map_location=DEVICE)
            # 無論 checkpoint 是否包 "state_dict"，都用相同結構建模再 load
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
                )
            else:
                # 簡易 fallback 結構（若沒有你自訂的 CNNLSTM 類別）
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
                        self.fc = nn.Linear(feat, self.num_classes if hasattr(self,'num_classes') else num_classes)
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
                # 完整模型也可，但通常我們存的是 state_dict
                m = state
            print(f"[InferManager] Loaded weights: in_ch={self.in_ch}, classes={self.num_classes}")
            return m
        except Exception as e:
            print(f"[InferManager] WARN: Load model failed: {e}")
            return _SimpleCNNLSTM(in_ch=self.in_ch, num_classes=self.num_classes)

    def _buf(self, user_id: str) -> UserBuffer:
        return self._buffers.setdefault(user_id, UserBuffer())

    def _lock(self, user_id: str) -> asyncio.Lock:
        if user_id not in self._locks:
            self._locks[user_id] = asyncio.Lock()
        return self._locks[user_id]

    def set_handlers(self, **handlers):
        """
        註冊事件 hook（皆為 async function）：
        - on_fall_start(user_id, start_time, start_frame, result)
        - on_fall_recover(user_id, start_time, end_time, start_frame, end_frame, peak_score, result)
        """
        self._handlers.update({k: v for k, v in handlers.items() if v})

    def _emit(self, name: str, *args, **kwargs):
        h = self._handlers.get(name)
        if h:
            asyncio.create_task(h(*args, **kwargs))  # 背景執行，不阻塞推論

    async def force_recover(self, user_id: str, reason: str = "manual"):
        """
        強制將某 user 尚未結束的跌倒事件補上 end_time 並觸發 on_fall_recover。
        在使用者斷線、或伺服器關閉時呼叫。
        """
        st = self._fall_state.get(user_id)
        if not st or not st.get("active"):
            return  # 沒有未結束事件就跳過

        start_time: Optional[str] = st.get("start_time")
        start_frame: Optional[int] = st.get("start_frame")
        end_frame: Optional[int] = st.get("last_end_frame")  # 我們在 ingest 裡會更新它
        peak_score: float = float(st.get("peak_score", 0.0))
        end_time = now_str()

        # 傳一個最小 result（你的 hook 若不需要可以忽略）
        result_stub = {"forced_reason": reason, "pred": "fall"}

        # 非阻塞觸發（但在 shutdown 我們會 await 全部）
        if self._handlers.get("on_fall_recover"):
            await self._handlers["on_fall_recover"](
                user_id, start_time, end_time, start_frame, end_frame, peak_score, result_stub
            )

        # 重置狀態
        st["active"] = False
        st["start_time"] = None
        st["start_frame"] = None
        st["peak_score"] = 0.0
        st["normal_streak"] = 0
        st["last_end_frame"] = None

    async def force_recover_all(self, reason: str = "shutdown"):
        """
        把所有使用者的未結束跌倒事件通通補上 end_time。
        用於伺服器關機的優雅關閉。
        """
        tasks = []
        for uid, st in list(self._fall_state.items()):
            if st and st.get("active"):
                tasks.append(self.force_recover(uid, reason=reason))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def ingest(self, user_id: str, msg: dict):
        """
        接收 WS 訊息（type=pose/object），加到該使用者 buffer。
        滿 20 幀即推論 → 結果回傳該 user。
        """
        async with self._lock(user_id):
            t = (msg.get("type") or "").lower()
            fid = int(msg.get("frame_id", -1))
            ts = int(msg.get("timestamp_ms", 0))
            img_w = float(((msg.get("image_size") or {}).get("width") or 640))
            img_h = float(((msg.get("image_size") or {}).get("height") or 480))

            buf = self._buf(user_id)
            if t == "pose":
                buf.upsert_pose(fid, ts, img_w, img_h, msg.get("persons") or [])
            elif t == "object":
                buf.upsert_objects(fid, ts, img_w, img_h, msg.get("detections") or [])
            else:
                return

            window = buf.ready_window()
            if window is None:
                return

            # 取出並平滑（半視窗 KF）
            kps_seq = [fr.kps for fr in window]
            if ENABLE_KALMAN:
                half_len = int(HALF_LEN_OVERRIDE) if HALF_LEN_OVERRIDE else max(1, WINDOW//2)
                kps_seq = kalman_smooth_kps(
                    kps_seq,
                    half_slide=KALMAN_HALF_SLIDE,
                    half_len=half_len,
                    require_full_first=True,
                    step=STRIDE
                )

            # 轉 Relation Maps（含 CoordConv/物件通道）
            clips = []
            for i, fr in enumerate(window):
                canvas = rasterize_frame(fr.bbox, kps_seq[i], fr.dets, fr.img_w, fr.img_h, self.cfg)
                clips.append(canvas)
            x = torch.from_numpy(np.stack(clips)).unsqueeze(0).float().to(DEVICE)  # (1,T,C,H,W)

            with torch.no_grad():
                logits = self.model(x)
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
                if logits.ndim > 2:
                    logits = logits.view(1, -1)
                prob = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]

            pred_idx = int(prob.argmax())
            result = {
                "type": "inference",
                "window": {"start_frame": window[0].frame_id, "end_frame": window[-1].frame_id},
                "pred_idx": pred_idx,
                "pred": self.class_names[pred_idx] if 0 <= pred_idx < len(self.class_names) else f"class_{pred_idx}",
                "probs": [float(p) for p in prob.tolist()]
            }

            await ws_manager.send_json(result, user_id)
            
            # === 跌倒狀態機（只處理跌倒與恢復） ===
            pred  = result["pred"]
            score = result["probs"][result["pred_idx"]]
            now_time = now_str()

            st = self._fall_state.setdefault(user_id, {
                "active": False,
                "start_time": None,
                "start_frame": None,
                "peak_score": 0.0,
                "normal_streak": 0,
                "started": False, 
            })

            if pred == "fall" and score >= TH_FALL:
                # 跌倒視窗
                st["normal_streak"] = 0
                if not st["active"]:
                    # 進入跌倒狀態：只「記開始時間與起始 frame」，不入庫
                    st["active"] = True
                    st["start_time"] = now_time
                    st["start_frame"] = result["window"]["start_frame"]
                    st["peak_score"] = score
                    # 如果你要即時做通知，可在 route 的 on_fall_start 裡實作
                    self._emit("on_fall_start", user_id, st["start_time"], st["start_frame"], result)
                else:
                    # 跌倒維持中，更新最高分
                    if score > st["peak_score"]:
                        st["peak_score"] = score
            else:
                # 非跌倒視窗（或跌倒分數低於門檻）
                if st["active"]:
                    st["normal_streak"] += 1
                    if st["normal_streak"] >= RECOVER_CONSEC:
                        # 視為恢復正常：這時候才「寫資料庫」（包含 start/end）
                        end_time = now_time
                        end_frame = result["window"]["end_frame"]
                        self._emit(
                            "on_fall_recover",
                            user_id,
                            st["start_time"],
                            end_time,
                            st["start_frame"],
                            end_frame,
                            st["peak_score"],
                            result
                        )
                        # 重置狀態
                        st["active"] = False
                        st["start_time"] = None
                        st["start_frame"] = None
                        st["peak_score"] = 0.0
                        st["normal_streak"] = 0

            buf.pop_left(STRIDE)

    def drop_user(self, user_id: str):
        self._buffers.pop(user_id, None)
        self._locks.pop(user_id, None)
        self._fall_state.pop(user_id, None)   


# 單例
stream_infer_manager = StreamInferManager()
