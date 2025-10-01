# utils/stream_infer_manager.py
# -*- coding: utf-8 -*-
"""
StreamInferManager (remove per-event recover_min_ms; keep fall-level duration if set)

What’s in this build
- Two-stage inference (binary -> multi). The two loaders only build+load models;
  all preprocessing / logic stays here.
- Robust relation-map rasterization (tolerant to None), Kalman smoothing, motion(9) + mask.
- FPS normalization (auto upsample to 10fps when input<10, unless SC_TARGET_FPS>0).
- Event logic:
    * Fall: trigger/recover with optional FALL_RECOVER_MIN_MS (ms). (This remains.)
    * Multi events (sit/lie): ONLY use trigger/recover thresholds + consecutive counts.
      (Removed the per-event recover_min_ms concept.)

WS integration
- Uses ws_manager.send_json(user_id, payload)  (user first, then data).

Utility hooks
- set_handlers(on_fall_start, on_fall_recover, on_state_event_start, on_state_event_recover)
- force_recover(user_id, reason) to clear states and emit recover on errors
- drop_user(user_id) to clear buffers and states on disconnect
"""
from __future__ import annotations
import datetime
import os, asyncio, math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import cv2

# Project utils
from .ws_connection_manager import ws_manager
from .paths import SC_MODELS

# Minimal loaders (build+load only)
from .binary_cnn_lstm_loader import build_model as build_binary_model
from .multi_cnn_lstm_loader import build_model as build_multi_model

from .time_utils import format_time as fmt_ts

try:
    from .cnn_lstm_utils import CNNLSTM  # optional fallback
except Exception:
    CNNLSTM = None

# ================= Config (env-overridable) =================
H, W = 64, 64
WINDOW = int(os.getenv("SC_WINDOW", "10"))
STRIDE = int(os.getenv("SC_STRIDE", "5"))
INCLUDE_BONE_LINES = True
OBJECT_CLASSES: List[str] = [s.strip() for s in os.getenv("SC_OBJECT_CLASSES", "bed,chair,bench").split(",") if s.strip()]
COORDCONV_2 = True
LSTM_HIDDEN = 256
BIDIRECTIONAL = False
TEMPORAL_POOL = "attn"
DROPOUT = 0.3
ENABLE_KALMAN = True
KP_CONF_TH = float(os.getenv("SC_KP_CONF_TH", "0.2"))
SIGMA_KP = float(os.getenv("SC_SIGMA_KP", "3.0"))

# FPS resample: if SC_TARGET_FPS<=0, auto-upsample to 10fps when input < 10fps
TARGET_FPS = float(os.getenv("SC_TARGET_FPS", "0"))
UPSAMPLE_MAX_MULT = int(os.getenv("SC_UPSAMPLE_MAX_MULT", "4"))

# Two-stage inference
USE_TWO_STAGE = os.getenv("SC_USE_TWO_STAGE", "1") != "0"
BINARY_MODEL_PATH   = os.getenv("SC_BINARY_MODEL_PATH", f"{SC_MODELS}/binary/best.pt")
BINARY_CLASSES_PATH = os.getenv("SC_BINARY_CLASSES_PATH", f"{SC_MODELS}/binary/classes.json")
BINARY_POS_NAME     = os.getenv("SC_BINARY_POS_NAME", "fall")
BINARY_THR          = float(os.getenv("SC_BINARY_THR", "0.50"))
MULTI_MODEL_PATH    = os.getenv("SC_MULTI_MODEL_PATH",  f"{SC_MODELS}/multi/best.pt")
MULTI_CLASSES_PATH  = os.getenv("SC_MULTI_CLASSES_PATH",f"{SC_MODELS}/multi/classes.json")
# Single-model fallback
MODEL_PATH   = os.getenv("SC_MODEL_PATH",   f"{SC_MODELS}/best.pt")
CLASSES_PATH = os.getenv("SC_CLASSES_PATH", f"{SC_MODELS}/classes.json")

# Fall thresholds
FALL_TRIGGER_THR = float(os.getenv("SC_FALL_TRIGGER_THR", "0.55"))
FALL_RECOVER_THR = float(os.getenv("SC_FALL_RECOVER_THR", "0.40"))
TRIGGER_CONSEC   = int(os.getenv("SC_FALL_TRIGGER_CONSEC", "2"))
RECOVER_CONSEC   = int(os.getenv("SC_FALL_RECOVER_CONSEC", "2"))

# Multi-event rules 
ACTION_EVENTS = {
    "sit": {
        "pos_labels":     ["sit"],
        "recover_labels": ["lie", "walk"],
        "trigger_thr":    float(os.getenv("SC_SIT_TRIGGER_THR", "0.50")),
        "recover_thr":    float(os.getenv("SC_SIT_RECOVER_THR", "0.40")),
        "trigger_consec": int(os.getenv("SC_SIT_TRIGGER_CONSEC", "2")),
        "recover_consec": int(os.getenv("SC_SIT_RECOVER_CONSEC", "2")),
    },
    "lie": {
        "pos_labels":     ["lie"],
        "recover_labels": ["sit", "walk"],
        "trigger_thr":    float(os.getenv("SC_LIE_TRIGGER_THR", "0.50")),
        "recover_thr":    float(os.getenv("SC_LIE_RECOVER_THR", "0.40")),
        "trigger_consec": int(os.getenv("SC_LIE_TRIGGER_CONSEC", "2")),
        "recover_consec": int(os.getenv("SC_LIE_RECOVER_CONSEC", "2")),
    },
}

COCO_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= Utilities =================
class RelationMapConfig:
    def __init__(self, H=64, W=64, sigma_kp=3.0, kp_conf_th=0.4,
                 include_bone_lines=True, object_classes=None):
        self.H = int(H); self.W = int(W)
        self.sigma_kp = float(sigma_kp)
        self.kp_conf_th = float(kp_conf_th)
        self.include_bone_lines = bool(include_bone_lines)
        self.object_classes = [c.strip() for c in (object_classes or []) if c and c.strip()]


def gaussian2d(H, W, cx, cy, sigma):
    y, x = np.ogrid[:H, :W]
    return np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma ** 2))


def draw_gaussian(canvas, x, y, sigma, mag=1.0):
    Hc, Wc = canvas.shape
    x = float(np.clip(x, 0, Wc - 1)); y = float(np.clip(y, 0, Hc - 1))
    g = gaussian2d(Hc, Wc, x, y, sigma).astype(np.float32)
    canvas[:] = np.maximum(canvas, g * float(mag))


def _bbox_xyxy_from_cxcywh(cx, cy, w, h):
    x1 = cx - w/2.0; y1 = cy - h/2.0
    x2 = cx + w/2.0; y2 = cy + h/2.0
    return (x1, y1, x2, y2)


def rasterize_frame(bbox_xyxy, kps_list, dets, img_w, img_h, cfg: RelationMapConfig):
    Hc, Wc = cfg.H, cfg.W
    num_edges = len(COCO_EDGES) if cfg.include_bone_lines else 0
    num_obj_ch = len(cfg.object_classes or [])
    coord_ch = 2 if COORDCONV_2 else 0
    num_kp = 17

    C = 2 + num_kp + num_edges + num_obj_ch + coord_ch
    canvas = np.zeros((C, Hc, Wc), dtype=np.float32)

    safe_w = max(1.0, float(img_w) if img_w else 1.0)
    safe_h = max(1.0, float(img_h) if img_h else 1.0)

    ch = 0
    # 1) person bbox mask (tolerant to missing coords)
    if bbox_xyxy:
        try:
            x1, y1, x2, y2 = bbox_xyxy
            if None not in (x1, y1, x2, y2):
                x1 = int(np.clip((float(x1) / safe_w) * Wc, 0, Wc - 1))
                y1 = int(np.clip((float(y1) / safe_h) * Hc, 0, Hc - 1))
                x2 = int(np.clip((float(x2) / safe_w) * Wc, 0, Wc - 1))
                y2 = int(np.clip((float(y2) / safe_h) * Hc, 0, Hc - 1))
                if x2 > x1 and y2 > y1:
                    canvas[ch, y1:y2 + 1, x1:x2 + 1] = 1.0
        except Exception:
            pass
    ch += 1

    # 2) distance transform
    canvas[ch] = cv2.distanceTransform((canvas[ch-1] > 0).astype(np.uint8), cv2.DIST_L2, 3)
    if canvas[ch].max() > 0:
        canvas[ch] /= canvas[ch].max()
    ch += 1

    # 3) keypoint heatmaps
    if kps_list:
        for i, kp in enumerate(kps_list[:num_kp]):
            try:
                conf = float(kp.get('conf', kp.get('confidence', 1.0)))
                if conf < KP_CONF_TH:
                    continue
                x = (float(kp['x']) / safe_w) * Wc; y = (float(kp['y']) / safe_h) * Hc
                draw_gaussian(canvas[ch + i], x, y, SIGMA_KP, mag=conf)
            except Exception:
                pass
    ch += num_kp

    # 4) bone lines
    if num_edges > 0 and kps_list and len(kps_list) >= num_kp:
        pts = []
        for i in range(num_kp):
            try:
                x = int(np.clip((float(kps_list[i]['x']) / safe_w) * Wc, 0, Wc - 1))
                y = int(np.clip((float(kps_list[i]['y']) / safe_h) * Hc, 0, Hc - 1))
                c = float(kps_list[i].get('conf', kps_list[i].get('confidence', 1.0)))
                pts.append((x, y, c))
            except Exception:
                pts.append((None, None, 0.0))
        for e_idx, (a, b) in enumerate(COCO_EDGES):
            x1, y1, c1 = pts[a]; x2, y2, c2 = pts[b]
            if x1 is None or x2 is None:
                continue
            if min(c1, c2) < KP_CONF_TH:
                continue
            cv2.line(canvas[ch + e_idx], (x1, y1), (x2, y2), 1.0, 1)
    ch += num_edges

    # 5) object masks (tolerant bbox)
    if num_obj_ch > 0 and dets:
        cls2ch = {name: i for i, name in enumerate(cfg.object_classes)}
        for d in dets:
            try:
                cname = str(d.get("class_name") or d.get("label") or d.get("name") or "").strip().lower()
                if not cname or cname not in cls2ch:
                    continue
                x1, y1, x2, y2 = (d.get("x1"), d.get("y1"), d.get("x2"), d.get("y2"))
                if None in (x1, y1, x2, y2):
                    bb = d.get("bbox") or d.get("xyxy") or d.get("cxcywh")
                    if isinstance(bb, (list, tuple)) and len(bb) == 4:
                        if d.get("cxcywh"):
                            bb = _bbox_xyxy_from_cxcywh(*bb)
                        x1, y1, x2, y2 = bb
                if None in (x1, y1, x2, y2):
                    continue
                xx1 = int(np.clip((float(x1) / safe_w) * Wc, 0, Wc - 1))
                yy1 = int(np.clip((float(y1) / safe_h) * Hc, 0, Hc - 1))
                xx2 = int(np.clip((float(x2) / safe_w) * Wc, 0, Wc - 1))
                yy2 = int(np.clip((float(y2) / safe_h) * Hc, 0, Hc - 1))
                if xx2 > xx1 and yy2 > yy1:
                    canvas[ch + cls2ch[cname], yy1:yy2 + 1, xx1:xx2 + 1] = 1.0
            except Exception:
                pass
    ch += num_obj_ch

    # 6) coord conv
    if coord_ch == 2:
        xs = np.linspace(0, 1, Wc, dtype=np.float32)[None, :].repeat(Hc, 0)
        ys = np.linspace(0, 1, Hc, dtype=np.float32)[:, None].repeat(Wc, 1)
        canvas[ch] = ys; canvas[ch + 1] = xs
        ch += 2

    return canvas


def _interp_val(a: Optional[float], b: Optional[float], w: float) -> Optional[float]:
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return a * (1.0 - w) + b * w


def _interp_kp(kpa: dict, kpb: dict, w: float) -> dict:
    ax = kpa.get('x'); ay = kpa.get('y'); ac = float(kpa.get('conf', kpa.get('confidence', 1.0)))
    bx = kpb.get('x'); by = kpb.get('y'); bc = float(kpb.get('conf', kpb.get('confidence', 1.0)))
    x = _interp_val(ax, bx, w); y = _interp_val(ay, by, w)
    c = ac * (1.0 - w) + bc * w
    out = {}
    if x is not None:
        out['x'] = float(x)
    if y is not None:
        out['y'] = float(y)
    out['conf'] = float(c)
    return out


def _interp_bbox_xyxy(bba: Optional[Tuple[float, float, float, float]],
                      bbb: Optional[Tuple[float, float, float, float]],
                      w: float) -> Optional[Tuple[float, float, float, float]]:
    if not bba and not bbb:
        return None
    if not bba:
        return bbb
    if not bbb:
        return bba
    x1a, y1a, x2a, y2a = bba
    x1b, y1b, x2b, y2b = bbb
    return (
        _interp_val(x1a, x1b, w),
        _interp_val(y1a, y1b, w),
        _interp_val(x2a, x2b, w),
        _interp_val(y2a, y2b, w),
    )


def _densify_sequences(kps_seq: List[List[dict]],
                       bbox_seq: List[Optional[Tuple[float, float, float, float]]],
                       dets_seq: List[List[dict]],
                       mult: int):
    """Insert evenly spaced frames: mult=1 no-op, mult=2 insert 1 between each pair, etc."""
    L = len(kps_seq)
    if mult <= 1 or L <= 1:
        return kps_seq, bbox_seq, dets_seq
    new_kps = [kps_seq[0]]; new_bbox = [bbox_seq[0]]; new_dets = [dets_seq[0]]
    for i in range(L - 1):
        a_kps, b_kps = kps_seq[i], kps_seq[i + 1]
        a_bb, b_bb = bbox_seq[i], bbox_seq[i + 1]
        a_dt, b_dt = dets_seq[i], dets_seq[i + 1]
        new_kps.append(a_kps)
        new_bbox.append(a_bb)
        new_dets.append(a_dt)
        for m in range(1, mult):
            w = m / float(mult)
            inter = []
            for j in range(max(len(a_kps or []), len(b_kps or []))):
                ka = a_kps[j] if (a_kps and j < len(a_kps)) else {"conf": 0.0}
                kb = b_kps[j] if (b_kps and j < len(b_kps)) else {"conf": 0.0}
                inter.append(_interp_kp(ka, kb, w))
            new_kps.append(inter)
            new_bbox.append(_interp_bbox_xyxy(a_bb, b_bb, w))
            new_dets.append(a_dt)
    new_kps.append(kps_seq[-1]); new_bbox.append(bbox_seq[-1]); new_dets.append(dets_seq[-1])
    return new_kps, new_bbox, new_dets


def _estimate_fps(ts_ms_list: List[int]) -> float:
    if not ts_ms_list or len(ts_ms_list) < 2:
        return 0.0
    dts = [float(ts_ms_list[i] - ts_ms_list[i - 1]) for i in range(1, len(ts_ms_list))]
    dts = [dt for dt in dts if dt > 0]
    if not dts:
        return 0.0
    avg = sum(dts) / len(dts)
    return 1000.0 / avg


def _normalize_in_fps(in_fps: float) -> float:
    if in_fps <= 0:
        return 0.0
    if in_fps > 70:
        return 70.0
    if in_fps < 0.1:
        return 0.1
    return in_fps


def _resample_sequences_to_fps(window: list, kps_seq: list, bbox_seq: list, dets_seq: list, target_fps: float):
    ts = np.array([fr.ts_ms for fr in window], np.float64)
    t0 = ts[0]
    rel = (ts - t0) / 1000.0
    if rel[-1] <= 0:
        return kps_seq, bbox_seq, dets_seq, window
    dt = 1.0 / float(target_fps)
    new_t = np.arange(0.0, rel[-1] + 1e-6, dt, dtype=np.float64)
    if len(new_t) < 2:
        return kps_seq, bbox_seq, dets_seq, window

    def interp_list(a_list, b_list, w):
        L = int(max(len(a_list or []), len(b_list or [])))
        out = []
        for j in range(L):
            ka = a_list[j] if (a_list and j < len(a_list)) else {"conf": 0.0}
            kb = b_list[j] if (b_list and j < len(b_list)) else {"conf": 0.0}
            out.append(_interp_kp(ka, kb, w))
        return out

    kps_new = [kps_seq[0]]; bbox_new = [bbox_seq[0]]; dets_new = [dets_seq[0]]; fr_new = [window[0]]

    def find_prev_next(t):
        lo = int(np.searchsorted(rel, t, side='right')) - 1
        hi = min(lo + 1, len(rel) - 1)
        if lo < 0:
            lo = 0
        return lo, hi

    for t in new_t[1:]:
        lo, hi = find_prev_next(float(t))
        if hi == lo:
            kps_new.append(kps_seq[hi])
            bbox_new.append(bbox_seq[hi])
            dets_new.append(dets_seq[hi])
            fr_new.append(window[hi])
        else:
            w = (t - rel[lo]) / max(1e-6, (rel[hi] - rel[lo]))
            kps_new.append(interp_list(kps_seq[lo], kps_seq[hi], w))
            bbox_new.append(_interp_bbox_xyxy(bbox_seq[lo], bbox_seq[hi], w))
            dets_new.append(dets_seq[lo])
            fr_new.append(window[lo])
    return kps_new, bbox_new, dets_new, fr_new


# ================= Data Structures =================
@dataclass
class FrameRecord:
    frame_id: int
    ts_ms: int
    img_w: float
    img_h: float
    kps: Optional[List[dict]] = None
    bbox: Optional[Tuple[float, float, float, float]] = None
    dets: List[dict] = field(default_factory=list)


class UserBuffer:
    def __init__(self):
        self.by_id: Dict[int, FrameRecord] = {}
        self.ids: List[int] = []

    def upsert_pose(self, frame_id: int, ts_ms: int, img_w: float, img_h: float, persons: list):
        if not persons:
            return
        p = persons[0] or {}
        kps = p.get("keypoints") or p.get("kps") or []
        bb  = p.get("bbox") or None
        if isinstance(bb, dict):
            x1, y1 = bb.get("x1"), bb.get("y1")
            x2, y2 = bb.get("x2"), bb.get("y2")
            bb = (x1, y1, x2, y2)
        elif isinstance(bb, (list, tuple)) and len(bb) == 4:
            bb = tuple(bb)
        fr = self.by_id.get(frame_id)
        if fr is None:
            fr = FrameRecord(frame_id, ts_ms, img_w, img_h, kps=kps, bbox=bb, dets=[])
            self.by_id[frame_id] = fr
            self.ids.append(frame_id)
            self.ids.sort()
        else:
            fr.ts_ms = ts_ms; fr.img_w = img_w; fr.img_h = img_h
            fr.kps = kps; fr.bbox = bb

    def upsert_detect(self, frame_id: int, ts_ms: int, img_w: float, img_h: float, detections: list):
        fr = self.by_id.get(frame_id)
        if fr is None:
            fr = FrameRecord(frame_id, ts_ms, img_w, img_h, kps=[], bbox=None, dets=(detections or []))
            self.by_id[frame_id] = fr
            self.ids.append(frame_id)
            self.ids.sort()
        else:
            fr.dets = (detections or [])

    def get_monotonic(self) -> List[FrameRecord]:
        out = [self.by_id[i] for i in sorted(self.ids)]
        out2 = []
        last_id = -1; last_ts = -1
        for fr in out:
            if fr.frame_id <= last_id or fr.ts_ms < last_ts:
                continue
            last_id = fr.frame_id; last_ts = fr.ts_ms
            out2.append(fr)
        return out2


# ================= Kalman + Motion =================
class Kalman2D:
    def __init__(self, x0: float, y0: float):
        self.x = float(x0); self.y = float(y0)
        self.vx = 0.0; self.vy = 0.0
        self.alpha = 0.7
    def predict(self):
        self.x += self.vx
        self.y += self.vy
        self.vx *= self.alpha
        self.vy *= self.alpha
    def update(self, mx: Optional[float], my: Optional[float]):
        if mx is None or my is None:
            return
        rx = float(mx) - self.x; ry = float(my) - self.y
        self.vx += 0.3 * rx
        self.vy += 0.3 * ry
        self.x += 0.6 * rx
        self.y += 0.6 * ry


def kalman_smooth_kps(kps_seq: List[List[dict]]):
    L = len(kps_seq)
    J = 17
    filters = []
    for j in range(J):
        # seed with first measurement if available
        x0 = y0 = 0.0
        for t in range(L):
            fr = kps_seq[t]
            if fr and j < len(fr) and ('x' in fr[j]) and ('y' in fr[j]):
                x0 = float(fr[j]['x']); y0 = float(fr[j]['y'])
                break
        kf = Kalman2D(x0, y0)
        filters.append(kf)
    out = []
    for t in range(L):
        fr = kps_seq[t]
        sm = []
        for j in range(J):
            kf = filters[j]
            kf.predict()
            if fr and j < len(fr) and ('x' in fr[j]) and ('y' in fr[j]):
                kf.update(float(fr[j]['x']), float(fr[j]['y']))
            sm.append({'x': kf.x, 'y': kf.y, 'conf': float(fr[j].get('conf', fr[j].get('confidence', 1.0))) if (fr and j < len(fr)) else 0.0})
        out.append(sm)
    return out


def _compute_motion_feats_with_mask_from_window(window: List[FrameRecord], kps_seq: List[List[dict]]):
    ycom = []; hgt = []; area = []; trunk = []; kneeL = []; kneeR = []
    for fr, kps in zip(window, kps_seq):
        if not kps or len(kps) < 17:
            ycom.append(None); hgt.append(None); area.append(None); trunk.append(None); kneeL.append(None); kneeR.append(None)
            continue
        xs = []; ys = []
        for i in range(17):
            try:
                xs.append(float(kps[i]['x'])); ys.append(float(kps[i]['y']))
            except Exception:
                xs.append(None); ys.append(None)
        ys = [y for y in ys if y is not None]
        xs = [x for x in xs if x is not None]
        if not xs or not ys:
            ycom.append(None); hgt.append(None); area.append(None); trunk.append(None); kneeL.append(None); kneeR.append(None)
            continue
        ycom.append(float(np.mean(ys)))
        hgt.append(float((max(ys) - min(ys)) / max(1.0, fr.img_h)))
        area.append(float((max(xs) - min(xs)) * (max(ys) - min(ys)) / max(1.0, fr.img_w * fr.img_h)))
        try:
            trunk.append(float(np.arctan2((kps[5]['y'] - kps[11]['y']), (kps[5]['x'] - kps[11]['x']))))
        except Exception:
            trunk.append(None)
        try:
            kneeL.append(float(np.hypot(kps[11]['x'] - kps[13]['x'], kps[11]['y'] - kps[13]['y']) / max(1.0, fr.img_h)))
        except Exception:
            kneeL.append(None)
        try:
            kneeR.append(float(np.hypot(kps[12]['x'] - kps[14]['x'], kps[12]['y'] - kps[14]['y']) / max(1.0, fr.img_h)))
        except Exception:
            kneeR.append(None)

    def fill_small(arr):
        if not arr:
            return arr
        n = len(arr)
        i = 0
        while i < n:
            if arr[i] is None:
                j = i + 1
                while j < n and arr[j] is None:
                    j += 1
                a = arr[i - 1] if i > 0 else None
                b = arr[j] if j < n else None
                for k in range(i, j):
                    w = (k - i + 1) / float((j - i + 1) if (j > i) else 1.0)
                    arr[k] = _interp_val(a, b, w)
                i = j
            else:
                i += 1
        return [0.0 if v is None else v for v in arr]

    ycom = np.array(fill_small(ycom), np.float32)
    hgt  = np.array(fill_small(hgt),  np.float32)
    area = np.array(fill_small(area), np.float32)
    trunk= np.array(fill_small(trunk),np.float32)
    kneeL= np.array(fill_small(kneeL),np.float32)
    kneeR= np.array(fill_small(kneeR),np.float32)

    def diff1(x):
        v = np.zeros_like(x); v[1:] = x[1:] - x[:-1]; return v
    def diff2(v):
        a = np.zeros_like(v); a[1:] = v[1:] - v[:-1]; return a

    v_y, a_y = diff1(ycom), diff2(diff1(ycom))
    v_h, a_h = diff1(hgt), diff2(diff1(hgt))
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

    # validity mask based on confident lower-body core keypoints
    valid = []
    for fr, kps in zip(window, kps_seq):
        cnt = 0
        for j in (11, 12, 5, 6, 13, 14):
            try:
                c = float(kps[j].get('conf', kps[j].get('confidence', 1.0)))
                if c >= KP_CONF_TH:
                    cnt += 1
            except Exception:
                pass
        valid.append(1 if cnt >= 3 else 0)
    mask = np.array(valid, np.float32)
    return feats, mask


def _make_window(fr_list: List[FrameRecord], end_index: int, window: int = WINDOW) -> Optional[List[FrameRecord]]:
    if end_index < 0:
        return None
    s = end_index - window + 1
    if s < 0:
        return None
    return fr_list[s: end_index + 1]


# ================= Core =================
class StreamInferManager:
    def __init__(self):
        coord = 2 if COORDCONV_2 else 0
        self.in_ch = 2 + 17 + (len(COCO_EDGES) if INCLUDE_BONE_LINES else 0) + (len(OBJECT_CLASSES) if OBJECT_CLASSES else 0) + coord

        # Class names (overwritten by loaders)
        self.class_names_bin   = ["non_fall", "fall"]
        self.class_names_multi = ["class_0", "class_1"]

        if USE_TWO_STAGE:
            # Binary
            try:
                m_bin, names_bin, stats_bin = build_binary_model(
                    in_ch=self.in_ch,
                    motion_dim=9,
                    lstm_h=LSTM_HIDDEN,
                    lstm_layers=2,
                    bidirectional=BIDIRECTIONAL,
                    temporal_pool=TEMPORAL_POOL,
                    dropout=DROPOUT,
                    weight_path=BINARY_MODEL_PATH,
                    classes_path=BINARY_CLASSES_PATH,
                    device=DEVICE,
                    strict=False,
                )
                self.model_bin = m_bin
                self.class_names_bin = names_bin
                print(f"[Load][binary] matched={stats_bin['matched']}/{stats_bin['total_in_ckpt']} missing={stats_bin['missing']} unexpected={stats_bin['unexpected']}")
            except Exception as e:
                print(f"[Load][binary][WARN] {e}")
                self.model_bin = None

            # Multi
            try:
                m_multi, names_multi, stats_multi = build_multi_model(
                    in_ch=self.in_ch,
                    motion_dim=9,
                    lstm_h=LSTM_HIDDEN,
                    lstm_layers=2,
                    bidirectional=BIDIRECTIONAL,
                    temporal_pool=TEMPORAL_POOL,
                    dropout=DROPOUT,
                    weight_path=MULTI_MODEL_PATH,
                    classes_path=MULTI_CLASSES_PATH,
                    device=DEVICE,
                    strict=False,
                )
                self.model_multi = m_multi
                self.class_names_multi = names_multi
                print(f"[Load][multi] matched={stats_multi['matched']}/{stats_multi['total_in_ckpt']} missing={stats_multi['missing']} unexpected={stats_multi['unexpected']}")
            except Exception as e:
                print(f"[Load][multi][WARN] {e}")
                self.model_multi = None

            print(f"[InferManager] Two-stage loaded. bin={len(self.class_names_bin)}, multi={len(self.class_names_multi)}")
        else:
            try:
                m_single, names_single, stats_single = build_multi_model(
                    in_ch=self.in_ch,
                    motion_dim=9,
                    lstm_h=LSTM_HIDDEN,
                    lstm_layers=2,
                    bidirectional=BIDIRECTIONAL,
                    temporal_pool=TEMPORAL_POOL,
                    dropout=DROPOUT,
                    weight_path=MODEL_PATH,
                    classes_path=CLASSES_PATH,
                    device=DEVICE,
                    strict=False,
                )
                self.model_multi = m_single
                self.class_names_multi = names_single
                print(f"[Load][single] matched={stats_single['matched']}/{stats_single['total_in_ckpt']} missing={stats_single['missing']} unexpected={stats_single['unexpected']}")
            except Exception as e:
                print(f"[Load][single][WARN] {e}")
                # fallback
                self.model_multi = self._load_model()
            self.model_multi.eval().to(DEVICE)

        # State & config
        self._handlers: Dict[str, callable] = {}
        self._buffers: Dict[str, UserBuffer] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self.cfg = RelationMapConfig(H=H, W=W, sigma_kp=SIGMA_KP, kp_conf_th=KP_CONF_TH,
                                     include_bone_lines=INCLUDE_BONE_LINES, object_classes=OBJECT_CLASSES)

        # Events
        self._fall_state: Dict[str, Dict[str, int | bool | Optional[int]]] = {}
        self.action_events = self._compile_action_events(ACTION_EVENTS, self.class_names_multi)
        self._multi_event_state: Dict[str, Dict[str, Dict[str, int | bool | Optional[int]]]] = {}
        self._last_multi_label: Dict[str, str] = {}
        
    # ----- Basics -----
    def _load_model(self) -> nn.Module:
        num_classes = len(self.class_names_multi)
        if CNNLSTM is not None:
            return CNNLSTM(
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
        # tiny fallback
        class _Tiny(nn.Module):
            def __init__(self, in_ch, num_classes):
                super().__init__()
                self.c1 = nn.Sequential(
                    nn.Conv2d(in_ch,64,3,padding=1), nn.ReLU(),
                    nn.Conv2d(64,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(128,256,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
                )
                self.p = nn.Linear(256,256)
                self.lstm = nn.LSTM(256, LSTM_HIDDEN, num_layers=2, batch_first=True, bidirectional=BIDIRECTIONAL)
                feat = LSTM_HIDDEN * (2 if BIDIRECTIONAL else 1)
                self.fc = nn.Linear(feat, num_classes)
            def forward(self, x, motion=None, mask=None):
                B,T,C,Hh,Ww = x.shape
                z = self.p(self.c1(x.view(B*T,C,Hh,Ww)).flatten(1)).view(B,T,-1)
                z,_ = self.lstm(z)
                return self.fc(z[:,-1])
        return _Tiny(self.in_ch, num_classes)

    def set_handlers(self, **handlers):
        self._handlers.update(handlers)

    async def _emit(self, name: str, payload: dict):
        """觸發對應的事件處理 hook"""
        handler = self._handlers.get(name)
        if handler:
            try:
                await handler(**payload)
            except Exception as e:
                print(f"[ERROR] Failed to execute handler '{name}': {e}")

    def _compile_action_events(self, defs: Dict[str, dict], class_names: List[str]) -> Dict[str, dict]:
        idx_map = {name: i for i, name in enumerate(class_names)}
        compiled = {}
        for ev, cfg in (defs or {}).items():
            pos_idx = [idx_map[n] for n in cfg.get("pos_labels", []) if n in idx_map]
            rec_idx = [idx_map[n] for n in cfg.get("recover_labels", []) if n in idx_map]
            compiled[ev] = {**cfg, "pos_idx": pos_idx, "recover_idx": rec_idx}
        return compiled

    # ----- Preprocess -----
    def _build_clip(self, window: List[FrameRecord]):
        # smooth keypoints
        raw_kps = [(fr.kps or []) for fr in window]
        kps_seq = kalman_smooth_kps(raw_kps) if ENABLE_KALMAN else raw_kps

        # fps normalize
        use_kps = kps_seq
        use_bbox = [fr.bbox for fr in window]
        use_dets = [fr.dets for fr in window]
        win_for_feats = window

        in_fps = _estimate_fps([fr.ts_ms for fr in window])
        in_fps = _normalize_in_fps(in_fps)
        tgt = TARGET_FPS if TARGET_FPS > 0 else (10.0 if in_fps > 0 and in_fps < 10.0 else 0.0)
        if tgt and in_fps > 0 and tgt > in_fps:
            mult = min(UPSAMPLE_MAX_MULT, max(2, int(math.ceil(tgt / max(1e-6, in_fps)))))
            use_kps, use_bbox, use_dets = _densify_sequences(use_kps, use_bbox, use_dets, mult)
        elif tgt and in_fps > 0 and tgt < in_fps:
            use_kps, use_bbox, use_dets, win_for_feats = _resample_sequences_to_fps(window, use_kps, use_bbox, use_dets, tgt)

        # crop/pad to WINDOW
        if len(use_kps) > WINDOW:
            use_kps  = use_kps[-WINDOW:]
            use_bbox = use_bbox[-WINDOW:]
            use_dets = use_dets[-WINDOW:]
            win_for_feats = win_for_feats[-WINDOW:]
        elif len(use_kps) < WINDOW:
            pad_n = WINDOW - len(use_kps)
            pad_k = [[{"conf":0.0} for _ in range(17)] for _ in range(pad_n)]
            pad_b = [None for _ in range(pad_n)]
            pad_d = [[] for _ in range(pad_n)]
            use_kps  = pad_k + use_kps
            use_bbox = pad_b + use_bbox
            use_dets = pad_d + use_dets
            win_for_feats = ([window[0]] * pad_n) + win_for_feats

        # relation maps
        frames = []
        for fr, kps, bb, dets in zip(win_for_feats, use_kps, use_bbox, use_dets):
            rm = rasterize_frame(bb, kps, dets, fr.img_w, fr.img_h, self.cfg)
            frames.append(rm)
        x = np.stack(frames, axis=0)  # [T,C,H,W]

        # motion + mask
        feats, mask = _compute_motion_feats_with_mask_from_window(win_for_feats, use_kps)
        return x, feats, mask

    def _to_tensors(self, x_np, feats_np, mask_np):
        x = torch.from_numpy(x_np).unsqueeze(0)           # [1,T,C,H,W]
        motion = torch.from_numpy(feats_np).unsqueeze(0)  # [1,T,9]
        mask = torch.from_numpy(mask_np).unsqueeze(0)     # [1,T]
        return x.to(DEVICE), motion.to(DEVICE), mask.to(DEVICE)

    # ----- Inference -----
    @torch.no_grad()
    def _forward_any(self, model: nn.Module, x: torch.Tensor, motion: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
        out = None
        try:
            out = model(x, motion=motion, mask=mask)
        except TypeError:
            try:
                out = model(x, motion)
            except TypeError:
                out = model(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
        out = out.float()
        if out.dim() == 1:
            out = out.unsqueeze(0)
        return torch.softmax(out, dim=1).cpu().numpy()  # [1,num_classes]

    def _fall_idx(self) -> int:
        if not self.class_names_bin:
            return 1
        try:
            return self.class_names_bin.index(BINARY_POS_NAME)
        except Exception:
            return min(1, len(self.class_names_bin)-1)

    # ----- Event states -----
    def _fall_state_of(self, user_id: str):
        st = self._fall_state.get(user_id)
        if st is None:
            st = {"pos": 0, "rec": 0, "active": False}
            self._fall_state[user_id] = st
        return st

    def _evt_state_of(self, user_id: str, name: str):
        D = self._multi_event_state.setdefault(user_id, {})
        st = D.get(name)
        if st is None:
            st = {"pos": 0, "rec": 0, "active": False}
            D[name] = st
        return st

    async def _handle_fall_timeline(self, user_id: str, fall_prob: float, ts_ms: int):
        """處理跌倒事件的時間線"""
        state = self._fall_state_of(user_id)
        start_time = state.get("start_time")
        peak_score = state.get("peak_score", 0.0)

        if fall_prob >= FALL_TRIGGER_THR and not state.get("active"):
            # 跌倒事件開始
            state["active"] = True
            state["start_time"] = ts_ms
            state["peak_score"] = fall_prob

            # 準備回傳的參數
            payload = {
                "user_id": user_id,
                "start_time": fmt_ts(ts_ms),
                "result": {
                    "probs": state.get("probs", []),
                    "pred_idx": state.get("pred_idx", 0),
                },
                "clip": state.get("clip", {}),
            }
            await self._emit("on_fall_start", payload)

        elif fall_prob < FALL_RECOVER_THR and state.get("active"):
            # 跌倒事件結束
            state["active"] = False
            end_time = ts_ms

            payload = {
                "user_id": user_id,
                "start_time": fmt_ts(start_time),
                "end_time": fmt_ts(end_time),
                "peak_score": peak_score,
                "result": {
                    "probs": state.get("probs", []),
                    "pred_idx": state.get("pred_idx", 0),
                },
            }
            await self._emit("on_fall_recover", payload)

    async def _handle_action_events(self, user_id: str, probs: np.ndarray, ts_ms: int):
        """處理多事件（如坐下、躺下）的時間線（使用 pos_idx/recover_idx 列表 + consecutive 次數）"""
        for event_name, event_def in (self.action_events or {}).items():
            state = self._evt_state_of(user_id, event_name)

            # 取該事件「觸發集合」與「恢復集合」的分數（多標籤取最大值）
            pos_idx = event_def.get("pos_idx") or []
            rec_idx = event_def.get("recover_idx") or []

            pos_score = max((float(probs[i]) for i in pos_idx), default=0.0)
            rec_score = max((float(probs[i]) for i in rec_idx), default=0.0)

            trig_thr = float(event_def.get("trigger_thr", 0.5))
            recv_thr = float(event_def.get("recover_thr", 0.4))
            trig_need = int(event_def.get("trigger_consec", 1))
            recv_need = int(event_def.get("recover_consec", 1))

            # 初始化計數
            state.setdefault("pos", 0)
            state.setdefault("rec", 0)
            state.setdefault("active", False)
            state.setdefault("peak_score", 0.0)

            if not state["active"]:
                # 累積觸發計數
                state["pos"] = state["pos"] + 1 if pos_score >= trig_thr else 0
                if pos_score > state["peak_score"]:
                    state["peak_score"] = pos_score

                # 達到連續門檻 → 事件開始
                if state["pos"] >= trig_need:
                    state["active"] = True
                    state["start_time"] = ts_ms
                    # 清空恢復計數
                    state["rec"] = 0

                    payload = {
                        "user_id": user_id,
                        "event_name": event_name,
                        "start_time": fmt_ts(ts_ms),
                        "peak_score": state["peak_score"],
                        "payload": state.get("payload", {}),
                    }
                    await self._emit("on_state_event_start", payload)

            else:
                # 事件進行中：累積恢復計數
                state["rec"] = state["rec"] + 1 if rec_score >= recv_thr else 0

                # 達到連續門檻 → 事件結束
                if state["rec"] >= recv_need:
                    state["active"] = False
                    end_time = ts_ms
                    peak = state.get("peak_score", 0.0)

                    payload = {
                        "user_id": user_id,
                        "event_name": event_name,
                        "start_time": fmt_ts(state.get("start_time", ts_ms)),
                        "end_time": fmt_ts(end_time),
                        "peak_score": peak,
                        "payload": state.get("payload", {}),
                    }
                    # 重置計數與峰值
                    state["pos"] = 0
                    state["rec"] = 0
                    state["peak_score"] = 0.0

                    await self._emit("on_state_event_recover", payload)

    # ----- Buffers, locks, send -----
    def _buf(self, user_id: str) -> UserBuffer:
        return self._buffers.setdefault(user_id, UserBuffer())

    def _lock(self, user_id: str) -> asyncio.Lock:
        lk = self._locks.get(user_id)
        if lk is None:
            lk = asyncio.Lock(); self._locks[user_id] = lk
        return lk

    async def _send(self, user_id: str, payload: dict):
        try:
            if not isinstance(user_id, (str, int)):
                raise ValueError(f"Invalid user_id type: {type(user_id)}. Expected str or int.")
            await ws_manager.send_json(user_id, payload)
        except Exception as e:
            print(f"[WS][WARN] send to user={user_id} failed: {e}")

    # ----- Force recover (clear states on error) -----
    async def force_recover(self, user_id: str, reason: str = "manual"):
        try:
            st = self._fall_state.get(user_id)
            if st and st.get("active"):
                st["active"] = False
                st["pos"] = 0
                st["rec"] = 0
                await self._emit("on_fall_recover", {"user_id": user_id, "score": 0.0, "reason": reason})
            # multi 事件照舊
            evtD = self._multi_event_state.get(user_id) or {}
            for name, est in list(evtD.items()):
                if est.get("active"):
                    est["active"] = False
                    est["pos"] = 0
                    est["rec"] = 0
                    await self._emit("on_state_event_recover", {"user_id": user_id, "name": name, "score": 0.0, "reason": reason})
        except Exception as e:
            print(f"[STATE][WARN] force_recover user={user_id} err={e}")

    # ----- Drop user (on disconnect) -----
    def drop_user(self, user_id: str):
        self._buffers.pop(user_id, None)
        self._locks.pop(user_id, None)
        self._fall_state.pop(user_id, None)
        self._multi_event_state.pop(user_id, None)

    # ----- WS ingest -----
    async def ingest(self, user_id: str, msg: dict):
        async with self._lock(user_id):
            t = (msg.get("type") or "frame").lower()
            fid = int(msg.get("frame_id", -1))
            ts = int(msg.get("timestamp_ms", 0))
            img_w = float(((msg.get("image_size") or {}).get("width") or 640))
            img_h = float(((msg.get("image_size") or {}).get("height") or 480))
            buf = self._buf(user_id)

            pose   = msg.get("pose")    or msg.get("persons")
            detect = msg.get("detect")  or msg.get("detections")

            if t == "frame" or (t not in ("pose","object","detect") and (pose is not None or detect is not None)):
                if pose   is not None: buf.upsert_pose(fid, ts, img_w, img_h, pose)
                if detect is not None: buf.upsert_detect(fid, ts, img_w, img_h, detect)
            elif t == "pose":
                if pose is not None: buf.upsert_pose(fid, ts, img_w, img_h, pose)
            elif t in ("object","detect"):
                if detect is not None: buf.upsert_detect(fid, ts, img_w, img_h, detect)
            else:
                return

            fr_list = buf.get_monotonic()
            if len(fr_list) < WINDOW:
                return
            last_index = len(fr_list) - 1
            if (last_index + 1) % STRIDE != 0:
                return
            window = _make_window(fr_list, last_index, WINDOW)
            if not window:
                return

            # Preprocess
            x_np, feats_np, mask_np = self._build_clip(window)
            x, motion, mask = self._to_tensors(x_np, feats_np, mask_np)
            ts_cur = int(window[-1].ts_ms)

            # ---- Binary → Multi ----
            binary_out = None; multi_out = None
            stage = "multi"

            if USE_TWO_STAGE and getattr(self, "model_bin", None) is not None:
                probs_bin = self._forward_any(self.model_bin, x, motion, mask)[0]
                binary_out = {
                    "class_names": self.class_names_bin,
                    "probs": [float(p) for p in probs_bin],
                }
                fall_idx = self._fall_idx()
                pred_idx_bin = int(np.argmax(probs_bin))
                binary_out["pred_idx"] = pred_idx_bin
                binary_out["pred"] = self.class_names_bin[pred_idx_bin]
                binary_out["thr"] = float(BINARY_THR)

                fall_prob = float(probs_bin[fall_idx])
                await self._handle_fall_timeline(user_id, fall_prob, ts_cur)

                if fall_prob >= BINARY_THR:
                    stage = "binary"
                    payload = {
                        "type": "inference",
                        "stage": stage,
                        "pred_idx": fall_idx,
                        "pred": self.class_names_bin[fall_idx],
                        "binary": binary_out,
                    }
                    await self._send(user_id, payload)
                    return

            # Multi-class stage
            if getattr(self, "model_multi", None) is not None:
                probs_multi = self._forward_any(self.model_multi, x, motion, mask)[0]
                multi_out = {
                    "class_names": self.class_names_multi,
                    "probs": [float(p) for p in probs_multi],
                }
                pred_idx_multi = int(np.argmax(probs_multi))
                multi_out["pred_idx"] = pred_idx_multi
                multi_out["pred"] = self.class_names_multi[pred_idx_multi]

                await self._handle_action_events(user_id, probs_multi, ts_cur)

                payload = {
                    "type": "inference",
                    "stage": stage,
                    "pred_idx": pred_idx_multi,
                    "pred": self.class_names_multi[pred_idx_multi],
                    "multi": multi_out,
                }
                if binary_out is not None:
                    payload["binary"] = binary_out
                await self._send(user_id, payload)
            else:
                # no multi: still send binary if available
                if binary_out is not None:
                    pred_idx = int(np.argmax(binary_out["probs"]))
                    payload = {
                        "type": "inference",
                        "stage": "binary-only",
                        "pred_idx": pred_idx,
                        "pred": self.class_names_bin[pred_idx],
                        "binary": binary_out,
                    }
                    await self._send(user_id, payload)


# Singleton
stream_infer_manager = StreamInferManager()
