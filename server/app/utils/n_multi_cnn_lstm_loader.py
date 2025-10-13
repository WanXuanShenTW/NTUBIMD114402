# -*- coding: utf-8 -*-
"""
JSON → CNN+LSTM 推論（常數設定版，含 NaN 防呆 + 掩碼穩定化）
- ✅ 檔頭使用「常數變數」設定（可直接 `python cnn_lstm_loader.py` 啟動）
- ✅ attention 掩碼改用大負數，並處理全 0 掩碼，避免 softmax NaN
- ✅ softmax 前做數值穩定化（減最大值 + nan_to_num）
- ✅ 距離變換正規化避免 0/0
- ✅ 與新版 trainer 的 RelationMap / Motion / Mask 介面一致

使用方式：
1) 直接在檔頭修改 `MODEL_PATH / CLASSES_PATH / POSE_JSON / OBJECT_JSON / OBJECT_CLASSES / WINDOW / STRIDE`。
2) 執行： `python cnn_lstm_loader.py`
3) 若要輸出 CSV，設定 `OUT_CSV = "preds.csv"`。
"""

import os, json, math
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import cv2

# ===================== 常數設定（請依環境調整） =====================
MODEL_PATH      = "models/multi/best.pt"       # best.pt（state_dict）或 epXXX_score*.pt（完整 ckpt）
CLASSES_PATH    = "models/multi/classes.json"   # 若載入 best.pt 需提供類別清單

# 測試檔（影片級 JSON，list 形式；索引=幀號）
# POSE_JSON       = "outputs/skeletons/YOLO/YOLO-pose/fall/fall_011.json"
# OBJECT_JSON     = "outputs/skeletons/YOLO/YOLO-detect/fall/fall_011.json"
# POSE_JSON       = "outputs/skeletons/YOLO/YOLO-pose/normal/normal_002.json"
# OBJECT_JSON     = "outputs/skeletons/YOLO/YOLO-detect/normal/normal_002.json"
POSE_JSON       = "medias/test/fall/IMG_7704-pose.json"    # ← 這裡放「pose」JSON
OBJECT_JSON     = "medias/test/fall/IMG_7704-detect.json"  # ← 這裡放「detect/objects」JSON
# POSE_JSON       = "outputs/skeletons/multi/YOLO-pose/lie/780251760.975590_back.json"
# OBJECT_JSON     = "outputs/skeletons/multi/YOLO-detect/lie/780251760.975590_back.json"

OUT_CSV         = None  # 例如 "preds.csv"；不要輸出就設 None

# Relation Map 與物件通道（需與訓練一致）
H, W                   = 64, 64
INCLUDE_BONE_LINES     = True
OBJECT_CLASSES         = ["bed", "chair", "bench"]  # 若不用物件通道 → []

# LSTM / 時序（需與訓練一致）
LSTM_HIDDEN            = 256
BIDIRECTIONAL          = False
TEMPORAL_POOL          = "attn"   # "last" | "mean" | "attn"
WINDOW                 = 20
STRIDE                 = 5
DROPOUT                = 0.3

# ===== 前置：KF 與完整性判斷（與訓練一致） =====
ENABLE_KALMAN               = True
KALMAN_HALF_SLIDE           = False    # 1~10 初始化；之後每 5 幀用 5~15、10~20 覆蓋尾段
REQUIRE_FULL_FIRST_FRAME    = False    # 視窗第一幀必須完整，否則此視窗不推論
HALF_LEN_OVERRIDE           = None     # 預設用 WINDOW//2 (=10)
KP_CONF_TH                  = 0.25      # 與 trainer 同步
SIGMA_KP                    = 2.0

# Motion & 掩碼
_USE_MOTION                 = True     # 與 trainer 一致；motion 維度固定 9
_MOTION_DIM                 = 9
_MOTION_CLIP_V              = 3.0
_MOTION_CLIP_A              = 9.0

# ===================== Relation Map（與訓練一致） =====================
COCO_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]

class RelationMapConfig:
    def __init__(self, H=64, W=64, sigma_kp=3.0, kp_conf_th=0.4,
                 include_bone_lines=True, object_classes=None):
        self.H = int(H); self.W = int(W)
        self.sigma_kp = float(sigma_kp); self.kp_conf_th = float(kp_conf_th)
        self.include_bone_lines = bool(include_bone_lines)
        self.object_classes = [c.strip() for c in (object_classes or []) if c.strip()]

def draw_gaussian(heatmap, x, y, sigma, mag=1.0):
    H,W = heatmap.shape
    xx,yy = np.meshgrid(np.arange(W), np.arange(H))
    g = np.exp(-((xx-x)**2+(yy-y)**2)/(2.0*sigma**2)).astype(np.float32)*mag
    heatmap += g

def _bbox_xyxy_from_any(b):
    if b is None: return None
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

def rasterize_frame(bbox, kps, objects, img_w, img_h, cfg: RelationMapConfig):
    Hc, Wc = cfg.H, cfg.W
    num_bbox_mask = 1
    num_dist = 1
    num_kp = 17
    num_edges = len(COCO_EDGES) if cfg.include_bone_lines else 0
    num_obj_ch = len(cfg.object_classes)
    num_coord = 2
    C = num_bbox_mask + num_dist + num_kp + num_edges + num_obj_ch + num_coord

    canvas = np.zeros((C,Hc,Wc), np.float32)
    ch = 0
    # 1) bbox mask
    if bbox:
        bxyxy = _bbox_xyxy_from_any(bbox)
        if bxyxy is not None:
            x1,y1,x2,y2 = bxyxy
            x1 = int(np.clip((x1/img_w)*Wc, 0, Wc-1)); y1 = int(np.clip((y1/img_h)*Hc, 0, Hc-1))
            x2 = int(np.clip((x2/img_w)*Wc, 0, Wc-1)); y2 = int(np.clip((y2/img_h)*Hc, 0, Hc-1))
            if x2>=x1 and y2>=y1:
                canvas[ch, y1:y2+1, x1:x2+1] = 1.0
    ch += 1
    # 2) distance transform（避免 0/0）
    inv = (1.0-canvas[0]).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    mx = float(dist.max())
    if mx>0: dist = dist/(mx+1e-6)
    canvas[ch] = dist; ch += 1
    # 3) keypoints
    if kps:
        for i, kp in enumerate(kps[:num_kp]):
            try:
                conf = float(kp.get('conf', kp.get('confidence',1.0)))
                if conf < KP_CONF_TH: continue
                x = (float(kp['x'])/img_w)*Wc; y = (float(kp['y'])/img_h)*Hc
                draw_gaussian(canvas[ch+i], x, y, cfg.sigma_kp, mag=conf)
            except Exception:
                pass
    ch += num_kp
    # 4) bone lines
    if cfg.include_bone_lines and kps and len(kps)>=num_kp:
        pts = []
        for i in range(num_kp):
            try:
                x = int(np.clip((float(kps[i]['x'])/img_w)*Wc,0,Wc-1))
                y = int(np.clip((float(kps[i]['y'])/img_h)*Hc,0,Hc-1))
                c = float(kps[i].get('conf', kps[i].get('confidence',1.0)))
                pts.append((x,y,c))
            except Exception:
                pts.append((None,None,0.0))
        for e_idx,(a,b) in enumerate(COCO_EDGES):
            x1,y1,c1 = pts[a]; x2,y2,c2 = pts[b]
            if x1 is None or x2 is None: continue
            if min(c1,c2) < KP_CONF_TH: continue
            cv2.line(canvas[ch+e_idx], (x1,y1), (x2,y2), 1.0, 1)
    ch += num_edges
    # 5) objects
    if num_obj_ch>0 and objects:
        name_to_idx = {n:i for i,n in enumerate(cfg.object_classes)}
        for det in objects:
            cname = None
            for key in ("cls_name","class_name","name","label"):
                val = det.get(key)
                if isinstance(val,str) and val.strip():
                    cname = val.strip(); break
            if cname not in name_to_idx: continue
            bxyxy = _bbox_xyxy_from_any(det.get('bbox') or det.get('xyxy') or det.get('box') or det.get('bbox_xyxy'))
            if bxyxy is None: continue
            x1,y1,x2,y2 = bxyxy
            x1 = int(np.clip((x1/img_w)*Wc, 0, Wc-1)); y1 = int(np.clip((y1/img_h)*Hc, 0, Hc-1))
            x2 = int(np.clip((x2/img_w)*Wc, 0, Wc-1)); y2 = int(np.clip((y2/img_h)*Hc, 0, Hc-1))
            if x2>=x1 and y2>=y1:
                canvas[ch + name_to_idx[cname], y1:y2+1, x1:x2+1] = 1.0
    ch += num_obj_ch
    # 6) CoordConv
    xv = np.linspace(-1,1,Wc)[None,:].repeat(Hc,0)
    yv = np.linspace(-1,1,Hc)[:,None].repeat(Wc,1)
    canvas[ch] = xv; canvas[ch+1] = yv; ch+=2
    assert ch==C, f"channel mismatch {ch} vs {C}"
    return canvas

# ===================== JSON 解析 =====================

def read_any_json(path: Optional[str]):
    if not path: return []
    with open(path, 'r', encoding='utf-8') as f:
        txt = f.read().strip()
    if not txt: return []
    try:
        data = json.loads(txt)
        if isinstance(data, list):
            return [r for r in data if isinstance(r,(dict,list))]
        if isinstance(data, dict):
            for k in ('frames','data','records','annotations','items','results'):
                if isinstance(data.get(k), list):
                    return [r for r in data[k] if isinstance(r,(dict,list))]
            # 若是 mapping: frame_id -> record
            kv = []
            for k,v in data.items():
                if isinstance(k,str) and k.isdigit() and isinstance(v,dict):
                    vv = v.copy(); vv.setdefault('frame_id', int(k)); kv.append(vv)
            return kv or [data]
    except Exception:
        pass
    # fallback: jsonl
    recs = []
    for line in txt.splitlines():
        line = line.strip()
        if not line: continue
        try:
            recs.append(json.loads(line))
        except Exception:
            pass
    return recs

# Pose record → (bbox, keypoints[17], img_w, img_h)

def extract_pose_from_record(rec: dict) -> Tuple[Optional[List[float]], List[Dict], float, float]:
    # 支援通用格式（boxes/keypoints），也容忍 person 格式
    if isinstance(rec, dict) and rec.get("persons"):
        person = max(rec.get("persons", []), key=lambda x: x.get("score", 0.0))
        bbox = person.get("bbox")
        kps  = person.get("keypoints") or []
        kps_list = ([{"x": float(k.get("x",0.0)), "y": float(k.get("y",0.0)), "conf": float(k.get("conf", k.get("confidence",1.0)))} for k in kps[:17]] if kps else [])
        img_w = rec.get("image_size",{}).get("width", 640.0) or 640.0
        img_h = rec.get("image_size",{}).get("height",480.0) or 480.0
        return bbox, kps_list, float(img_w), float(img_h)

    boxes = (rec.get('boxes') if isinstance(rec, dict) else None) or []
    kps_all = (rec.get('keypoints') if isinstance(rec, dict) else None) or []
    # 選最大的 bbox 當主體
    best_i, best_area = 0, -1
    for i,b in enumerate(boxes):
        if not (isinstance(b,(list,tuple)) and len(b)==4):
            continue
        x1,y1,x2,y2 = b
        area = max(0,x2-x1)*max(0,y2-y1)
        if area>best_area: best_i=i; best_area=area
    bbox = boxes[best_i] if boxes else None
    kp_raw = kps_all[best_i] if (kps_all and best_i < len(kps_all)) else (kps_all[0] if kps_all else [])
    kps_list = ([{"x":float(x),"y":float(y),"conf":1.0} for (x,y) in kp_raw[:17]] if kp_raw else [])
    # 估計影像尺寸：取座標最大值
    xs, ys = [], []
    for b in boxes:
        if isinstance(b,(list,tuple)) and len(b)==4:
            xs += [b[0], b[2]]; ys += [b[1], b[3]]
    for grp in kps_all:
        for pt in grp:
            if isinstance(pt,(list,tuple)) and len(pt)>=2:
                xs.append(pt[0]); ys.append(pt[1])
    img_w = max(1.0, max(xs) if xs else 640.0)
    img_h = max(1.0, max(ys) if ys else 480.0)
    return bbox, kps_list, img_w, img_h

# Object record → list of {class_name, bbox}

def extract_objects_from_record(rec: dict) -> List[dict]:
    objs = (rec.get('objects') if isinstance(rec, dict) else None) \
        or (rec.get('detections') if isinstance(rec, dict) else None) \
        or (rec.get('boxes') if isinstance(rec, dict) else None) \
        or (rec.get('bboxes') if isinstance(rec, dict) else None) \
        or (rec.get('predictions') if isinstance(rec, dict) else None)
    if objs is None:
        return []
    out = []
    for o in objs:
        if not isinstance(o, dict):
            continue
        name = None
        for key in ("cls_name","class_name","name","label"):
            val = o.get(key)
            if isinstance(val,str) and val.strip():
                name = val.strip(); break
        bbox = o.get('bbox') or o.get('xyxy') or o.get('box') or o.get('bbox_xyxy')
        if name is None or bbox is None:
            continue
        out.append({'class_name': name, 'bbox': bbox})
    return out

# =============== 骨架完整性（第一幀用） ===============

def frame_has_full_skeleton(bbox, kps, *, kp_need=17, require_bbox=True):
    if require_bbox and bbox is None:
        return False
    cnt = 0
    for j in range(min(17, len(kps))):
        x = kps[j].get('x', None); y = kps[j].get('y', None)
        if x is None or y is None: continue
        if not np.isfinite(x) or not np.isfinite(y): continue
        conf = float(kps[j].get('conf', kps[j].get('confidence', 1.0)))
        if conf < KP_CONF_TH: continue
        cnt += 1
    return cnt >= kp_need

# ===================== Kalman Filter（2D 常速模型，一點一濾） =====================
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
    """半視窗 KF：0~half 覆蓋，之後每 step 覆蓋尾段（與 trainer/online 一致）"""
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

# ===================== Motion 特徵（與 trainer 一致，dt=1 不用 timestamp） =====================

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
        if conf >= KP_CONF_TH and ("x" in d) and ("y" in d):
            return (float(d["x"]) / img_w, float(d["y"]) / img_h)
    return None

def compute_motion_feats_with_mask_from_parsed(parsed, conf_th=0.3):
    """
    parsed: list of (bbox, kps_list, detections, img_w, img_h)
    回傳:
      feats: (T, 9)  -> [v_y, a_y, v_h, a_h, v_A, a_A, dtrunk, dkneeL, dkneeR]
      mask:  (T,)    -> 0/1 有效幀
    """
    T = len(parsed)
    if T == 0:
        return np.zeros((0,9), np.float32), np.zeros((0,), np.float32)

    ycom, hgt, area, trunk, kneeL, kneeR = [], [], [], [], [], []

    for (bbox, kps, _, img_w, img_h) in parsed:
        # y_com（優先髖 11/12，其次肩 5/6，最後高 conf 的平均）
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

        # 身高 proxy 與面積（歸一化）
        if bbox and all(k in bbox for k in ("w","h")):
            h = float(bbox["h"]) / img_h; w = float(bbox["w"]) / img_w
        else:
            ys = [float(p["y"]) / img_h for p in kps if float(p.get("conf", p.get("confidence",1.0))) >= conf_th and ("y" in p)]
            if len(ys) >= 2:
                h = max(ys) - min(ys); w = 0.4 * h
            else:
                h=None; w=None
        hgt.append(h)
        area.append((w*h) if (w is not None and h is not None) else None)

        # 軀幹角 vs 垂直、膝角
        shL=_kp_xy(kps,5,img_w,img_h,conf_th); shR=_kp_xy(kps,6,img_w,img_h,conf_th)
        hpL=_kp_xy(kps,11,img_w,img_h,conf_th); hpR=_kp_xy(kps,12,img_w,img_h,conf_th)
        knL=_kp_xy(kps,13,img_w,img_h,conf_th); anL=_kp_xy(kps,15,img_w,img_h,conf_th)
        knR=_kp_xy(kps,14,img_w,img_h,conf_th); anR=_kp_xy(kps,16,img_w,img_h,conf_th)
        if shL and shR and hpL and hpR:
            sh=((shL[0]+shR[0])/2,(shL[1]+shR[1])/2)
            hp=((hpL[0]+hpR[0])/2,(hpL[1]+hpR[1])/2)
            vec=(hp[0]-sh[0], hp[1]-sh[1])
            ang=abs(math.atan2(vec[0], vec[1]))  # 0=垂直，越大越斜
        else:
            ang=None
        trunk.append(ang)
        kneeL.append(_angle(hpL, knL, anL))
        kneeR.append(_angle(hpR, knR, anR))

    # 小洞線性補（連續缺值<=3）
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

    # 差分（dt=1）
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
        np.clip(v_y, -_MOTION_CLIP_V, _MOTION_CLIP_V),
        np.clip(a_y, -_MOTION_CLIP_A, _MOTION_CLIP_A),
        np.clip(v_h, -_MOTION_CLIP_V, _MOTION_CLIP_V),
        np.clip(a_h, -_MOTION_CLIP_A, _MOTION_CLIP_A),
        np.clip(v_A, -_MOTION_CLIP_V, _MOTION_CLIP_V),
        np.clip(a_A, -_MOTION_CLIP_A, _MOTION_CLIP_A),
        np.clip(dtrunk, -_MOTION_CLIP_V, _MOTION_CLIP_V),
        np.clip(dkneeL, -_MOTION_CLIP_V, _MOTION_CLIP_V),
        np.clip(dkneeR, -_MOTION_CLIP_V, _MOTION_CLIP_V),
    ], axis=1).astype(np.float32)  # (T,9)

    # 有效幀 mask：關鍵點命中數>=2 或 有 bbox
    valid=[]
    for (bbox, kps, _, _, _) in parsed:
        cnt=0
        for j in (11,12,5,6,13,14):
            if j < len(kps):
                conf=float(kps[j].get("conf", kps[j].get("confidence",1.0)))
                if conf>=KP_CONF_TH: cnt+=1
        ok=(cnt>=2) or (bbox is not None)
        valid.append(1.0 if ok else 0.0)
    valid=np.array(valid,np.float32)
    return feats, valid

# ===================== 模型（與訓練一致：支援 motion + mask） =====================
class SpaceCNN(nn.Module):
    def __init__(self, in_ch, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,64,3,padding=1), nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(256,out_dim)
    def forward(self,x):
        return self.fc(self.net(x).flatten(1))

class TemporalHead(nn.Module):
    def __init__(self, in_dim, num_classes, mode="attn", dropout=0.0):
        super().__init__()
        self.mode = mode
        self.drop = nn.Dropout(dropout) if dropout>0 else nn.Identity()
        if mode == "attn":
            self.attn = nn.Linear(in_dim,1)
        self.fc = nn.Linear(in_dim, num_classes)
    def forward(self, seq_feats, mask: Optional[torch.Tensor]=None):
        # seq_feats: (B,T,D), mask: (B,T) in {0,1}
        if (mask is not None) and (self.mode in ("mean","attn")):
            if self.mode == "mean":
                m = mask.unsqueeze(-1)
                den = m.sum(dim=1).clamp_min(1e-6)
                g = (seq_feats * m).sum(dim=1) / den
            else:
                a = self.attn(seq_feats).squeeze(-1)       # (B,T)
                # 👇 防呆：若某列全 0，改成全 1，避免 softmax([-inf,...]) 造成 NaN
                zero_rows = (mask.sum(dim=1) == 0)
                if zero_rows.any():
                    mask = mask.clone()
                    mask[zero_rows] = 1.0
                # 👇 用大負數，不用 -inf，避免 0/0
                a = a.masked_fill((mask<=0), -1e4)
                w = torch.softmax(a, dim=1).unsqueeze(-1)  # (B,T,1)
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
                 bidirectional=False, temporal_pool="attn", dropout=0.3, motion_dim=0):
        super().__init__()
        self.cnn = SpaceCNN(in_ch, cnn_out)
        self.motion_dim = int(motion_dim) if motion_dim else 0
        lstm_in = cnn_out + self.motion_dim
        self.lstm = nn.LSTM(lstm_in, lstm_h, lstm_layers, batch_first=True, bidirectional=bidirectional)
        feat_dim = lstm_h * (2 if bidirectional else 1)
        self.head = TemporalHead(feat_dim, num_classes, mode=temporal_pool, dropout=dropout)
    def forward(self, x, motion: Optional[torch.Tensor]=None, mask: Optional[torch.Tensor]=None):
        # x: (B,T,C,H,W)  motion: (B,T,D)  mask: (B,T)
        B,T,C,Hh,Ww = x.shape
        z = self.cnn(x.view(B*T, C, Hh, Ww)).view(B, T, -1)  # (B,T,cnn_out)
        if self.motion_dim > 0 and motion is not None:
            z = torch.cat([z, motion], dim=-1)                # (B,T,cnn_out+D)
        out, _ = self.lstm(z)                                 # (B,T,H)
        return self.head(out, mask=mask)

# ===================== 推論主流程 =====================

def stable_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # 數值穩定：清 NaN/Inf + 減最大值
    logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
    logits = logits - logits.max(dim=dim, keepdim=True).values
    return torch.softmax(logits, dim=dim)


def run_on_json(pose_json: str, object_json: Optional[str]=None):
    poses = read_any_json(pose_json)
    objs  = read_any_json(object_json) if object_json else None

    T = len(poses)
    print(f"[Debug] T={T}, window={WINDOW}, stride={STRIDE}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 計算輸入通道數（與訓練一致）
    in_ch = 1 + 1 + 17 + (len(COCO_EDGES) if INCLUDE_BONE_LINES else 0) + len(OBJECT_CLASSES) + 2

    # 讀類別 & 權重
    class_names = None
    sd = torch.load(MODEL_PATH, map_location=device)
    state = None
    if isinstance(sd, dict) and 'model_state' in sd:
        state = sd['model_state']
        class_names = sd.get('class_names')
    elif isinstance(sd, dict):
        state = sd
    else:
        raise RuntimeError('Unsupported checkpoint format')
    if class_names is None:
        if CLASSES_PATH and os.path.isfile(CLASSES_PATH):
            with open(CLASSES_PATH,'r',encoding='utf-8') as f:
                class_names = json.load(f)
        else:
            guess = os.path.join(os.path.dirname(MODEL_PATH), 'classes.json')
            if os.path.isfile(guess):
                with open(guess,'r',encoding='utf-8') as f:
                    class_names = json.load(f)
    if class_names is None:
        raise RuntimeError('Class names not found. 請提供 CLASSES_PATH 或使用含 class_names 的 ckpt')

    # 建模（含 motion_dim）
    model = CNNLSTM(in_ch=in_ch, num_classes=len(class_names), cnn_out=256,
                    lstm_h=LSTM_HIDDEN, lstm_layers=2, bidirectional=BIDIRECTIONAL,
                    temporal_pool=TEMPORAL_POOL, dropout=DROPOUT,
                    motion_dim=(_MOTION_DIM if _USE_MOTION else 0)).to(device)
    model.load_state_dict(state)
    model.eval()

    # 物件取得函式
    def get_obj(i):
        if objs is None: return []
        if i < len(objs):
            return extract_objects_from_record(objs[i])
        return []

    results = []
    with torch.no_grad():
        for start in range(0, max(0, T - WINDOW + 1), STRIDE):
            parsed = []
            for i in range(start, start + WINDOW):
                bbox, kps, img_w, img_h = extract_pose_from_record(poses[i])
                dets = get_obj(i)
                parsed.append((bbox, kps, dets, img_w, img_h))

            # 第一幀完整性檢查（不完整就跳過這個 20 幀結果）
            if REQUIRE_FULL_FIRST_FRAME:
                bbox0, kps0, _, _, _ = parsed[0]
                if not frame_has_full_skeleton(bbox0, kps0, kp_need=17, require_bbox=True):
                    continue

            # 半視窗 KF 平滑（覆蓋尾段）
            if ENABLE_KALMAN:
                window_kps = [kps for (_, kps, _, _, _) in parsed]
                half_len = int(HALF_LEN_OVERRIDE) if HALF_LEN_OVERRIDE else max(1, WINDOW//2)
                smoothed = kalman_smooth_kps(window_kps,
                                             half_slide=KALMAN_HALF_SLIDE,
                                             half_len=half_len,
                                             require_full_first=True,
                                             step=STRIDE)
                parsed = [(bbox, smoothed[i], dets, img_w, img_h) for i,(bbox, _, dets, img_w, img_h) in enumerate(parsed)]

            # Motion + mask（與訓練一致）
            motion_feats, valid_mask = compute_motion_feats_with_mask_from_parsed(parsed, conf_th=KP_CONF_TH)

            # 轉 relation maps
            cfg = RelationMapConfig(H=H, W=W, sigma_kp=SIGMA_KP, kp_conf_th=KP_CONF_TH,
                                     include_bone_lines=INCLUDE_BONE_LINES, object_classes=OBJECT_CLASSES)
            clips = [rasterize_frame(bbox, kps, dets, img_w, img_h, cfg) for (bbox,kps,dets,img_w,img_h) in parsed]

            x = torch.from_numpy(np.stack(clips)).unsqueeze(0).float().to(device)            # (1,T,C,H,W)
            M = torch.from_numpy(motion_feats).unsqueeze(0).float().to(device) if _USE_MOTION else None  # (1,T,9)
            mask = torch.from_numpy(valid_mask).unsqueeze(0).float().to(device)                           # (1,T)

            # 清理 NaN/Inf（保守）
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            if M is not None:
                M = torch.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)

            logits = model(x, motion=M, mask=mask)
            prob = stable_softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(prob.argmax())
            results.append({
                'start_frame': start,
                'end_frame': start + WINDOW - 1,
                'pred_idx': pred_idx,
                'pred': class_names[pred_idx],
                'probs': prob.tolist(),
            })
    return results, class_names

# ===================== 輸出 CSV（可選） =====================

def save_results_csv(results: list, out_csv: str, class_names: List[str]):
    import csv
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        header = ['start_frame','end_frame','pred','pred_idx'] + [f'p_{c}' for c in class_names]
        w.writerow(header)
        for r in results:
            row = [r['start_frame'], r['end_frame'], r['pred'], r['pred_idx']] + r['probs']
            w.writerow(row)

# ===================== Main =====================
if __name__ == '__main__':
    results, classes = run_on_json(POSE_JSON, OBJECT_JSON)
    for r in results:
        import numpy as _np
        print(f"frames {r['start_frame']:>5}-{r['end_frame']:<5} | pred={r['pred']} | probs={_np.round(r['probs'],3)}")
    if OUT_CSV:
        save_results_csv(results, OUT_CSV, classes)
        print(f"Saved CSV: {OUT_CSV}")
