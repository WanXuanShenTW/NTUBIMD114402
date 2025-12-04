# -*- coding: utf-8 -*-
"""
multi_cnn_lstm_loader.py (對齊 multi_cnn_lstm_trainer_rareaware v2)

功能：
- ✅ 自動讀取同名影片檔以鎖定解析度 (解決座標抖動)
- ✅ 對齊 Motion Clip, GroupNorm, Channel 數
- ✅ 支援多類別推論
- ✅ 移除 Kalman，改用插值補點

使用方式：
1. 修改 MODEL_PATH, POSE_JSON, OBJECT_JSON
2. python multi_cnn_lstm_loader.py
"""

import os, json, math
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import cv2

# ===================== 常數設定 =====================
MODEL_PATH      = "outputs/models/HM/focal/multi/best.pt"       
CLASSES_PATH    = "outputs/models/HM/focal/multi/classes.json"   

# 測試檔 (請依需求修改)
POSE_JSON   = "outputs/skeletons/binary/YOLO-pose/non_fall/Office_video (18)_back_front.json"  
OBJECT_JSON = "outputs/skeletons/binary/YOLO-detect/non_fall/Office_video (18)_back_front.json" # 若無則設 None
OUT_CSV     = None 

# 影片根目錄 (用於讀取解析度，若測試檔是單獨的 JSON 且有同名影片在旁，也可設為該目錄)
VIDEO_ROOT  = "medias/train_video/multi" 
VIDEO_EXTS  = [".mp4", ".avi", ".mov", ".mkv"]

# Relation Map (需與 Trainer 一致)
H, W                   = 64, 64
INCLUDE_BONE_LINES     = True   # Multi Trainer 有開
OBJECT_CLASSES         = ["bed", "chair", "bench"] 
SIGMA_KP               = 2.0
SIGMA_OBJ              = 4.0    
KP_CONF_TH             = 0.4    

# LSTM
LSTM_HIDDEN            = 256
BIDIRECTIONAL          = False
TEMPORAL_POOL          = "attn" 
WINDOW                 = 20     # Multi Trainer 設定為 20
STRIDE                 = 5
DROPOUT                = 0.3

# Motion Clip (10 FPS)
_USE_MOTION            = True   
_MOTION_DIM            = 9
_MOTION_CLIP_V         = 5.0    # 10 FPS 建議值
_MOTION_CLIP_A         = 10.0

# 前處理
REQUIRE_FULL_FIRST_FRAME = False 
MAX_MISSING_FILL       = 3      

# 骨架定義
COCO_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]

class RelationMapConfig:
    def __init__(self, H=64, W=64, sigma_kp=2.0, sigma_obj=4.0, kp_conf_th=0.4,
                 include_bone_lines=False, object_classes=None):
        self.H = int(H); self.W = int(W)
        self.sigma_kp = float(sigma_kp)
        self.sigma_obj = float(sigma_obj)
        self.kp_conf_th = float(kp_conf_th)
        self.include_bone_lines = bool(include_bone_lines)
        self.object_classes = [c.strip() for c in (object_classes or []) if c.strip()]

# ===================== 繪圖工具 =====================
def gaussian2d(shape, sigma):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y)/(2*sigma*sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_gaussian_on_channel(channel, cx, cy, sigma, mag=1.0):
    if cx is None or cy is None: return
    Hc, Wc = channel.shape
    x = int(cx); y = int(cy)
    if x < 0 or x >= Wc or y < 0 or y >= Hc: return
    size = int(6 * sigma + 3)
    g = gaussian2d((size, size), sigma)
    g *= mag 
    x0 = x - size // 2; y0 = y - size // 2
    x1 = x0 + size;    y1 = y0 + size
    g_x0 = max(0, -x0); g_y0 = max(0, -y0)
    g_x1 = g_x0 + min(Wc, x1) - max(0, x0)
    g_y1 = g_y0 + min(Hc, y1) - max(0, y0)
    c_x0 = max(0, x0); c_y0 = max(0, y0)
    c_x1 = min(Wc, x1); c_y1 = min(Hc, y1)
    if c_x1 > c_x0 and c_y1 > c_y0:
        channel[c_y0:c_y1, c_x0:c_x1] = np.maximum(
            channel[c_y0:c_y1, c_x0:c_x1],
            g[g_y0:g_y1, g_x0:g_x1]
        )

def rasterize_frame(bbox, kps_list, dets, img_w, img_h, cfg: RelationMapConfig):
    Hc, Wc = cfg.H, cfg.W
    
    num_kp = 17
    num_edges = len(COCO_EDGES) if cfg.include_bone_lines else 0
    num_obj_ch = len(cfg.object_classes)
    num_coord = 2
    C = num_kp + num_edges + num_obj_ch + num_coord
    
    canvas = np.zeros((C, Hc, Wc), dtype=np.float32)
    ch = 0

    # 1. Keypoints
    if kps_list:
        current_kps = kps_list[:num_kp]
        for i, kp in enumerate(current_kps):
            target_ch = ch + i 
            try:
                conf = float(kp.get('conf', kp.get('confidence', 1.0)))
                if conf < cfg.kp_conf_th: continue
                raw_x, raw_y = float(kp['x']), float(kp['y'])
                x = np.clip((raw_x / img_w) * Wc, 0, Wc - 1)
                y = np.clip((raw_y / img_h) * Hc, 0, Hc - 1)
                draw_gaussian_on_channel(canvas[target_ch], x, y, cfg.sigma_kp, mag=1.0)
            except Exception: pass
    ch += num_kp

    # 2. Edges
    if cfg.include_bone_lines and kps_list:
        # 簡單畫線 (不帶高斯，或者你可以用 cv2.line)
        pts = []
        for i in range(num_kp):
            if i < len(kps_list):
                kp = kps_list[i]
                conf = float(kp.get('conf', 1.0))
                if conf >= cfg.kp_conf_th:
                    rx, ry = float(kp['x']), float(kp['y'])
                    x = int(np.clip((rx / img_w) * Wc, 0, Wc - 1))
                    y = int(np.clip((ry / img_h) * Hc, 0, Hc - 1))
                    pts.append((x, y))
                else: pts.append(None)
            else: pts.append(None)
            
        for e_idx, (a, b) in enumerate(COCO_EDGES):
            if pts[a] and pts[b]:
                # 在對應 channel 畫線
                cv2.line(canvas[ch + e_idx], pts[a], pts[b], 1.0, 1)
    ch += num_edges 

    # 3. Objects
    if num_obj_ch > 0 and dets:
        cls2ch = {name: i for i, name in enumerate(cfg.object_classes)}
        for d in dets or []:
            name = d.get("cls_name") or d.get("class_name") or d.get("label") or d.get("name")
            if name not in cls2ch: continue
            bb = d.get("bbox") or d.get("xyxy") or {}
            bxyxy = _bbox_xyxy_from_any(bb)
            if bxyxy is None: continue
            x1, y1, x2, y2 = bxyxy
            cx = (x1 + x2) / 2.0; cy = (y1 + y2) / 2.0
            map_cx = np.clip((cx / img_w) * Wc, 0, Wc - 1)
            map_cy = np.clip((cy / img_h) * Hc, 0, Hc - 1)
            obj_idx = cls2ch[name]
            draw_gaussian_on_channel(canvas[ch + obj_idx], map_cx, map_cy, cfg.sigma_obj, mag=1.0)
    ch += num_obj_ch

    # 4. CoordConv
    xv = np.linspace(-1, 1, Wc)[None, :].repeat(Hc, 0)
    yv = np.linspace(-1, 1, Hc)[:, None].repeat(Wc, 1)
    canvas[ch] = xv; canvas[ch+1] = yv; ch += 2

    return canvas

# ===================== 模型定義 (GroupNorm) =====================
class SpaceCNN(nn.Module):
    def __init__(self, in_ch, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GroupNorm(8, 128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.GroupNorm(8, 256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.proj = nn.Linear(256, out_dim)
    def forward(self, x):
        z = self.net(x).flatten(1)
        return self.proj(z)

class TemporalHead(nn.Module):
    def __init__(self, in_dim, num_classes, mode="attn", dropout=0.0):
        super().__init__()
        self.mode = mode
        self.drop = nn.Dropout(dropout) if dropout>0 else nn.Identity()
        if mode == "attn":
            self.attn = nn.Linear(in_dim,1)
        self.fc = nn.Linear(in_dim, num_classes)
    def forward(self, seq_feats, mask: Optional[torch.Tensor]=None):
        if (mask is not None) and (self.mode in ("mean","attn")):
            mask_sum = mask.sum(dim=1, keepdim=True)
            mask = torch.where(mask_sum > 0, mask, torch.ones_like(mask)) # 防 NaN
            
            if self.mode == "mean":
                m = mask.unsqueeze(-1)
                den = m.sum(dim=1).clamp_min(1e-6)
                g = (seq_feats * m).sum(dim=1) / den
            else:
                a = self.attn(seq_feats).squeeze(-1)
                a = a.masked_fill((mask<=0), -1e9)
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
                 bidirectional=False, temporal_pool="attn", dropout=0.3, motion_dim=0):
        super().__init__()
        self.cnn = SpaceCNN(in_ch, cnn_out)
        self.motion_dim = int(motion_dim) if motion_dim else 0
        lstm_in = cnn_out + self.motion_dim
        self.lstm = nn.LSTM(lstm_in, lstm_h, lstm_layers, batch_first=True, bidirectional=bidirectional)
        feat_dim = lstm_h * (2 if bidirectional else 1)
        self.head = TemporalHead(feat_dim, num_classes, mode=temporal_pool, dropout=dropout)
    def forward(self, x, motion: Optional[torch.Tensor]=None, mask: Optional[torch.Tensor]=None):
        B,T,C,Hh,Ww = x.shape
        z = self.cnn(x.view(B*T, C, Hh, Ww)).view(B, T, -1)
        if self.motion_dim > 0 and motion is not None:
            z = torch.cat([z, motion], dim=-1)
        out, _ = self.lstm(z)
        return self.head(out, mask=mask)

# ===================== 補點與工具 =====================
def fill_missing_kps_inference(parsed_window, max_missing=3):
    T = len(parsed_window)
    if T < 2: return parsed_window
    num_kp = 17
    data = np.full((T, num_kp, 3), np.nan, dtype=np.float32)
    for t, item in enumerate(parsed_window):
        kps = item[1]
        for k_idx in range(min(len(kps), num_kp)):
            k_data = kps[k_idx]
            if 'x' in k_data and 'y' in k_data:
                conf = float(k_data.get('conf', k_data.get('confidence', 1.0)))
                if conf > 0.1:
                    data[t, k_idx, 0] = float(k_data['x'])
                    data[t, k_idx, 1] = float(k_data['y'])
                    data[t, k_idx, 2] = conf
    for k in range(num_kp):
        last_val = None
        missing_count = 0
        for t in range(T):
            curr = data[t, k]
            if not np.isnan(curr[0]):
                last_val = curr.copy()
                missing_count = 0
            elif last_val is not None and missing_count < max_missing:
                data[t, k] = last_val
                missing_count += 1
            else:
                missing_count += 1
    new_window = []
    for t in range(T):
        bbox, _, dets, iw, ih = parsed_window[t]
        new_kps = []
        for k in range(num_kp):
            x, y, c = data[t, k]
            if np.isnan(x): new_kps.append({'x':0.0,'y':0.0,'conf':0.0})
            else: new_kps.append({'x':x,'y':y,'conf':c})
        new_window.append((bbox, new_kps, dets, iw, ih))
    return new_window

def read_any_json(path: Optional[str]):
    if not path: return []
    with open(path, 'r', encoding='utf-8') as f: txt = f.read().strip()
    if not txt: return []
    try:
        data = json.loads(txt)
        if isinstance(data, list): return [r for r in data if isinstance(r,(dict,list))]
        if isinstance(data, dict):
            for k in ('frames','data','records','annotations','items','results'):
                if isinstance(data.get(k), list): return [r for r in data[k] if isinstance(r,(dict,list))]
            kv = []
            for k,v in data.items():
                if isinstance(k,str) and k.isdigit() and isinstance(v,dict):
                    vv = v.copy(); vv.setdefault('frame_id', int(k)); kv.append(vv)
            return kv or [data]
    except: pass
    recs = []
    for line in txt.splitlines():
        try: recs.append(json.loads(line))
        except: pass
    return recs

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
        return float(b[0]),float(b[1]),float(b[2]),float(b[3])
    return None

# ===================== Pose Extraction with Fixed Resolution =====================
def extract_pose_from_record(rec: dict, fixed_w=None, fixed_h=None):
    if isinstance(rec, dict) and rec.get("persons"):
        person = max(rec.get("persons", []), key=lambda x: x.get("score", 0.0))
        bbox = person.get("bbox")
        kps = person.get("keypoints") or []
        kps_list = ([{"x": float(k.get("x",0.0)), "y": float(k.get("y",0.0)), "conf": float(k.get("conf", k.get("confidence",1.0)))} for k in kps[:17]] if kps else [])
        img_w = rec.get("image_size",{}).get("width", 0.0)
        img_h = rec.get("image_size",{}).get("height",0.0)
    else:
        boxes = (rec.get('boxes') if isinstance(rec, dict) else None) or []
        kps_all = (rec.get('keypoints') if isinstance(rec, dict) else None) or []
        best_i, best_area = 0, -1
        for i,b in enumerate(boxes):
            if not (isinstance(b,(list,tuple)) and len(b)==4): continue
            x1,y1,x2,y2 = b
            area = max(0,x2-x1)*max(0,y2-y1)
            if area>best_area: best_i=i; best_area=area
        bbox = boxes[best_i] if boxes else None
        kp_raw = kps_all[best_i] if (kps_all and best_i < len(kps_all)) else (kps_all[0] if kps_all else [])
        kps_list = ([{"x":float(x),"y":float(y),"conf":1.0} for (x,y) in kp_raw[:17]] if kp_raw else [])
        img_w = 0.0; img_h = 0.0

    if fixed_w and fixed_h and fixed_w > 0 and fixed_h > 0:
        final_w = fixed_w
        final_h = fixed_h
    elif img_w > 0 and img_h > 0:
        final_w = img_w
        final_h = img_h
    else:
        final_w = 1920.0 
        final_h = 1080.0
        
    return bbox, kps_list, float(final_w), float(final_h)

def extract_objects_from_record(rec: dict):
    objs = (rec.get('objects') if isinstance(rec, dict) else None) or \
           (rec.get('detections') if isinstance(rec, dict) else None) or []
    out = []
    for o in objs:
        if not isinstance(o, dict): continue
        name = o.get('cls_name') or o.get('class_name') or o.get('name') or o.get('label')
        bbox = o.get('bbox') or o.get('xyxy')
        if name and bbox:
            out.append({'class_name': name, 'bbox': bbox})
    return out

# ===================== Motion Helpers =====================
def _safe_mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals)/len(vals) if vals else None

def _angle(a, b, c):
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

def compute_motion_feats_with_mask_from_parsed(parsed, conf_th=KP_CONF_TH):
    T = len(parsed)
    if T == 0: return np.zeros((0,9), np.float32), np.zeros((0,), np.float32)
    ycom, hgt, area, trunk, kneeL, kneeR = [], [], [], [], [], []
    for (bbox, kps, _d, img_w, img_h) in parsed:
        hips = [_kp_xy(kps,11,img_w,img_h,conf_th), _kp_xy(kps,12,img_w,img_h,conf_th)]
        hs = [p for p in hips if p is not None]
        if hs:
            y_c = _safe_mean([p[1] for p in hs])
        else:
            shs = [_kp_xy(kps,5,img_w,img_h,conf_th), _kp_xy(kps,6,img_w,img_h,conf_th)]
            ss = [p for p in shs if p is not None]
            if ss: y_c = _safe_mean([p[1] for p in ss])
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
            if len(ys) >= 2: h = max(ys)-min(ys); w = 0.4*h
            else: h=None; w=None
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
        else: ang=None
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
            else: i+=1
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
        np.clip(v_y, -_MOTION_CLIP_V, _MOTION_CLIP_V),
        np.clip(a_y, -_MOTION_CLIP_A, _MOTION_CLIP_A),
        np.clip(v_h, -_MOTION_CLIP_V, _MOTION_CLIP_V),
        np.clip(a_h, -_MOTION_CLIP_A, _MOTION_CLIP_A),
        np.clip(v_A, -_MOTION_CLIP_V, _MOTION_CLIP_V),
        np.clip(a_A, -_MOTION_CLIP_A, _MOTION_CLIP_A),
        np.clip(dtrunk, -_MOTION_CLIP_V, _MOTION_CLIP_V),
        np.clip(dkneeL, -_MOTION_CLIP_V, _MOTION_CLIP_V),
        np.clip(dkneeR, -_MOTION_CLIP_V, _MOTION_CLIP_V),
    ], axis=1).astype(np.float32)
    valid=[]
    for (bbox, kps, _d, _iw, _ih) in parsed:
        cnt=0
        for j in (11,12,5,6,13,14):
            if j < len(kps):
                conf=float(kps[j].get("conf", kps[j].get("confidence",1.0)))
                if conf>=KP_CONF_TH: cnt+=1
        ok=(cnt>=2) or (bbox is not None)
        valid.append(1.0 if ok else 0.0)
    valid=np.array(valid,np.float32)
    return feats, valid

# ===================== Video Resolution Helpers =====================
def frame_has_full_skeleton(bbox, kps, *, kp_need=17, require_bbox=True):
    # 如果設定需要 bbox 但 bbox 是 None，則視為不完整
    if require_bbox and bbox is None:
        return False
    
    # 計算有效骨架點數量
    cnt = 0
    # kps 是 list of dict {'x':..., 'y':..., 'conf':...}
    for j in range(min(17, len(kps))):
        pt = kps[j]
        x = pt.get('x')
        y = pt.get('y')
        # 檢查座標是否存在且有效
        if x is None or y is None: continue
        if not (math.isfinite(x) and math.isfinite(y)): continue
        
        # 檢查信心度
        conf = float(pt.get('conf', pt.get('confidence', 1.0)))
        if conf >= KP_CONF_TH:
            cnt += 1
            
    return cnt >= kp_need

def try_get_video_resolution(json_path):
    if not json_path: return 0, 0
    base_no_ext = os.path.splitext(json_path)[0]
    
    # 嘗試在 VIDEO_ROOT 及其子目錄中搜尋
    found_path = None
    
    # 策略 1: 直接在 VIDEO_ROOT 下找同名檔案
    for ext in VIDEO_EXTS:
        test_path = os.path.join(VIDEO_ROOT, os.path.basename(base_no_ext) + ext)
        if os.path.exists(test_path):
            found_path = test_path
            break
            
    # 策略 2: 在 VIDEO_ROOT 的子目錄 (class folder) 中找
    if not found_path:
        for root, dirs, files in os.walk(VIDEO_ROOT):
            for ext in VIDEO_EXTS:
                if (os.path.basename(base_no_ext) + ext) in files:
                    found_path = os.path.join(root, os.path.basename(base_no_ext) + ext)
                    break
            if found_path: break

    # 策略 3: 原地找 (如果影片和 JSON 放一起)
    if not found_path:
        for ext in VIDEO_EXTS:
            test_path = base_no_ext + ext
            if os.path.exists(test_path):
                found_path = test_path
                break

    if found_path:
        try:
            cap = cv2.VideoCapture(found_path)
            if cap.isOpened():
                w = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                print(f"[Info] Loader found video: {os.path.basename(found_path)} ({int(w)}x{int(h)})")
                return w, h
        except:
            pass
            
    print(f"[Warn] Loader could not find video for {os.path.basename(json_path)}, using fallback resolution.")
    return 0, 0

# ===================== 主流程 =====================
def run_on_json(pose_json: str, object_json: Optional[str]=None):
    poses = read_any_json(pose_json)
    objs  = read_any_json(object_json) if object_json else None
    T = len(poses)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 通道數 (對齊 Trainer)
    in_ch = 17 + (len(COCO_EDGES) if INCLUDE_BONE_LINES else 0) + len(OBJECT_CLASSES) + 2
    
    if not os.path.exists(MODEL_PATH): raise FileNotFoundError(MODEL_PATH)
    ckpt = torch.load(MODEL_PATH, map_location=device)
    state = ckpt['model_state'] if 'model_state' in ckpt else ckpt
    class_names = ckpt.get('class_names')
    
    if class_names is None and os.path.exists(CLASSES_PATH):
        with open(CLASSES_PATH) as f: class_names = json.load(f)
    if not class_names: raise RuntimeError("Class names not found")

    model = CNNLSTM(in_ch=in_ch, num_classes=len(class_names), cnn_out=256,
                    lstm_h=LSTM_HIDDEN, lstm_layers=2, bidirectional=BIDIRECTIONAL,
                    temporal_pool=TEMPORAL_POOL, dropout=DROPOUT,
                    motion_dim=(_MOTION_DIM if _USE_MOTION else 0)).to(device)
    
    model.load_state_dict(state, strict=False)
    model.eval()

    motion_stats = ckpt.get('motion_norm', {"mean": [0.0]*9, "std": [1.0]*9})
    mm = np.array(motion_stats['mean'], np.float32)
    ss = np.array(motion_stats['std'], np.float32)

    results = []

    def get_obj(i): return extract_objects_from_record(objs[i]) if objs and i < len(objs) else []

    # 嘗試取得影片解析度以鎖定座標
    fixed_w, fixed_h = try_get_video_resolution(pose_json)
    # fixed_w, fixed_h = 320, 180  # 強制鎖定解析度以符合訓練設定

    with torch.no_grad():
        for start in range(0, max(0, T - WINDOW + 1), STRIDE):
            parsed = []
            for i in range(start, start + WINDOW):
                # 傳入固定寬高
                bbox, kps, img_w, img_h = extract_pose_from_record(poses[i], fixed_w=fixed_w, fixed_h=fixed_h)
                dets = get_obj(i)
                parsed.append((bbox, kps, dets, img_w, img_h))

            if REQUIRE_FULL_FIRST_FRAME:
                if not frame_has_full_skeleton(parsed[0][0], parsed[0][1], kp_need=17, require_bbox=True):
                    continue
            
            parsed = fill_missing_kps_inference(parsed, max_missing=MAX_MISSING_FILL)

            # Motion
            motion_feats, valid_mask = compute_motion_feats_with_mask_from_parsed(parsed, conf_th=KP_CONF_TH)
            motion_feats = (motion_feats - mm) / ss

            # Rasterize
            cfg = RelationMapConfig(H=H, W=W, sigma_kp=SIGMA_KP, sigma_obj=SIGMA_OBJ, 
                                    kp_conf_th=KP_CONF_TH, include_bone_lines=INCLUDE_BONE_LINES, 
                                    object_classes=OBJECT_CLASSES)
            clips = [rasterize_frame(bbox, kps, dets, img_w, img_h, cfg) for (bbox,kps,dets,img_w,img_h) in parsed]

            x = torch.from_numpy(np.stack(clips)).unsqueeze(0).float().to(device)
            M = torch.from_numpy(motion_feats).unsqueeze(0).float().to(device) if _USE_MOTION else None
            mask = torch.from_numpy(valid_mask).unsqueeze(0).float().to(device)

            logits = model(x, motion=M, mask=mask)
            prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(prob.argmax())
            
            out_row = {
                'start_frame': start, 'end_frame': start + WINDOW - 1,
                'pred_idx': pred_idx, 'pred': class_names[pred_idx], 'probs': prob.tolist(),
            }
            results.append(out_row)
            
    return results, class_names

if __name__ == '__main__':
    res, cls = run_on_json(POSE_JSON, OBJECT_JSON)
    for r in res:
        probs_str = ", ".join([f"{cls[i]}:{p:.3f}" for i, p in enumerate(r['probs'])])
        print(f"{r['start_frame']:>3}-{r['end_frame']:<3} | Pred: {r['pred']} | {probs_str}")