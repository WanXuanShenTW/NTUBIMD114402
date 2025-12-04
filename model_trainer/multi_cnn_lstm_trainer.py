# -*- coding: utf-8 -*-
"""
multi_cnn_lstm_trainer_rareaware.py  (multi-minority/majority version)

新增：可用「陣列」設定多個少數類與多數類（像 class_whitelist/blacklist 一樣）。
- Config.minority_class_names: List[str]  把這些類當作 "少數（fall-like）"
- Config.majority_class_names: Optional[List[str]]  指定哪些類當作 "多數（non-fall-like）"
  （若不指定，預設為：所有非少數類 = 多數類）

策略：
- 對多數類做下採樣（non_minority_downsample_ratio）
- 對少數類做索引過採樣（minority_oversample_mult 或 minority_oversample_mult_map）
- 只對少數類啟用 on-the-fly 擴增（aug_for_minority=True）
- 其餘保持多分類輸出，不改變模型與標註

注意：
- 當 oversample（少數類倍增）>1（整體或任一類）時，會自動「暫時關閉 WeightedRandomSampler」避免雙重平衡。
- 若同時指定 minority_class_names 與 majority_class_names，兩者不可交集；並且都必須存在於資料集合中的類名。
"""

import os, json, random, math
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from tqdm import tqdm
from typing import Optional, List, Tuple, Dict, Union
from collections import Counter

# ===================== Config =====================
class Config:
    # 路徑
    pose_root = "outputs/skeletons/multi/YOLO-pose"
    obj_root  = "outputs/skeletons/multi/YOLO-detect"
    out_dir   = "outputs/models/HM/focal/multi"
    video_root = "medias/train_video/multi" # 請確認你的 multi 影片放在這
    video_exts = [".mp4", ".avi", ".mov", ".mkv"]
    
    # 是否使用物件框
    use_objects = True
    object_classes = ["bed", "chair", "bench"]  # 空則不建立物件通道

    # 時序
    window = 10
    stride = 5

    # Relation Map
    H, W = 64, 64
    include_bone_lines = True
    
    # === Heatmap 參數 ===
    sigma_kp = 2.0   # 骨架點的高斯 sigma
    sigma_obj = 4.0  # 物件中心填滿的高斯 sigma (較大)

    # 模型
    lstm_hidden  = 256
    bidirectional = False
    temporal_pool = "attn"       # "last" | "mean" | "attn"
    dropout = 0.3

    # 訓練
    epochs     = 60
    batch_size = 8
    lr         = 1e-5
    use_sampler = True
    loss = "ce"                  # "focal" | "ce"

    # 前處理
    require_full_first_frame = True
    half_len_override = None
    require_full_skeleton_all = False
    full_kp_min = 8
    bbox_required = False
    max_missing_interpolate = 3

    # 類別白/黑名單（擇一使用）
    class_whitelist: Optional[List[str]] = ["liestill", "sitstill", "walk"]  # ← 多個類，像 ["fall", "lie-rare"]
    class_blacklist: Optional[List[str]] = None

    # ===== 多少數（關鍵） =====
    minority_class_names: List[str] = ["liestill"]       # ← 多個少數類，像 ["fall", "lie-rare"]
    majority_class_names: Optional[List[str]] = None  # ← 可選；若不給，則「所有非少數」= 多數

    non_minority_downsample_ratio = 1        # 多數類保留比例（0.0~1.0；1.0 表示不下採樣）
    minority_oversample_mult: int = 1          # 少數類「統一倍數」（索引過採樣）；1=不過採樣
    # （可選）針對各少數類自訂倍數；若提供任何 >1，則使用此映射，忽略上面的整體倍數
    # 例：{"fall": 3, "lie-rare": 2}
    minority_oversample_mult_map: Dict[str, int] = {"liestill": 1}

    # ===== 只對少數類做 on-the-fly 擴增 =====
    aug_for_minority = False
    aug_prob = 0.40
    aug_hflip = True
    aug_affine_scale = (0.90, 1.10)
    aug_affine_rotate_deg = (-12.0, 12.0)
    aug_affine_translate = (-0.05, 0.05)
    aug_kp_noise_px = 2.0
    aug_kp_dropout_prob = 0.05

# ===================== 穩定預設 =====================
_SEED = 42
_KP_CONF_TH = 0.2
_SIGMA_KP = 3.0
_CNN_OUT = 256
_LSTM_LAYERS = 2
_WEIGHT_DECAY = 1e-4
_GRAD_CLIP = 1.0
_AMP = False
_EARLY_STOP_PATIENCE = 5
_SAVE_TOP_K = 3
_FOCAL_GAMMA = 2.0
_MOTION_CLIP_V = 5.0
_MOTION_CLIP_A = 10.0

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

# ===================== Relation Map 等（略） =====================
# ---（以下大段內容與先前版本一致；為節省篇幅只在關鍵處做「多少數/多多數」更新）---

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

class RelationMapConfig:
    def __init__(self, H=64, W=64, sigma_kp=3.0, sigma_obj=4.0 ,kp_conf_th=0.4,
                include_bone_lines=True, object_classes=None):
        self.H = int(H); self.W = int(W)
        self.sigma_kp = float(sigma_kp)
        self.sigma_obj = float(sigma_obj)
        self.kp_conf_th = float(kp_conf_th)
        self.include_bone_lines = bool(include_bone_lines)
        self.object_classes = [c.strip() for c in (object_classes or []) if c and c.strip()]

def gaussian2d(shape, sigma):
    """生成中心為 1.0 的高斯核"""
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y)/(2*sigma*sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_gaussian_on_channel(channel, cx, cy, sigma, mag=1.0):
    """
    在指定的 channel (H, W) 上，於 (cx, cy) 繪製高斯熱圖。
    使用 max 取值 (避免重疊時數值過大)。
    """
    if cx is None or cy is None:
        return
    
    Hc, Wc = channel.shape
    x = int(cx); y = int(cy)
    
    # 確保在畫布範圍內
    if x < 0 or x >= Wc or y < 0 or y >= Hc:
        return

    size = int(6 * sigma + 3)
    g = gaussian2d((size, size), sigma)
    g *= mag # 應用亮度 (信心度)

    x0 = x - size // 2; y0 = y - size // 2
    x1 = x0 + size;    y1 = y0 + size
    
    # 計算在 g 中的裁切範圍
    g_x0 = max(0, -x0); g_y0 = max(0, -y0)
    g_x1 = g_x0 + min(Wc, x1) - max(0, x0)
    g_y1 = g_y0 + min(Hc, y1) - max(0, y0)
    
    # 計算在 channel 中的裁切範圍
    c_x0 = max(0, x0); c_y0 = max(0, y0)
    c_x1 = min(Wc, x1); c_y1 = min(Hc, y1)

    if c_x1 > c_x0 and c_y1 > c_y0:
        # 使用 np.maximum 避免疊加過亮
        channel[c_y0:c_y1, c_x0:c_x1] = np.maximum(
            channel[c_y0:c_y1, c_x0:c_x1],
            g[g_y0:g_y1, g_x0:g_x1]
        )

def rasterize_frame(bbox, kps_list, dets, img_w, img_h, cfg: RelationMapConfig):
    Hc, Wc = cfg.H, cfg.W
    
    # === Channel 定義 ===
    # 1. 骨架點: 17 個 Channel (獨立)
    num_kp = 17
    # 2. 骨骼連線: (可選)
    num_edges = len(COCO_EDGES) if cfg.include_bone_lines else 0
    # 3. 物件: N 個 Channel (獨立)
    num_obj_ch = len(cfg.object_classes)
    # 4. CoordConv: 2 個 Channel (x, y)
    num_coord = 2
    
    # 總 Channel 數
    C = num_kp + num_edges + num_obj_ch + num_coord
    canvas = np.zeros((C, Hc, Wc), dtype=np.float32)
    ch = 0

    # --- 1. 繪製骨架點 (17 Channels) ---
    # 修正正規化邏輯：直接依比例縮放，不需額外減 mean 除 std
    if kps_list:
        # 只取前 17 點
        current_kps = kps_list[:num_kp]
        for i, kp in enumerate(current_kps):
            # 每個點佔用獨立 Channel
            target_ch = ch + i 
            try:
                conf = float(kp.get('conf', kp.get('confidence', 1.0)))
                if conf < cfg.kp_conf_th:
                    continue
                
                # 座標縮放 (直接映射)
                raw_x, raw_y = float(kp['x']), float(kp['y'])
                x = np.clip((raw_x / img_w) * Wc, 0, Wc - 1)
                y = np.clip((raw_y / img_h) * Hc, 0, Hc - 1)
                
                # 繪製高斯 (mag=1.0 固定亮度，或可改為 mag=conf)
                # 這裡我們先依之前的結論：位置準確比較重要，暫時用固定亮度 1.0，
                # 但若要抗雜訊，這裡改 mag=conf 也是可以的 (我們上一輪決定先不加)
                draw_gaussian_on_channel(canvas[target_ch], x, y, cfg.sigma_kp, mag=1.0)
                
            except Exception:
                pass
    ch += num_kp

    # --- 2. 繪製骨骼連線 (可選) ---
    if cfg.include_bone_lines:
        # ... (這部分第一階段先跳過，程式碼可以先留著或註解掉) ...
        # 如果需要加，記得把線畫在 canvas[ch + e_idx]
        pass
    ch += num_edges # 即使不畫，也要佔位，或者動態調整 C

    # --- 3. 繪製物件 (Object Channels - Gaussian Fill) ---
    if num_obj_ch > 0 and dets:
        cls2ch = {name: i for i, name in enumerate(cfg.object_classes)}
        for d in dets or []:
            name = d.get("cls_name") or d.get("class_name") or d.get("label") or d.get("name")
            if name not in cls2ch:
                continue
            
            bb = d.get("bbox") or d.get("xyxy") or {}
            bxyxy = _bbox_xyxy_from_any(bb)
            if bxyxy is None: continue
            
            x1, y1, x2, y2 = bxyxy
            # 計算中心點與概略半徑
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            
            # 縮放至 Map 座標
            map_cx = (cx / img_w) * Wc
            map_cy = (cy / img_h) * Hc
            
            # 繪製物件 (使用較大的 sigma_obj)
            # 這裡我們用 "高斯填滿" 的概念：在中心畫一個大高斯
            # 這樣比填滿矩形好，因為有中心強度的概念
            obj_idx = cls2ch[name]
            draw_gaussian_on_channel(canvas[ch + obj_idx], map_cx, map_cy, cfg.sigma_obj, mag=1.0)
            
    ch += num_obj_ch

    # --- 4. CoordConv ---
    xv = np.linspace(-1, 1, Wc)[None, :].repeat(Hc, 0)
    yv = np.linspace(-1, 1, Hc)[:, None].repeat(Wc, 1)
    canvas[ch] = xv; canvas[ch+1] = yv; ch += 2

    return canvas

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

def linear_interpolate_kps(window_parsed, max_missing=None):
    """
    對一個 window 的 keypoints 進行簡單線性插值 (Training Only)。
    參數:
      max_missing: 如果連續缺失幀數超過此值，就不補點 (視為長時間遮擋)。
    """
    if max_missing is None:
        max_missing = Config.max_missing_interpolate
    T = len(window_parsed)
    if T < 2: return window_parsed
    
    num_kp = 17
    # 建立 (T, 17, 3) 存 x, y, conf
    data = np.full((T, num_kp, 3), np.nan, dtype=np.float32)
    
    # 1. 填入原始資料
    for t, item in enumerate(window_parsed):
        kps = item[1]
        for k_idx in range(min(len(kps), num_kp)):
            k_data = kps[k_idx]
            if 'x' in k_data and 'y' in k_data:
                conf = float(k_data.get('conf', k_data.get('confidence', 1.0)))
                # 門檻設低一點，保留原始微弱訊號，讓插值有支點
                if conf > 0.1: 
                    data[t, k_idx, 0] = float(k_data['x'])
                    data[t, k_idx, 1] = float(k_data['y'])
                    data[t, k_idx, 2] = conf
    
    # 2. 執行插值 (加入 max_missing 判斷)
    all_indices = np.arange(T)
    
    for k in range(num_kp):
        seq = data[:, k, :]
        valid_mask = ~np.isnan(seq[:, 0]) # 哪些幀是有值的
        valid_indices = all_indices[valid_mask]
        
        # 至少要有兩點才能連線
        if len(valid_indices) < 2:
            continue
            
        # 先計算所有點的線性插值結果 (暫存)
        # np.interp 會把所有空洞都填滿
        interp_x = np.interp(all_indices, valid_indices, seq[valid_indices, 0])
        interp_y = np.interp(all_indices, valid_indices, seq[valid_indices, 1])
        interp_c = np.interp(all_indices, valid_indices, seq[valid_indices, 2])
        
        # 3. 過濾：只套用符合 max_missing 限制的區段
        # 我們建立一個 mask，標記哪些位置是「允許填補」的
        fill_mask = np.zeros(T, dtype=bool)
        fill_mask[valid_indices] = True # 原始存在的點當然要留著
        
        # 檢查每一段間隔
        for i in range(len(valid_indices) - 1):
            start_idx = valid_indices[i]
            end_idx = valid_indices[i+1]
            gap_size = end_idx - start_idx - 1 # 中間缺幾幀
            
            if 0 < gap_size <= max_missing:
                # 只有當缺口夠小，才把中間設為 True
                fill_mask[start_idx+1 : end_idx] = True
        
        # 4. 將合法的插值結果寫回 data
        # fill_mask 為 False 的地方保持 NaN (代表長時間缺失，不補)
        data[fill_mask, k, 0] = interp_x[fill_mask]
        data[fill_mask, k, 1] = interp_y[fill_mask]
        data[fill_mask, k, 2] = interp_c[fill_mask]

    # 5. 轉回 List 格式
    new_window = []
    for t in range(T):
        bbox, _, dets, iw, ih = window_parsed[t]
        new_kps = []
        for k in range(num_kp):
            x, y, c = data[t, k]
            # 如果是 NaN (原始就缺 且 超過 max_missing 沒補)，填 0
            if np.isnan(x) or np.isnan(y):
                 new_kps.append({'x': 0.0, 'y': 0.0, 'conf': 0.0})
            else:
                 new_kps.append({'x': x, 'y': y, 'conf': c})
        
        new_window.append((bbox, new_kps, dets, iw, ih))
        
    return new_window

# ---- Motion features（與前版相同，略；保留 compute_motion_feats_with_mask_from_parsed） ----
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
                if conf>=_KP_CONF_TH: cnt+=1
        ok=(cnt>=2) or (bbox is not None)
        valid.append(1.0 if ok else 0.0)
    valid=np.array(valid,np.float32)
    return feats, valid

# ===================== 讀檔（同前版，略） =====================
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

def load_pose_sequence(pose_path, video_root=None, video_exts=None):
    """
    讀取骨架序列，並嘗試從同名影片檔讀取正確的 W, H。
    如果找不到影片，會印出警告並回退到掃描骨架最大值。
    """
    if video_exts is None: video_exts = [".mp4"]
    
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
        fixed_w, fixed_h = 0.0, 0.0
        
        # 嘗試找影片
        found_video = False
        if video_root and os.path.isdir(video_root):
            base_name = os.path.splitext(os.path.basename(fp))[0]
            parent_dir = os.path.basename(os.path.dirname(fp))
            
            for ext in video_exts:
                # 嘗試 1: video_root/class/video.mp4
                vid_path = os.path.join(video_root, parent_dir, base_name + ext)
                if not os.path.exists(vid_path):
                    # 嘗試 2: video_root/video.mp4
                    vid_path = os.path.join(video_root, base_name + ext)
                
                if os.path.exists(vid_path):
                    try:
                        cap = cv2.VideoCapture(vid_path)
                        if cap.isOpened():
                            fixed_w = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            fixed_h = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            found_video = True
                            print(f"[OK] {os.path.basename(fp)} -> 影片: {fixed_w:.0f}x{fixed_h:.0f}")
                        cap.release()
                    except: pass
                    if found_video: break

        if not found_video and video_root:
             # 可以註解掉這行以免洗版，或者留著除錯
             print(f"[Fallback] 找不到影片: {os.path.basename(fp)}")

        # Fallback
        if fixed_w == 0 or fixed_h == 0:
            max_x, max_y = 0.0, 0.0
            for r in recs:
                persons = r.get("persons", []) if isinstance(r, dict) else []
                for p in persons:
                    for k in p.get("keypoints", []):
                        max_x = max(max_x, float(k.get("x", 0)))
                        max_y = max(max_y, float(k.get("y", 0)))
                boxes = (r.get('boxes') if isinstance(r,dict) else []) or []
                for b in boxes: 
                     if isinstance(b,(list,tuple)) and len(b)==4: 
                         max_x=max(max_x, b[2]); max_y=max(max_y, b[3])
            fixed_w = max(640.0, max_x + 10)
            fixed_h = max(480.0, max_y + 10)
        
        for idx, r in enumerate(recs):
            if isinstance(r, dict) and r.get("type") not in (None, "pose"): continue
            fid = _get_frame_id(r) if isinstance(r, dict) else None
            if fid is None: fid = idx
            frame_data = r if isinstance(r, dict) else {"raw": r}
            frame_data['_fixed_w'] = fixed_w
            frame_data['_fixed_h'] = fixed_h
            poses[fid] = frame_data
            
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

# ===================== 擴增（同前版，略；保留與少數類綁定） =====================
import numpy as _np
import cv2 as _cv2

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def _affine_matrix(iw, ih, scale=1.0, rot_deg=0.0, tx_rel=0.0, ty_rel=0.0):
    cx, cy = iw * 0.5, ih * 0.5
    M = _cv2.getRotationMatrix2D((cx, cy), rot_deg, scale)  # 2x3
    A = _np.vstack([M, [0, 0, 1]]).astype(_np.float32)
    A[0, 2] += tx_rel * iw
    A[1, 2] += ty_rel * ih
    return A[:2, :]

def _apply_affine_to_point(x, y, M):
    nx = M[0,0]*x + M[0,1]*y + M[0,2]
    ny = M[1,0]*x + M[1,1]*y + M[1,2]
    return float(nx), float(ny)

def _apply_affine_to_bbox_xyxy(bxyxy, M, iw, ih):
    if bxyxy is None:
        return None
    x1,y1,x2,y2 = bxyxy
    pts = [(x1,y1),(x1,y2),(x2,y1),(x2,y2)]
    tp = [_apply_affine_to_point(px,py,M) for (px,py) in pts]
    xs = [p[0] for p in tp]; ys = [p[1] for p in tp]
    nx1, nx2 = _clamp(min(xs), 0, iw-1), _clamp(max(xs), 0, iw-1)
    ny1, ny2 = _clamp(min(ys), 0, ih-1), _clamp(max(ys), 0, ih-1)
    if nx2 < nx1 or ny2 < ny1:
        return None
    return [nx1, ny1, nx2, ny2]

def _maybe_dropout_kps(kps, drop_prob):
    out = []
    for d in (kps or []):
        dd = dict(d)
        if random.random() < float(drop_prob):
            dd['conf'] = min(0.05, float(dd.get('conf', dd.get('confidence', 1.0))))
        out.append(dd)
    return out

def _jitter_kps_gauss(kps, noise_px, iw, ih):
    if not noise_px or noise_px <= 0:
        return kps
    out = []
    for d in (kps or []):
        x = float(d.get('x', 0.0)); y = float(d.get('y', 0.0))
        nx = _clamp(x + _np.random.normal(0, noise_px), 0, iw-1)
        ny = _clamp(y + _np.random.normal(0, noise_px), 0, ih-1)
        out.append({'x': nx, 'y': ny, 'conf': float(d.get('conf', d.get('confidence',1.0)))})
    return out

def _maybe_hflip_point(x, y, iw):
    return iw - 1 - x, y

def _maybe_hflip(parsed_win, do_flip):
    if not do_flip:
        return parsed_win
    out = []
    for (bbox, kps, dets, iw, ih) in parsed_win:
        bxyxy = _bbox_xyxy_from_any(bbox)
        if bxyxy is not None:
            x1,y1,x2,y2 = bxyxy
            fx1, _ = _maybe_hflip_point(x1, y1, iw)
            fx2, _ = _maybe_hflip_point(x2, y2, iw)
            bbox = [min(fx1,fx2), y1, max(fx1,fx2), y2]
        nkps = []
        for d in (kps or []):
            nx, ny = _maybe_hflip_point(float(d['x']), float(d['y']), iw)
            nkps.append({'x': nx, 'y': ny, 'conf': float(d.get('conf', d.get('confidence',1.0)))})
        ndets = []
        for od in (dets or []):
            bb = od.get('bbox') or od.get('xyxy')
            bxyxy = _bbox_xyxy_from_any(bb)
            if bxyxy is not None:
                x1,y1,x2,y2 = bxyxy
                fx1,_ = _maybe_hflip_point(x1, y1, iw)
                fx2,_ = _maybe_hflip_point(x2, y2, iw)
                nb = [min(fx1,fx2), y1, max(fx1,fx2), y2]
                od = dict(od); od['bbox'] = nb
            ndets.append(od)
        out.append((bbox, nkps, ndets, iw, ih))
    return out

def _augment_window_consistently(parsed_win, cfg: Config):
    if not parsed_win:
        return parsed_win
    do_flip = bool(getattr(cfg, "aug_hflip", False) and (random.random() < 0.5))
    win = _maybe_hflip(parsed_win, do_flip)
    iw0, ih0 = win[0][3], win[0][4]
    sc  = random.uniform(*getattr(cfg, "aug_affine_scale", (1.0,1.0)))
    rot = random.uniform(*getattr(cfg, "aug_affine_rotate_deg", (0.0,0.0)))
    tx  = random.uniform(*getattr(cfg, "aug_affine_translate", (0.0,0.0)))
    ty  = random.uniform(*getattr(cfg, "aug_affine_translate", (0.0,0.0)))
    M = _affine_matrix(iw0, ih0, scale=sc, rot_deg=rot, tx_rel=tx, ty_rel=ty)

    out = []
    for (bbox, kps, dets, iw, ih) in win:
        bxyxy = _bbox_xyxy_from_any(bbox)
        nb = _apply_affine_to_bbox_xyxy(bxyxy, M, iw, ih) if bxyxy is not None else None
        nkps = []
        for d in (kps or []):
            x, y = float(d.get('x', 0.0)), float(d.get('y', 0.0))
            ax, ay = _apply_affine_to_point(x, y, M)
            ax = _clamp(ax, 0, iw-1); ay = _clamp(ay, 0, ih-1)
            nkps.append({'x': ax, 'y': ay, 'conf': float(d.get('conf', d.get('confidence',1.0)))})
        nkps = _maybe_dropout_kps(nkps, getattr(cfg, "aug_kp_dropout_prob", 0.0))
        nkps = _jitter_kps_gauss(nkps, getattr(cfg, "aug_kp_noise_px", 0.0), iw, ih)
        ndets = []
        for od in (dets or []):
            bb = od.get('bbox') or od.get('xyxy')
            bxyxy = _bbox_xyxy_from_any(bb)
            if bxyxy is not None:
                nb2 = _apply_affine_to_bbox_xyxy(bxyxy, M, iw, ih)
                od = dict(od)
                if nb2 is not None:
                    od['bbox'] = nb2
            ndets.append(od)
        out.append((nb, nkps, ndets, iw, ih))
    return out

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
        self.half_len = int(cfg.half_len_override) if cfg.half_len_override else max(1,self.window//2)
        self.require_full_first = bool(cfg.require_full_first_frame)
        self.require_full_all = bool(cfg.require_full_skeleton_all)
        self.full_kp_min = int(cfg.full_kp_min)
        self.bbox_required = bool(cfg.bbox_required)
        if not os.path.isdir(cfg.pose_root):
            raise FileNotFoundError(cfg.pose_root)
        self.pose_root = cfg.pose_root
        self.obj_root = (cfg.obj_root if self.use_objects else None)

        # 掃類別
        pose_classes = {d for d in os.listdir(self.pose_root) if os.path.isdir(os.path.join(self.pose_root, d))}
        if self.use_objects:
            obj_classes = {d for d in os.listdir(self.obj_root) if os.path.isdir(os.path.join(self.obj_root, d))}
            class_names = sorted(list(pose_classes & obj_classes))
        else:
            class_names = sorted(list(pose_classes))

        # 白/黑名單
        wl = getattr(cfg, "class_whitelist", None)
        bl = getattr(cfg, "class_blacklist", None)
        if wl is not None and bl is not None:
            raise RuntimeError("class_whitelist 與 class_blacklist 不能同時設定；請擇一使用")
        if wl is not None:
            ws = {str(w).strip() for w in wl if w is not None and str(w).strip()}
            class_names = [c for c in class_names if c in ws]
        elif bl is not None:
            bs = {str(b).strip() for b in bl if b is not None and str(b).strip()}
            class_names = [c for c in class_names if c not in bs]
        if not class_names:
            raise RuntimeError("篩選後沒有可用的類別資料夾")

        self.class_names = class_names
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        # ====== 多少數 / 多多數 驗證與建立 ======
        mn_names = [s for s in (cfg.minority_class_names or []) if isinstance(s, str) and s.strip()]
        mn_names = [s.strip() for s in mn_names]
        for s in mn_names:
            if s not in self.class_names:
                raise ValueError(f"minority_class_names 包含未知類別：{s}，有效類別={self.class_names}")
        mn_set = set(mn_names)

        if cfg.majority_class_names is not None:
            mj_names = [s for s in (cfg.majority_class_names or []) if isinstance(s, str) and s.strip()]
            mj_names = [s.strip() for s in mj_names]
            for s in mj_names:
                if s not in self.class_names:
                    raise ValueError(f"majority_class_names 包含未知類別：{s}，有效類別={self.class_names}")
            if mn_set & set(mj_names):
                inter = list(mn_set & set(mj_names))
                raise ValueError(f"minority 與 majority 交集不為空：{inter}")
            mj_set = set(mj_names)
        else:
            mj_set = set([c for c in self.class_names if c not in mn_set])

        self.minority_idxs = {self.class_to_idx[c] for c in mn_set}
        self.majority_idxs = {self.class_to_idx[c] for c in mj_set}

        # 建 windows
        self.windows = []
        for cname in self.class_names:
            pose_cdir = os.path.join(self.pose_root, cname)
            obj_cdir  = os.path.join(self.obj_root, cname) if self.obj_root else None
            seq_paths = _find_sequences_for_class(pose_cdir, obj_cdir, self.use_objects)
            for pose_path, obj_path in seq_paths:
                poses = load_pose_sequence(
                    pose_path, 
                    video_root=getattr(self.cfg, "video_root", None),
                    video_exts=getattr(self.cfg, "video_exts", [".mp4"])
                )
                objs  = load_object_sequence(obj_path) if (self.use_objects and obj_path) else {}
                frames_pose = set(poses.keys())
                frames_obj  = set(objs.keys()) if self.use_objects else frames_pose
                frames = sorted(list(frames_pose & frames_obj))
                if len(frames) < self.window:
                    continue
                for i in range(0, len(frames) - self.window + 1, self.stride):
                    win_frames = frames[i:i + self.window]
                    if self.require_full_all:
                        ok_all = all(frame_has_full_skeleton(_extract_pose_basic(poses[f])[0],
                                                             _extract_pose_basic(poses[f])[1],
                                                             kp_need=self.full_kp_min,
                                                             require_bbox=self.bbox_required)
                                     for f in win_frames)
                        if not ok_all:
                            continue
                    elif self.require_full_first:
                        f0 = win_frames[0]
                        if not frame_has_full_skeleton(_extract_pose_basic(poses[f0])[0],
                                                       _extract_pose_basic(poses[f0])[1],
                                                       kp_need=self.full_kp_min,
                                                       require_bbox=self.bbox_required):
                            continue
                    self.windows.append((self.class_to_idx[cname], win_frames, poses, objs))

        if not self.windows:
            raise RuntimeError("No training windows built. Check data roots.")

        # ===== 多數類下採樣（不碰少數） =====
        r = float(getattr(self.cfg, "non_minority_downsample_ratio", 1.0))
        if r < 1.0 and len(self.majority_idxs) > 0:
            idxs_mj = [i for i, (y, _f, _p, _o) in enumerate(self.windows) if y in self.majority_idxs]
            idxs_mn = [i for i, (y, _f, _p, _o) in enumerate(self.windows) if y in self.minority_idxs]
            keep = []
            by_cls = {}
            for i in idxs_mj:
                y = self.windows[i][0]
                by_cls.setdefault(y, []).append(i)
            kept_total = 0
            for y, arr in by_cls.items():
                random.shuffle(arr)
                k = max(1, int(len(arr) * r))
                keep.extend(arr[:k]); kept_total += k
            selected = sorted(keep + idxs_mn)
            print(f"[Downsample] majority kept {kept_total}/{len(idxs_mj)} (ratio={r}), minority kept {len(idxs_mn)}")
            self.windows = [self.windows[i] for i in selected]

        # motion mean/std
        self.motion_stats = self._estimate_motion_stats(n_limit=2000)

        # 擴增開關（訓練時才打開）
        self.enable_aug = True

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
            parsed = linear_interpolate_kps(parsed, max_missing=3)
            
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
        parsed = linear_interpolate_kps(parsed, max_missing=3)

        # 只對少數類做擴增
        if self.enable_aug and getattr(self.cfg, "aug_for_minority", False) and (y_idx in self.minority_idxs):
            if random.random() < float(getattr(self.cfg, "aug_prob", 0.5)):
                parsed = _augment_window_consistently(parsed, self.cfg)

        # motion + mask（z-score）
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

# ============== 解析一幀 pose ==============
def _extract_pose_basic(p):
    bbox = None
    kps_list = []
    
    if isinstance(p, dict) and '_fixed_w' in p:
        img_w = float(p['_fixed_w'])
        img_h = float(p['_fixed_h'])
    else:
        img_w = 640.0
        img_h = 480.0

    if isinstance(p, dict) and p.get("persons"):
        person = max(p.get("persons", []), key=lambda x: x.get("score", 0.0))
        bbox = person.get("bbox")
        kps_list = person.get("keypoints") or []
        kps_list = [
            {"x": float(k.get("x", 0.0)), "y": float(k.get("y", 0.0)), "conf": float(k.get("conf", k.get("confidence", 1.0)))}
            for k in (kps_list[:17] if isinstance(kps_list, list) else [])
        ]
        if '_fixed_w' not in p:
            img_w = p.get("image_size", {}).get("width", img_w) or img_w
            img_h = p.get("image_size", {}).get("height", img_h) or img_h
    else:
        boxes = (p.get("boxes") if isinstance(p, dict) else None) or []
        kps_all = (p.get("keypoints") if isinstance(p, dict) else None) or []
        best_i, best_area = 0, -1
        for i, b in enumerate(boxes):
            if not (isinstance(b, (list, tuple)) and len(b) == 4): continue
            x1, y1, x2, y2 = b
            area = max(0, x2 - x1) * max(0, y2 - y1)
            if area > best_area: best_area = area; best_i = i
        bbox = boxes[best_i] if boxes else None
        kp_raw = kps_all[best_i] if (kps_all and best_i < len(kps_all)) else (kps_all[0] if kps_all else [])
        kps_list = ([{"x": float(x), "y": float(y), "conf": 1.0} for (x, y) in kp_raw[:17]] if kp_raw else [])

    return bbox, kps_list, float(img_w), float(img_h)

# ===================== Model =====================
class SpaceCNN(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GroupNorm(8,128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.GroupNorm(8, 256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.proj = nn.Linear(256, out_ch)
    def forward(self, x):
        z = self.net(x).flatten(1)
        return self.proj(z)

class TemporalHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, mode: str = "attn", dropout: float = 0.3):
        super().__init__()
        self.mode = mode
        self.drop = nn.Dropout(dropout)
        if mode == "attn":
            self.attn = nn.Linear(in_dim, 1)
        self.fc = nn.Linear(in_dim, num_classes)
    def forward(self, seq_feats: torch.Tensor, mask = None):
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

    # ===== 少數類索引過採樣（支援整體倍數或 per-class 倍數映射） =====
    dl_tr = dl_va = None
    use_map = any(v > 1 for v in (cfg.minority_oversample_mult_map or {}).values())
    need_os = (cfg.minority_oversample_mult > 1) or use_map

    if need_os:
        # 依類別收集索引
        pos_by_cls = {}
        neg = []
        for i in idx_tr:
            y = ds.windows[i][0]
            if y in ds.minority_idxs:
                pos_by_cls.setdefault(y, []).append(i)
            else:
                neg.append(i)

        new_idx_tr = list(neg)
        if use_map:
            # 依映射倍增
            for y_cls, pos_list in pos_by_cls.items():
                cname = ds.class_names[y_cls]
                mult = int(cfg.minority_oversample_mult_map.get(cname, 1))
                if mult > 1:
                    new_idx_tr.extend(pos_list * mult)
                else:
                    new_idx_tr.extend(pos_list)
            print(f"[Oversample-map] per-class applied; totals: neg={len(neg)}, "
                  f"sum_pos={sum(len(v) for v in pos_by_cls.values())} -> "
                  f"{len(new_idx_tr)-len(neg)} (after mult)")
        else:
            # 統一倍數
            m = int(cfg.minority_oversample_mult)
            for pos_list in pos_by_cls.values():
                new_idx_tr.extend(pos_list * m)
            print(f"[Oversample] minority x{m}: pos {sum(len(v) for v in pos_by_cls.values())} "
                  f"-> {sum(len(v) for v in pos_by_cls.values())*m}, others {len(neg)}")

        random.shuffle(new_idx_tr)
        idx_tr = new_idx_tr

        # 暫時關閉 WeightedRandomSampler，避免雙重平衡
        _use_sampler_backup = cfg.use_sampler
        cfg.use_sampler = False
        dl_tr, dl_va = build_loaders(cfg, ds, idx_tr, idx_va)
        cfg.use_sampler = _use_sampler_backup
    else:
        dl_tr, dl_va = build_loaders(cfg, ds, idx_tr, idx_va)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_ch = 17 + (len(COCO_EDGES) if cfg.include_bone_lines else 0) + (len(ds.rm_cfg.object_classes) if cfg.use_objects else 0) + 2
    model = CNNLSTM(in_ch=in_ch, num_classes=len(ds.class_names), cnn_out=_CNN_OUT,
                    lstm_h=cfg.lstm_hidden, lstm_layers=_LSTM_LAYERS, bidirectional=bool(cfg.bidirectional),
                    temporal_pool=cfg.temporal_pool, dropout=cfg.dropout, motion_dim=9).to(device)

    # Loss（alpha 以頻率反比；亦可改成 Class-Balanced 權重）
    counts = [0]*len(ds.class_names)
    for i in idx_tr:
        counts[ds.windows[i][0]] += 1
    total = float(sum(max(1,c) for c in counts))
    alpha = torch.tensor([total/max(1,c) for c in counts], dtype=torch.float32)
    alpha = (alpha/alpha.sum()).to(device)
    criterion = nn.CrossEntropyLoss() if cfg.loss=="ce" else FocalLoss(alpha=alpha, gamma=_FOCAL_GAMMA)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=_WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda', enabled=_AMP and device.type=="cuda")

    # ---- Top-3 ----
    TOPK = _SAVE_TOP_K
    topk_list = []
    _topk_index_path = os.path.join(cfg.out_dir, 'topk.json')
    best_score = -1.0
    best_path = None

    for ep in range(1, cfg.epochs+1):
        # ========= Train =========
        ds.enable_aug = True
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
            loss_sum += float(loss.item()) * len(y)
            corr += (out.argmax(1) == y).sum().item(); tot += len(y)
            pbar.set_postfix({"loss": f"{loss_sum/max(1,tot):.4f}", "acc": f"{corr/max(1,tot):.3f}"})
        tr_loss = loss_sum/max(1,tot); tr_acc = corr/max(1,tot)

        # ========= Valid =========
        ds.enable_aug = False
        acc, mf1 = evaluate(model, dl_va, device, use_motion=True)
        score = mf1
        print(f"[Ep {ep:03d}] tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f} | va_acc={acc:.3f} va_mf1={mf1:.3f}")

        # ========= Save last / best =========
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
            with open(os.path.join(cfg.out_dir,'classes.json'),'w',encoding='utf-8') as f:
                json.dump(ds.class_names,f,ensure_ascii=False)
            with open(os.path.join(cfg.out_dir,'motion_norm.json'),'w',encoding='utf-8') as f:
                json.dump(ds.motion_stats,f,ensure_ascii=False)
            print(f"  ↳ saved new best: {best_path}")

        # ========= Top-3（保留前三名） =========
        ep_ck_name = f"ep{ep:03d}_score{score:.4f}.pt"
        ep_path = os.path.join(cfg.out_dir, ep_ck_name)
        try:
            torch.save(ck, ep_path)
        except Exception:
            pass
        topk_list.append({'epoch': int(ep), 'score': float(score), 'path': ep_ck_name})
        topk_list = sorted(topk_list, key=lambda d: d['score'], reverse=True)
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
        print("\\nLoading best model for final report...")
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
        print("\\nConfusion Matrix:\\n", confusion_matrix(y_true, y_pred))
        print("\\nClassification Report:\\n", classification_report(y_true, y_pred, target_names=ds.class_names, digits=3))

if __name__ == "__main__":
    cfg = Config()
    # Example:
    # cfg.minority_class_names = ["fall", "lie-rare"]
    # cfg.majority_class_names = ["walk", "sitstill"]  # optional
    # cfg.minority_oversample_mult = 2
    # cfg.minority_oversample_mult_map = {"fall": 3}  # 若指定此 map，會以 map 為主（忽略統一倍數）
    # cfg.non_minority_downsample_ratio = 0.5
    train(cfg)
