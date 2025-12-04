# -*- coding: utf-8 -*-
"""
功能清單：
- ✅ tqdm 進度條：訓練 / 驗證過程顯示進度與每 Epoch 摘要列印
- ✅ Checkpoint 機制：
    • 每回合保存 `last.pt`
    • 自動更新最佳模型 `best.pt`
    • 維護 Top-3 → `best_epXXX.pt` + `topk.json`
- ✅ Kalman Filter 半視窗處理（不改 raw，只在 Dataset 前處理）
- ✅ 輸入特徵：
    • motion (9 維) + 有效幀 mask
    • TemporalHead 支援 mask（mean/attn 忽略無效幀）
    • motion 正規化（dataset 估計 mean/std，存入 ckpt）
    • RelationMap 通道：bbox(1) + dist(1) + kp(17) + bone(12, 可選) + object(N) + coord(2)
- ✅ 二元 / 多類分類皆可：
    • 多類模式：依資料夾遞迴自動建立分類
    • 二元模式：可於 Config 設定 `binary_mode=True`
        - 指定 `rare_class_name` 為稀有類別（標籤=1）
        - 其餘所有類別自動併為非稀有（標籤=0）
        - `binary_class_names=(非稀有, 稀有)` 控制輸出名稱
    • sampler、loss、checkpoint 均自動適應類別數
- ✅ 參數型別設計沿用基準（例如 `use_objects: bool`）

使用方式：
- 無 object → `use_objects=False`
- 多類分類 → `binary_mode=False`
- 二元分類 → `binary_mode=True`，並設定 `rare_class_name` 與 `binary_class_names`
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
from collections import Counter

# ===================== Config（沿用基準類別＋布林 use_objects） =====================
class Config:
    # 路徑（分開設定）
    pose_root = "outputs/skeletons/binary/YOLO-pose"
    obj_root  = "outputs/skeletons/binary/YOLO-detect"
    out_dir   = "outputs/models/HM/focal/binary"
    video_root = "medias/train_video/binary"   # 請改成你存放影片的根目錄
    video_exts = [".mp4", ".avi", ".mov", ".mkv"] # 支援多種格式 (優先順序)
    
    # 是否使用物件框
    use_objects = True            
    object_classes = ["bed", "chair", "bench"]  # 空則不建立物件通道

    # 時序
    window = 10
    stride = 5
    
    # —— 針對稀有/非稀有類別的差別步長（只影響建 window）——
    stride_pos = 1   # fall 類用更密的 stride（例：2 幀）
    stride_neg = 5   # non_fall 類維持原本（例：5 幀）

    # （可選）限制每支 non_fall 影片最多切多少個 window，0=不限制
    max_neg_windows_per_seq = 0

    # Relation Map
    H, W = 64, 64
    include_bone_lines = False

    # === Heatmap 參數 ===
    sigma_kp = 2.0   # 骨架點的高斯 sigma
    sigma_obj = 4.0  # 物件中心填滿的高斯 sigma (較大)

    # 模型
    lstm_hidden  = 256
    bidirectional = False
    temporal_pool = "attn"       # "last" | "mean" | "attn"
    dropout = 0.3

    # 訓練
    epochs     = 50
    batch_size = 8
    lr         = 1e-5
    use_sampler = True
    loss = "focal"               # "focal" | "ce"
    
     # === 二元分類開關與命名 ===
    binary_mode = True           # 開啟後即做二元分類（稀有 vs 非稀有）
    rare_class_name = "fall"     # 指定資料夾名中誰是「稀有」類
    binary_class_names = ("non_fall", "fall")  # (非稀有, 稀有) 兩類名稱

    # 輸入前處理 / 過濾規則（沿用基準）
    require_full_first_frame = True
    half_len_override = None
    require_full_skeleton_all = False
    full_kp_min = 8
    bbox_required = False

    # 插值設定
    max_missing_interpolate = 3  # 訓練時最多連續補幾幀

    # ====== On-the-fly 擴增（只給稀有類：fall；不改模型大小/輸出）======
    aug_for_rare = False           # 開啟針對稀有類（fall）的擴增
    aug_prob = 0.50               # 每個 window 觸發擴增的機率
    aug_hflip = True              # 允許左右翻轉（不會違反重力/因果）
    # 仿射變換（整段 window 使用同一組參數；避免跨幀不一致）
    aug_affine_scale = (0.90, 1.10)        # 縮放
    aug_affine_rotate_deg = (-12.0, 12.0)  # 小角度旋轉；禁止 180°（上下顛倒）
    aug_affine_translate = (-0.05, 0.05)   # 位移（相對於寬/高的比例）
    # 關節點雜訊/遮擋
    aug_kp_noise_px = 2.0          # 每幀對 keypoints 加入高斯噪聲（像素）
    aug_kp_dropout_prob = 0.05     # 少量 keypoints 置為低信心，模擬遮擋

    # ---- Non-fall 下採樣（只影響資料量，不改模型結構）----
    non_fall_downsample_ratio = 1.0   # 設 1.0 表示不下採樣；0.5 表示保留一半

    # ---- Fall 過採樣（訓練索引層面重複正類樣本；不改模型結構）----
    pos_oversample_mult = 1           # 1=不過採樣；2=將 fall 索引重複 2 倍

# ===================== 穩定預設 =====================
_SEED = 42
_SAMPLE_EVERY = 1
_KP_CONF_TH = 0.4
_SIGMA_KP = 3.0
_CNN_OUT = 256
_LSTM_LAYERS = 2
_WEIGHT_DECAY = 1e-4
_GRAD_CLIP = 1.0
_AMP = False
_EARLY_STOP_PATIENCE = 5
_SAVE_TOP_K = 3
_FOCAL_GAMMA = 2.0
_MOTION_CLIP_V = 5.0   # 10 FPS 建議放寬一點 (原本 3.0)
_MOTION_CLIP_A = 10.0  # (原本 9.0)

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
    def __init__(self, H=64, W=64, sigma_kp=2.0, sigma_obj=4.0, kp_conf_th=0.4,
                 include_bone_lines=False, object_classes=None):
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

def load_pose_sequence(pose_path, video_root=None, video_exts=None):
    """
    讀取骨架序列，並嘗試從同名影片檔讀取正確的 W, H。
    如果找不到影片，會印出警告並回退到掃描骨架最大值。
    """
    if video_exts is None: video_exts = [".mp4"] # 預設
    
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
        
        # === [核心修改] 嘗試讀取對應的影片解析度 ===
        fixed_w, fixed_h = 0.0, 0.0
        
        # 嘗試尋找影片的邏輯
        found_video = False
        if video_root and os.path.isdir(video_root):
            # 取得檔名 (不含副檔名)
            base_name = os.path.splitext(os.path.basename(fp))[0]
            # 取得父資料夾名稱 (例如 "fall")
            parent_dir = os.path.basename(os.path.dirname(fp))
            
            # 嘗試所有可能的副檔名
            for ext in video_exts:
                # 嘗試 1: video_root/class/video.mp4
                vid_path = os.path.join(video_root, parent_dir, base_name + ext)
                if not os.path.exists(vid_path):
                    # 嘗試 2: video_root/video.mp4 (不分資料夾)
                    vid_path = os.path.join(video_root, base_name + ext)
                
                if os.path.exists(vid_path):
                    try:
                        cap = cv2.VideoCapture(vid_path)
                        if cap.isOpened():
                            fixed_w = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            fixed_h = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            found_video = True
                            # print(f"[Info] Found video: {vid_path} -> {fixed_w}x{fixed_h}")
                        cap.release()
                    except Exception as e:
                        print(f"[Error] Failed to read video metadata: {vid_path}, err={e}")
                    
                    if found_video: break # 找到了就跳出迴圈

        # 如果找不到影片，印出警告 (方便除錯)
        if not found_video and video_root:
             print(f"[Warn] Video not found for JSON: {os.path.basename(fp)} (checked in {video_root})")

        # === [備案] 如果沒影片或讀失敗，回退到「掃描骨架最大值」 ===
        if fixed_w == 0 or fixed_h == 0:
            max_x, max_y = 0.0, 0.0
            for r in recs:
                # 掃描 persons
                persons = r.get("persons", []) if isinstance(r, dict) else []
                for p in persons:
                    for k in p.get("keypoints", []):
                        max_x = max(max_x, float(k.get("x", 0)))
                        max_y = max(max_y, float(k.get("y", 0)))
                # 掃描 boxes
                boxes = (r.get('boxes') if isinstance(r,dict) else []) or []
                for b in boxes: 
                     if isinstance(b,(list,tuple)) and len(b)==4: 
                         max_x=max(max_x, b[2]); max_y=max(max_y, b[3])
            
            # 給一點緩衝，避免座標貼邊
            fixed_w = max(640.0, max_x + 10)
            fixed_h = max(480.0, max_y + 10)
            
            # 如果是回退模式，也可以印個 log 讓你知道
            # print(f"[Info] Fallback resolution for {os.path.basename(fp)}: {fixed_w}x{fixed_h}")
        
        # 3. 寫入每一幀
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

# ============== 解析一幀 pose ==============

def _extract_pose_basic(p):
    bbox = None
    kps_list = []
    
    # 優先讀取 load_pose_sequence 算好的固定解析度
    if isinstance(p, dict) and '_fixed_w' in p:
        img_w = float(p['_fixed_w'])
        img_h = float(p['_fixed_h'])
    else:
        # 萬一沒有，才用舊邏輯猜 (通常不會發生)
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
        # 如果 JSON 本身有 image_size 且沒有 _fixed_w，也可以用
        if '_fixed_w' not in p:
            img_w = p.get("image_size", {}).get("width", img_w) or img_w
            img_h = p.get("image_size", {}).get("height", img_h) or img_h

    else:
        # Fallback for old format
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
        
        # 如果是舊邏輯且沒有 _fixed_w，這裡原本是用 max(xs) 猜
        # 但現在我們盡量依賴 _fixed_w

    return bbox, kps_list, float(img_w), float(img_h)

# ===================== Dataset =====================
# ===================== Augmentation helpers (cause-safe, on-the-fly for FALL) =====================
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
        out.append({'x': nx, 'y': ny, 'conf': float(d.get('conf', d.get('confidence', 1.0)))})
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

def _augment_window_consistently(parsed_win, cfg):
    if not parsed_win:
        return parsed_win
    do_flip = bool(getattr(cfg, "aug_hflip", False) and (random.random() < 0.5))
    win = _maybe_hflip(parsed_win, do_flip)
    iw0, ih0 = win[0][3], win[0][4]
    sc = random.uniform(*getattr(cfg, "aug_affine_scale", (1.0,1.0)))
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
        pose_classes = {d for d in os.listdir(self.pose_root) if os.path.isdir(os.path.join(self.pose_root, d))}
        if self.use_objects:
            obj_classes = {d for d in os.listdir(self.obj_root) if os.path.isdir(os.path.join(self.obj_root, d))}
            orig_class_names = sorted(list(pose_classes & obj_classes))
        else:
            orig_class_names = sorted(list(pose_classes))

        # --- 二元分類設定：rare vs non-rare ---
        self.binary_mode = bool(getattr(cfg, "binary_mode", False))
        self.orig_class_names = orig_class_names  # 保留原始類別（之後誤差分析會用到）

        if self.binary_mode:
            rare = getattr(cfg, "rare_class_name", None)
            if (rare is None) or (rare not in orig_class_names):
                raise ValueError(f"rare_class_name='{rare}' 不在資料夾列表：{orig_class_names}")

            # 對外顯示名稱（寫入 ckpt / classes.json）
            self.class_names = list(getattr(cfg, "binary_class_names", ("negative", "positive")))
            assert len(self.class_names) == 2, "binary_class_names 必須是 2 個名稱（非稀有, 稀有）"

            # 建立 0/1 映射：rare -> 1，其餘 -> 0
            self._to_binary = {c: (1 if c == rare else 0) for c in orig_class_names}
        else:
            # 維持多類
            self.class_names = orig_class_names
            self._to_binary = None

        # 仍保留原本的索引（多類情境下會用到）
        self.class_to_idx = {c: i for i, c in enumerate(self.orig_class_names)}
        
        # 建立 windows
        self.windows = []

        self.enable_aug = True
        
        # 重要：掃「實際存在的資料夾名」
        scan_classes = self.orig_class_names

        for cname in scan_classes:
            pose_cdir = os.path.join(self.pose_root, cname)
            obj_cdir  = os.path.join(self.obj_root, cname) if self.obj_root else None
            seq_paths = _find_sequences_for_class(pose_cdir, obj_cdir, self.use_objects)

            for pose_path, obj_path in seq_paths:
                # 讀單一序列（以 frame_id 為 key 的 dict）
                poses = load_pose_sequence(pose_path, 
                           video_root=getattr(self.cfg, "video_root", None),
                           video_exts=getattr(self.cfg, "video_exts", [".mp4"]))
                objs  = load_object_sequence(obj_path) if (self.use_objects and obj_path) else {}

                # 兩邊取交集的 frame 清單
                frames_pose = set(poses.keys())
                frames_obj  = set(objs.keys()) if self.use_objects else frames_pose
                frames = sorted(list(frames_pose & frames_obj))
                if len(frames) < self.window:
                    continue

                # 以 window/stride 切片
                
                # —— 依類別決定 stride（fall 用 stride_pos，non_fall 用 stride_neg）——
                is_binary = bool(getattr(self, "binary_mode", False))
                if is_binary:
                    # 正類（稀有類）名稱：cfg.rare_class_name（預設 'fall'）
                    is_pos = (cname == getattr(self.cfg, "rare_class_name", "fall"))
                else:
                    is_pos = False  # 多類情境這段不啟用；此檔主要針對 binary

                step_this = int(getattr(self.cfg, "stride_pos", self.stride) if is_pos
                                else getattr(self.cfg, "stride_neg", self.stride))
                step_this = max(1, step_this)

                # ——（可選）限制每支 non_fall 序列最多切出的 window 數；0=不限制 —— 
                neg_added_this_seq = 0
                max_neg_windows = int(getattr(self.cfg, "max_neg_windows_per_seq", 0))
                for i in range(0, len(frames) - self.window + 1, step_this):
                    win_frames = frames[i:i + self.window]

                    # 是否要求完整骨架（全幀/首幀）
                    if self.require_full_all:
                        if not all(
                            frame_has_full_skeleton(
                                _extract_pose_basic(poses[f])[0],  # bbox
                                _extract_pose_basic(poses[f])[1],  # kps
                                kp_need=self.full_kp_min,
                                require_bbox=self.bbox_required
                            )
                            for f in win_frames
                        ):
                            continue
                    elif self.require_full_first:
                        f0 = win_frames[0]
                        if not frame_has_full_skeleton(
                            _extract_pose_basic(poses[f0])[0],
                            _extract_pose_basic(poses[f0])[1],
                            kp_need=self.full_kp_min,
                            require_bbox=self.bbox_required
                        ):
                            continue

                    # 解析本 window 每一幀 → (bbox, kps, dets, iw, ih)
                    parsed = []
                    for fid in win_frames:
                        p = poses[fid]
                        o = objs.get(fid) if self.use_objects else None
                        bbox, kps, iw, ih = _extract_pose_basic(p)
                        # 物件偵測欄位名稱可能不同，逐一嘗試
                        dets = []
                        if o:
                            dets = (
                                o.get("detections")
                                or o.get("objects")
                                or o.get("boxes")
                                or o.get("bboxes")
                                or o.get("predictions")
                                or []
                            )
                        parsed.append((bbox, kps, dets, iw, ih))


                    parsed = linear_interpolate_kps(parsed, max_missing=3)

                    # 指定 y（0/1 或多類）
                    if self.binary_mode:
                        y_idx = self._to_binary[cname]   # 0 = 非稀有, 1 = 稀有
                    else:
                        y_idx = self.class_to_idx[cname] # 多類情境

                    # 存 window（__getitem__ 會再用 poses/objs 重建張量）
                    self.windows.append((y_idx, win_frames, poses, objs))
                    
                    # —— 若是 non_fall（y=0），且設定了每序列上限，則累計並可能提早結束 —— 
                    if is_binary and (int(y_idx) == 0) and (max_neg_windows > 0):
                        neg_added_this_seq += 1
                        if neg_added_this_seq >= max_neg_windows:
                            break



        if not self.windows:
            raise RuntimeError("No training windows built. Check data roots.")

        # ---- 針對 non_fall(=0) 做下採樣：不改模型大小，只減少負樣本數量 ----
        if self.binary_mode:
            r = float(getattr(self.cfg, "non_fall_downsample_ratio", 1.0))
            if r < 1.0:
                idxs0 = [i for i, (y, _f, _p, _o) in enumerate(self.windows) if y == 0]  # non_fall
                idxs1 = [i for i, (y, _f, _p, _o) in enumerate(self.windows) if y == 1]  # fall
                keep0 = max(1, int(len(idxs0) * r))
                random.shuffle(idxs0)
                selected = sorted(idxs1 + idxs0[:keep0])
                self.windows = [self.windows[i] for i in selected]
                print(f"[Downsample] non_fall kept {keep0}/{len(idxs0)} (ratio={r}), fall kept {len(idxs1)}")

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
                bbox, kps, iw, ih = _extract_pose_basic(p)
                dets = (o.get("detections") or o.get("objects") or o.get("boxes") or o.get("bboxes") or o.get("predictions") or []) if o else []
                parsed.append((bbox, kps, dets, iw, ih))
            parsed = linear_interpolate_kps(parsed, max_missing=3)
            if getattr(self.cfg, "aug_for_rare", False) and bool(getattr(self, "binary_mode", False)):
                try:
                    is_rare = (int(y) == 1)
                except Exception:
                    is_rare = False
                if is_rare and (random.random() < float(getattr(self.cfg, "aug_prob", 0.5))):
                    parsed = _augment_window_consistently(parsed, self.cfg)
            M, _ = compute_motion_feats_with_mask_from_parsed(parsed)
            if M.shape[0] > 0:
                ms.append(M)
        if not ms:
            return {"mean": [0.0]*9, "std": [1.0]*9}
        M = np.concatenate(ms, axis=0)
        mean = M.mean(axis=0); std = M.std(axis=0) + 1e-6
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

        # 在 __getitem__ 用這個旗標
        if self.enable_aug and getattr(self.cfg, "aug_for_rare", False) and self.binary_mode:
            is_rare = (int(y_idx) == 1)
            if is_rare and (random.random() < float(getattr(self.cfg, "aug_prob", 0.5))):
                parsed = _augment_window_consistently(parsed, self.cfg)

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
    def __init__(self, in_ch: int, out_ch: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GroupNorm(8, 128), nn.ReLU(inplace=True),
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
    
    # ---- Fall 過採樣（索引層面；避免與 WeightedRandomSampler 疊加）----
    # 僅在二元分類（fall=1）時有意義；多類可以保留為 1
    if getattr(cfg, "pos_oversample_mult", 1) and cfg.pos_oversample_mult > 1:
        pos = [i for i in idx_tr if ds.windows[i][0] == 1]  # fall=1（稀有類）
        neg = [i for i in idx_tr if ds.windows[i][0] == 0]  # non_fall=0
        m = int(cfg.pos_oversample_mult)
        idx_tr = neg + pos * m
        random.shuffle(idx_tr)
        print(f"[Oversample] fall x{m}: pos {len(pos)} -> {len(pos)*m}, neg {len(neg)}")

        # 避免「索引過採樣」再搭配 WeightedRandomSampler 造成雙重平衡
        _use_sampler_backup = cfg.use_sampler
        cfg.use_sampler = False
        dl_tr, dl_va = build_loaders(cfg, ds, idx_tr, idx_va)
        cfg.use_sampler = _use_sampler_backup
    else:
        dl_tr, dl_va = build_loaders(cfg, ds, idx_tr, idx_va)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_kp = 17
    num_edges = len(COCO_EDGES) if cfg.include_bone_lines else 0
    num_obj = len(ds.rm_cfg.object_classes) if cfg.use_objects else 0
    num_coord = 2
    # in_ch 對齊 rasterizer
    in_ch = num_kp + num_edges + num_obj + num_coord

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
    scaler = torch.amp.GradScaler('cuda', enabled=_AMP and device.type=="cuda")

    # ---- Top-3 索引檔 ----
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
            # stats
            loss_sum += float(loss.item()) * len(y)
            corr += (out.argmax(1) == y).sum().item(); tot += len(y)
            pbar.set_postfix({"loss": f"{loss_sum/max(1,tot):.4f}", "acc": f"{corr/max(1,tot):.3f}"})
        tr_loss = loss_sum/max(1,tot); tr_acc = corr/max(1,tot)

        ds.enable_aug = False
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
