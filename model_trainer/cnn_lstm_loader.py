# -*- coding: utf-8 -*-
"""
JSON → CNN+LSTM 推論（常數版，直接改檔頭即可）
- 用『影片級 JSON』(pose_json, object_json) 做滑動視窗推論，不需要原始影像。
- 結構與你的訓練程式一致，能直接載入 best.pt 或 epoch ckpt。

如何使用：
1) 在檔頭填好 MODEL_PATH / CLASSES_PATH 與兩個 JSON 路徑 (POSE_JSON / OBJECT_JSON)。
2) OBJECT_CLASSES 要與訓練時一致（例如只用 "tv" 就改成 ["tv"]）。
3) 直接執行：python json_infer.py
4) 若要輸出 CSV，設 OUT_CSV = "preds.csv"。
"""

import os, json
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import cv2

# ===================== 常數設定（請直接修改） =====================
MODEL_PATH      = "outputs/models/test/best.pt"       # best.pt（state_dict）或 epXXX_score*.pt（完整 ckpt）
CLASSES_PATH    = "outputs/models/test/classes.json"   # 若載入 best.pt 需提供類別清單

# 你的兩支影片級 JSON（list 形式；索引=幀號）
POSE_JSON       = "outputs/skeletons/YOLO/YOLO-pose/fall/fall_009.json"       # boxes + keypoints
OBJECT_JSON     = "outputs/skeletons/YOLO/YOLO-detect/fall/fall_009.json"     # objects/detections（可為 None）
OUT_CSV         = None                                  # 例如 "preds.csv"；不要輸出就設 None

# Relation Map 與物件通道（需與訓練一致）
H, W            = 64, 64
INCLUDE_BONE_LINES = True
OBJECT_CLASSES  = ["bed", "chair"]          # 只要 tv 就寫 ["tv"]；若不用物件通道設為 []

# LSTM / 時序（需與訓練一致）
LSTM_HIDDEN     = 256
BIDIRECTIONAL   = False
TEMPORAL_POOL   = "attn"          # "last" | "mean" | "attn"
DROPOUT         = 0.3
WINDOW          = 20              # 每段片段幀數
STRIDE          = 5               # 視窗滑動步幅

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
    H,W = cfg.H, cfg.W
    num_bbox_mask = 1
    num_dist = 1
    num_kp = 17
    num_edges = len(COCO_EDGES) if cfg.include_bone_lines else 0
    num_obj_ch = len(cfg.object_classes)
    num_coord = 2
    C = num_bbox_mask + num_dist + num_kp + num_edges + num_obj_ch + num_coord

    canvas = np.zeros((C,H,W), np.float32)
    ch = 0
    # 1) bbox mask
    if bbox:
        bxyxy = _bbox_xyxy_from_any(bbox)
        if bxyxy is not None:
            x1,y1,x2,y2 = bxyxy
            x1 = int(np.clip((x1/img_w)*W, 0, W-1)); y1 = int(np.clip((y1/img_h)*H, 0, H-1))
            x2 = int(np.clip((x2/img_w)*W, 0, W-1)); y2 = int(np.clip((y2/img_h)*H, 0, H-1))
            if x2>=x1 and y2>=y1: canvas[ch, y1:y2+1, x1:x2+1] = 1.0
    ch += 1
    # 2) distance transform
    inv = (1.0-canvas[0]).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    if dist.max()>0: dist /= dist.max()
    canvas[ch] = dist; ch+=1
    # 3) keypoints
    if kps:
        for i, kp in enumerate(kps[:num_kp]):
            try:
                conf = float(kp.get('conf', kp.get('confidence',1.0)))
                if conf < 0.4: continue
                x = (float(kp['x'])/img_w)*W; y = (float(kp['y'])/img_h)*H
                draw_gaussian(canvas[ch+i], x, y, 3.0, mag=conf)
            except Exception:
                pass
    ch += num_kp
    # 4) bone lines
    if cfg.include_bone_lines and kps and len(kps)>=num_kp:
        pts = []
        for i in range(num_kp):
            try:
                x = int(np.clip((float(kps[i]['x'])/img_w)*W,0,W-1))
                y = int(np.clip((float(kps[i]['y'])/img_h)*H,0,H-1))
                c = float(kps[i].get('conf', kps[i].get('confidence',1.0)))
                pts.append((x,y,c))
            except Exception:
                pts.append((None,None,0.0))
        for e_idx,(a,b) in enumerate(COCO_EDGES):
            x1,y1,c1 = pts[a]; x2,y2,c2 = pts[b]
            if x1 is None or x2 is None: continue
            if min(c1,c2) < 0.4: continue
            cv2.line(canvas[ch+e_idx], (x1,y1), (x2,y2), 1.0, 1)
    ch += num_edges
    # 5) objects
    if num_obj_ch>0 and objects:
        name_to_idx = {n:i for i,n in enumerate(cfg.object_classes)}
        for det in objects:
            cname = str(det.get('class_name', det.get('name', det.get('label','')))).strip()
            if cname not in name_to_idx: continue
            bxyxy = _bbox_xyxy_from_any(det.get('bbox') or det.get('xyxy') or det.get('box') or det.get('bbox_xyxy'))
            if bxyxy is None: continue
            x1,y1,x2,y2 = bxyxy
            x1 = int(np.clip((x1/img_w)*W, 0, W-1)); y1 = int(np.clip((y1/img_h)*H, 0, H-1))
            x2 = int(np.clip((x2/img_w)*W, 0, W-1)); y2 = int(np.clip((y2/img_h)*H, 0, H-1))
            if x2>=x1 and y2>=y1:
                canvas[ch + name_to_idx[cname], y1:y2+1, x1:x2+1] = 1.0
    ch += num_obj_ch
    # 6) CoordConv
    xv = np.linspace(-1,1,W)[None,:].repeat(H,0)
    yv = np.linspace(-1,1,H)[:,None].repeat(W,1)
    canvas[ch] = xv; canvas[ch+1] = yv; ch+=2
    assert ch==C, f"channel mismatch {ch} vs {C}"
    return canvas

# ===================== JSON 解析 =====================

def read_any_json(path: Optional[str]):
    if not path: return []
    with open(path, 'r', encoding='utf-8') as f:
        txt = f.read().strip()
    if not txt: return []
    # 先試 json
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
        or rec.get('detections') if isinstance(rec, dict) else None \
        or rec.get('boxes') if isinstance(rec, dict) else None \
        or rec.get('bboxes') if isinstance(rec, dict) else None \
        or rec.get('predictions') if isinstance(rec, dict) else None
    if objs is None:
        return []
    out = []
    for o in objs:
        if not isinstance(o, dict):
            continue
        name = o.get('class_name', o.get('name', o.get('label','')))
        bbox = o.get('bbox') or o.get('xyxy') or o.get('box') or o.get('bbox_xyxy')
        if name is None or bbox is None:
            continue
        out.append({'class_name': str(name), 'bbox': bbox})
    return out

# ===================== 模型（與訓練一致） =====================
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
    def __init__(self, in_dim, num_classes, mode="mean", dropout=0.0):
        super().__init__()
        self.mode = mode
        self.drop = nn.Dropout(dropout) if dropout>0 else nn.Identity()
        if mode == "attn":
            self.attn = nn.Linear(in_dim,1)
        self.fc = nn.Linear(in_dim, num_classes)
    def forward(self, seq_feats):
        if self.mode=="mean":
            g = seq_feats.mean(dim=1)
        elif self.mode=="attn":
            a = self.attn(seq_feats).squeeze(-1)
            w = torch.softmax(a, dim=1).unsqueeze(-1)
            g = (seq_feats*w).sum(dim=1)
        else:
            g = seq_feats[:,-1]
        g = self.drop(g)
        return self.fc(g)

class CNNLSTM(nn.Module):
    def __init__(self, in_ch, num_classes, cnn_out=256, lstm_h=256, lstm_layers=2,
                 bidirectional=True, temporal_pool="mean", dropout=0.3):
        super().__init__()
        self.cnn = SpaceCNN(in_ch, cnn_out)
        self.lstm = nn.LSTM(cnn_out, lstm_h, lstm_layers, batch_first=True, bidirectional=bidirectional)
        feat_dim = lstm_h * (2 if bidirectional else 1)
        self.head = TemporalHead(feat_dim, num_classes, mode=temporal_pool, dropout=dropout)
    def forward(self, x):
        B,T,C,H,W = x.shape
        z = self.cnn(x.view(B*T, C, H, W)).view(B,T,-1)
        out,_ = self.lstm(z)
        return self.head(out)

# ===================== 推論主流程 =====================

def run_on_json(pose_json: str, object_json: Optional[str]=None) -> list:
    poses = read_any_json(pose_json)
    objs  = read_any_json(object_json) if object_json else None
    
#===================== 檢查 =====================
    T = len(poses)
    expected = (T - WINDOW) // STRIDE + 1 if T >= WINDOW else 0
    print(f"[Debug] T={T}, window={WINDOW}, stride={STRIDE}, expected={expected}")
#===================== 檢查 =====================

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 計算輸入通道數（與訓練一致）
    in_ch = 1 + 1 + 17 + (len(COCO_EDGES) if INCLUDE_BONE_LINES else 0) + len(OBJECT_CLASSES) + 2

    # 讀類別
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

    # 建模
    model = CNNLSTM(in_ch=in_ch, num_classes=len(class_names), cnn_out=256,
                    lstm_h=LSTM_HIDDEN, lstm_layers=2, bidirectional=BIDIRECTIONAL,
                    temporal_pool=TEMPORAL_POOL, dropout=DROPOUT).to(device)
    model.load_state_dict(state)
    model.eval()

    # 物件取得函式
    def get_obj(i):
        if objs is None: return []
        if i < len(objs):
            return extract_objects_from_record(objs[i])
        return []

    # slide
    T = len(poses)
    results = []
    with torch.no_grad():
        for start in range(0, max(0, T - WINDOW + 1), STRIDE):
            clips = []
            for i in range(start, start + WINDOW):
                bbox, kps, img_w, img_h = extract_pose_from_record(poses[i])
                dets = get_obj(i)
                rel = rasterize_frame(bbox, kps, dets, img_w, img_h,
                                      RelationMapConfig(H=H,W=W,include_bone_lines=INCLUDE_BONE_LINES,object_classes=OBJECT_CLASSES))
                clips.append(rel)
            x = torch.from_numpy(np.stack(clips)).unsqueeze(0).float().to(device)  # (1,T,C,H,W)
            logits = model(x)
            prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
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
        print(f"frames {r['start_frame']:>5}-{r['end_frame']:<5} | pred={r['pred']} | probs={np.round(r['probs'],3)}")
    if OUT_CSV:
        save_results_csv(results, OUT_CSV, classes)
        print(f"Saved CSV: {OUT_CSV}")
