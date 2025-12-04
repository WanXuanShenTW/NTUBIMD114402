# app/utils/stream_infer_manager.py
# -*- coding: utf-8 -*-
"""
Stream Inference Manager (v8 Warning Feedback)
- ✅ [NEW] TargetTracker: 當所有目標被過濾時，回傳 warning 原因 ("screen_blocked" 或 "poor_quality")。
- ✅ [NEW] ingest: 接收 warning 並即時回傳給前端 (type="warning")。
- ✅ [FIX] TargetTracker: update 簽章變更，增加 warning 回傳值。
- ✅ 功能保留：雙軌制、解析度鎖定、數值正規化、Location、補點。
"""

import os
import asyncio
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict, deque
from datetime import datetime
import time
import json

import numpy as np
import torch

# === 導入兩個 loader ===
from . import binary_cnn_lstm_loader as bl
from . import multi_cnn_lstm_loader as ml

# WS 管理
from .ws_connection_manager import ws_manager


# ===================== 統一設定 =====================
# Relation Map
H = int(os.getenv("SC_REL_H", "64"))
W = int(os.getenv("SC_REL_W", "64"))
INCLUDE_BONE_LINES = bool(int(os.getenv("SC_INCLUDE_BONE", "1")))
OBJECT_CLASSES = [
    s.strip()
    for s in os.getenv("SC_OBJECT_CLASSES", "bed,chair,bench").split(",")
    if s.strip()
]

# Keypoint
KP_CONF_TH = float(os.getenv("SC_KP_CONF_TH", "0.4"))
SIGMA_KP = float(os.getenv("SC_SIGMA_KP", "2.0"))

# Motion
USE_MOTION = bool(int(os.getenv("SC_USE_MOTION", "1")))
MOTION_DIM = int(os.getenv("SC_MOTION_DIM", "9"))

# LSTM
LSTM_HIDDEN = int(os.getenv("SC_LSTM_H", "256"))
BIDIRECTIONAL = bool(int(os.getenv("SC_BIDIR", "0")))
TEMPORAL_POOL = os.getenv("SC_TEMPORAL_POOL", "attn")
DROPOUT = float(os.getenv("SC_DROPOUT", "0.3"))

# Windows
BIN_WINDOW = int(os.getenv("SC_BIN_WINDOW", "10"))
BIN_STRIDE = int(os.getenv("SC_BIN_STRIDE", "5"))
MULTI_WINDOW = int(os.getenv("SC_MULTI_WINDOW", "10"))
MULTI_STRIDE = int(os.getenv("SC_MULTI_STRIDE", "5"))

# Paths
BIN_MODEL_PATH = os.getenv("SC_BIN_MODEL_PATH", "models/binary/best.pt")
BIN_CLASSES_PATH = os.getenv("SC_BIN_CLASSES_PATH", "models/binary/classes.json")
MULTI_MODEL_PATH = os.getenv("SC_MULTI_MODEL_PATH", "models/multi/best.pt")
MULTI_CLASSES_PATH = os.getenv("SC_MULTI_CLASSES_PATH", "models/multi/classes.json")

# Thresholds
FALL_START_THR = float(os.getenv("SC_FALL_START_THR", "0.70"))
FALL_RECOVER_THR = float(os.getenv("SC_FALL_RECOVER_THR", "0.50"))
FALL_START_HITS = int(os.getenv("SC_FALL_START_HITS", "2"))
FALL_RECOVER_HITS = int(os.getenv("SC_FALL_RECOVER_HITS", "3"))

ACTION_START_THR = float(os.getenv("SC_ACTION_START_THR", "0.50"))
ACTION_START_HITS = int(os.getenv("SC_ACTION_START_HITS", "3"))
ENABLED_ACTIONS = [
    s.strip()
    for s in os.getenv("SC_ENABLED_ACTIONS", "walk,sitstill,liestill").split(",")
    if s.strip()
]

# Filter Settings
MAX_SCREEN_RATIO = float(os.getenv("SC_MAX_SCREEN_RATIO", "0.6"))
MIN_VALID_KPS = int(os.getenv("SC_MIN_VALID_KPS", "5"))

# Feature: Double FPS Interpolation
INTERP_DOUBLE = bool(int(os.getenv("SC_INTERP_DOUBLE", "0")))
# Feature: Metrics
METRICS_ON = bool(int(os.getenv("SC_METRICS", "0")))
METRICS_INTERVAL_MS = int(os.getenv("SC_METRICS_INTERVAL_MS", "5000"))


# ===================== 工具 =====================
def _now_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

def _torch_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _calc_stage_in_channels(stage: str, include_bone: bool, num_obj: int, num_edges: int) -> int:
    edges = num_edges if include_bone else 0
    if stage == "binary":
        return 17 + edges + num_obj + 2
    else:
        return 3 + edges + num_obj + 2

def _safe_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
    logits = logits - logits.max(dim=dim, keepdim=True).values
    return torch.softmax(logits, dim=dim)

# ===================== 解析單幀 (含解析度鎖定) =====================
def _extract_basic_frame(
    rec: Dict[str, Any],
    fixed_w: Optional[float] = None,
    fixed_h: Optional[float] = None
) -> Tuple[Optional[list], List[dict], float, float]:
    img_w, img_h = 640.0, 480.0
    if fixed_w is not None and fixed_h is not None and fixed_w > 0 and fixed_h > 0:
        img_w, img_h = fixed_w, fixed_h
    else:
        if "img_w" in rec and "img_h" in rec:
            img_w = float(rec.get("img_w") or 640.0)
            img_h = float(rec.get("img_h") or 480.0)
        elif isinstance(rec.get("image_size"), dict):
            img_w = float(rec["image_size"].get("width", 640.0))
            img_h = float(rec["image_size"].get("height", 480.0))
        else:
            img_w = float(rec.get("image_w", rec.get("width", rec.get("w", 640.0))))
            img_h = float(rec.get("image_h", rec.get("height", rec.get("h", 480.0))))

    bbox = rec.get("bbox")
    if bbox is None and isinstance(rec.get("persons"), list) and rec["persons"]:
        bbox = rec["persons"][0].get("bbox")
    
    final_bbox = None
    if isinstance(bbox, dict):
        if all(k in bbox for k in ("x", "y", "w", "h")):
            x, y, w, h = float(bbox["x"]), float(bbox["y"]), float(bbox["w"]), float(bbox["h"])
            final_bbox = [x, y, x + w, y + h]
        elif all(k in bbox for k in ("cx", "cy", "w", "h")):
            cx, cy, w, h = float(bbox["cx"]), float(bbox["cy"]), float(bbox["w"]), float(bbox["h"])
            final_bbox = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
        elif all(k in bbox for k in ("x1", "y1", "x2", "y2")):
            final_bbox = [float(bbox["x1"]), float(bbox["y1"]), float(bbox["x2"]), float(bbox["y2"])]
    elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        final_bbox = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]

    kps = rec.get("kps")
    if kps is None and isinstance(rec.get("persons"), list) and rec["persons"]:
        kps = rec["persons"][0].get("keypoints")
    
    out_kps: List[dict] = []
    if isinstance(kps, list) and kps and isinstance(kps[0], (int, float)) and len(kps) >= 51:
        temp_kps = []
        for i in range(17):
            temp_kps.append({
                "x": float(kps[3*i]), 
                "y": float(kps[3*i+1]), 
                "conf": float(kps[3*i+2])
            })
        kps = temp_kps

    if isinstance(kps, list):
        for p in kps[:17]:
            if isinstance(p, dict):
                x = float(p.get("x", p.get("X", 0.0)))
                y = float(p.get("y", p.get("Y", 0.0)))
                c = float(p.get("conf", p.get("confidence", p.get("score", 1.0))))
                out_kps.append({"x": x, "y": y, "conf": c})
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                x = float(p[0])
                y = float(p[1])
                c = float(p[2]) if len(p) > 2 else 1.0
                out_kps.append({"x": x, "y": y, "conf": c})

    return final_bbox, out_kps, img_w, img_h

# ===================== 插值 =====================
def _lin(v1, v2, a):
    if v1 is None and v2 is None: return 0.0
    if v1 is None: return float(v2)
    if v2 is None: return float(v1)
    return float(v1)*(1.0-a) + float(v2)*a

def _interp_frame(prev_rec, curr_rec, alpha=0.5, fixed_w=None, fixed_h=None):
    p_bbox, p_kps, p_w, p_h = _extract_basic_frame(prev_rec, fixed_w, fixed_h)
    c_bbox, c_kps, c_w, c_h = _extract_basic_frame(curr_rec, fixed_w, fixed_h)
    bbox = None
    if p_bbox and c_bbox:
        bbox = [_lin(p_bbox[i], c_bbox[i], alpha) for i in range(4)]
    else:
        bbox = c_bbox or p_bbox
    kps = []
    for i in range(17):
        px = py = pc = None
        cx = cy = cc = None
        if i < len(p_kps): px, py, pc = p_kps[i]['x'], p_kps[i]['y'], p_kps[i]['conf']
        if i < len(c_kps): cx, cy, cc = c_kps[i]['x'], c_kps[i]['y'], c_kps[i]['conf']
        kps.append({
            "x": _lin(px, cx, alpha),
            "y": _lin(py, cy, alpha),
            "conf": _lin(pc, cc, alpha)
        })
    img_w = c_w if c_w > 0 else (p_w if p_w > 0 else 640.0)
    img_h = c_h if c_h > 0 else (p_h if p_h > 0 else 480.0)
    ts_mid = None
    if prev_rec.get("ts_ms") and curr_rec.get("ts_ms"):
        ts_mid = int((prev_rec["ts_ms"] + curr_rec["ts_ms"]) / 2)
    return {
        "type": "frame_interp", "frame_id": prev_rec.get("frame_id"), "ts_ms": ts_mid,
        "bbox": bbox, "kps": kps, "detections": [], "img_w": img_w, "img_h": img_h, "synth": True
    }

# ===================== Stage Model Wrapper =====================
class _StageModels:
    def __init__(self, stage: str, device: torch.device):
        assert stage in ("binary", "multi")
        self.stage = stage
        self.m = bl if stage == "binary" else ml 
        self.device = device

        self.H = int(H)
        self.W = int(W)
        self.INCLUDE_BONE = bool(INCLUDE_BONE_LINES)
        self.OBJECT_CLASSES = list(OBJECT_CLASSES)
        self.USE_MOTION = bool(USE_MOTION)
        self.MOTION_DIM = int(MOTION_DIM)
        
        self.NUM_EDGES = len(getattr(self.m, "COCO_EDGES", []))
        in_ch = _calc_stage_in_channels(self.stage, self.INCLUDE_BONE, len(self.OBJECT_CLASSES), self.NUM_EDGES)

        CNNLSTM = getattr(self.m, "CNNLSTM")
        self.model = CNNLSTM(
            in_ch=in_ch, num_classes=1, cnn_out=256, lstm_h=LSTM_HIDDEN, lstm_layers=2,
            bidirectional=BIDIRECTIONAL, temporal_pool=TEMPORAL_POOL, dropout=DROPOUT,
            motion_dim=(self.MOTION_DIM if self.USE_MOTION else 0),
        ).to(self.device)

        self.class_names: List[str] = []
        
        # Motion Norm Params
        self.motion_mean = np.zeros(self.MOTION_DIM, dtype=np.float32)
        self.motion_std = np.ones(self.MOTION_DIM, dtype=np.float32)

    def load_weights(self, model_path: str, classes_path: Optional[str] = None):
        if not os.path.exists(model_path):
            print(f"[WARN] {self.stage} model not found at {model_path}")
            return

        sd = torch.load(model_path, map_location=self.device)
        state = sd.get("model_state", sd) if isinstance(sd, dict) else sd
        class_names = sd.get("class_names") if isinstance(sd, dict) else None
        
        if isinstance(sd, dict) and "motion_norm" in sd:
            mn = sd["motion_norm"]
            self.motion_mean = np.array(mn.get("mean", [0.0]*self.MOTION_DIM), dtype=np.float32)
            self.motion_std = np.array(mn.get("std", [1.0]*self.MOTION_DIM), dtype=np.float32)
            print(f"[{self.stage}] Loaded motion_norm: mean_avg={self.motion_mean.mean():.3f}, std_avg={self.motion_std.mean():.3f}")

        if class_names is None and classes_path:
             if os.path.exists(classes_path):
                with open(classes_path, "r") as f:
                    class_names = json.load(f)

        if not class_names:
            class_names = ["non_fall", "fall"] if self.stage == "binary" else ["none"]
            print(f"[WARN] No class names found for {self.stage}, using default: {class_names}")

        # Auto-align channels
        first_key = next((k for k in state.keys() if "cnn.net.0.weight" in k or state[k].ndim==4), None)
        if first_key:
            expected_in = int(state[first_key].shape[1])
            curr_in = _calc_stage_in_channels(self.stage, self.INCLUDE_BONE, len(self.OBJECT_CLASSES), self.NUM_EDGES)
            
            if curr_in != expected_in:
                print(f"[{self.stage}] Channel mismatch: Code={curr_in}, Model={expected_in}. Auto-aligning...")
                found_config = False
                original_objs = list(OBJECT_CLASSES)
                for try_bone in [True, False]:
                    try_edges = self.NUM_EDGES if try_bone else 0
                    base_ch = _calc_stage_in_channels(self.stage, try_bone, 0, try_edges)
                    needed_obj = expected_in - base_ch
                    if needed_obj >= 0:
                        self.INCLUDE_BONE = try_bone
                        if len(original_objs) >= needed_obj:
                            self.OBJECT_CLASSES = original_objs[:needed_obj]
                        else:
                            self.OBJECT_CLASSES = original_objs + [f"obj{i}" for i in range(len(original_objs), needed_obj)]
                        print(f"[{self.stage}] Aligned success: Bone={self.INCLUDE_BONE}, Edges={try_edges}, ObjCount={len(self.OBJECT_CLASSES)}")
                        found_config = True
                        break
                if not found_config:
                    print(f"[{self.stage}] FATAL: Could not match model channels {expected_in}. Check configs.")

        final_in = _calc_stage_in_channels(self.stage, self.INCLUDE_BONE, len(self.OBJECT_CLASSES), self.NUM_EDGES)
        CNNLSTM = getattr(self.m, "CNNLSTM")
        self.model = CNNLSTM(
            in_ch=final_in, num_classes=len(class_names), cnn_out=256, lstm_h=LSTM_HIDDEN, lstm_layers=2,
            bidirectional=BIDIRECTIONAL, temporal_pool=TEMPORAL_POOL, dropout=DROPOUT,
            motion_dim=(self.MOTION_DIM if self.USE_MOTION else 0),
        ).to(self.device)

        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.class_names = list(class_names)

    @torch.no_grad()
    def infer(self, parsed_window: List[Tuple[dict, list, list, float, float]]) -> Dict[str, Any]:
        """
        Parsed Window: (bbox, kps, dets, img_w, img_h)
        """
        # Fill missing kps
        if hasattr(self.m, "fill_missing_kps_inference"):
             parsed_window = self.m.fill_missing_kps_inference(parsed_window, max_missing=3)

        # 1. Motion & Mask
        motion_feats, valid_mask = self.m.compute_motion_feats_with_mask_from_parsed(
            parsed_window, conf_th=KP_CONF_TH
        )
        
        # Motion Normalization
        if self.USE_MOTION and self.MOTION_DIM > 0:
            motion_feats = (motion_feats - self.motion_mean) / self.motion_std

        # 2. Relation Map
        cfg = self.m.RelationMapConfig(
            H=self.H, W=self.W, sigma_kp=SIGMA_KP, kp_conf_th=KP_CONF_TH,
            include_bone_lines=self.INCLUDE_BONE, object_classes=self.OBJECT_CLASSES
        )
        frames = [
            self.m.rasterize_frame(bbox, kps, dets, img_w, img_h, cfg)
            for (bbox, kps, dets, img_w, img_h) in parsed_window
        ]

        # 3. Batching
        x = torch.from_numpy(np.stack(frames)).unsqueeze(0).float().to(self.device)
        mask = torch.from_numpy(valid_mask).unsqueeze(0).float().to(self.device)
        M = None
        if self.USE_MOTION and self.MOTION_DIM > 0:
            M = torch.from_numpy(motion_feats).unsqueeze(0).float().to(self.device)

        x = torch.nan_to_num(x, 0.0)
        if M is not None: M = torch.nan_to_num(M, 0.0)

        # Multi Model Mismatch Fix (52->36)
        if self.stage == "multi" and x.shape[2] == 52 and self.model.cnn.net[0].weight.shape[1] == 36:
            mask_dist = x[:, :, 0:2, :, :]      
            kps_separate = x[:, :, 2:19, :, :]  
            rest = x[:, :, 19:, :, :]           
            kps_combined, _ = torch.max(kps_separate, dim=2, keepdim=True)
            x = torch.cat([mask_dist, kps_combined, rest], dim=2)

        # 4. Forward
        logits = self.model(x, motion=M, mask=mask)
        prob = _safe_softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(prob.argmax())
        
        return {
            "class_names": self.class_names,
            "probs": prob.tolist(),
            "pred_idx": pred_idx,
            "pred": self.class_names[pred_idx]
        }

# ===================== Target Tracker =====================
def _bbox_area(b): return max(0, b[2]-b[0]) * max(0, b[3]-b[1]) if b else 0
def _iou(b1, b2):
    if not b1 or not b2: return 0
    ix = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
    iy = max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
    i = ix*iy
    u = _bbox_area(b1) + _bbox_area(b2) - i
    return i/(u+1e-6)

class TargetTracker:
    def __init__(self, timeout=5.0):
        self.last_bbox = None
        self.last_ts = time.time()
        self.timeout = timeout
        self.max_ratio = MAX_SCREEN_RATIO
        self.min_kps = MIN_VALID_KPS
    
    def update(self, persons, img_w):
        now = time.time()
        reset = False
        if (now - self.last_ts) > self.timeout:
            self.last_bbox = None
            reset = True
        
        if not persons: return None, reset, None
        self.last_ts = now

        # [NEW] Pre-filtering (Giant Object & KPS Check)
        processed_persons = []
        reject_reasons = set()

        for p in persons:
            bbox = p.get("bbox")
            if not bbox: continue
            
            # 1. 寬度佔比檢查
            bw = bbox[2] - bbox[0]
            if bw > img_w * 0.85: 
                reject_reasons.add("screen_blocked")
                continue
                
            # 2. 面積估算檢查
            bh = bbox[3] - bbox[1]
            est_img_h = img_w * 0.75
            area_ratio = (bw * bh) / (img_w * est_img_h)
            if area_ratio > self.max_ratio:
                reject_reasons.add("screen_blocked")
                continue
                
            # 3. 骨架點數量檢查
            kps = p.get("keypoints", [])
            valid_cnt = sum(1 for k in kps if k.get("conf", 0) > KP_CONF_TH)
            if valid_cnt < self.min_kps:
                reject_reasons.add("poor_quality")
                continue
                
            processed_persons.append(p)

        # 如果所有候選人都被過濾掉了
        if not processed_persons:
            # 優先回傳遮擋原因
            if "screen_blocked" in reject_reasons:
                return None, reset, "screen_blocked"
            if "poor_quality" in reject_reasons:
                return None, reset, "poor_quality"
            return None, reset, None

        # Track Logic
        if self.last_bbox:
            best_p, best_iou = None, -1
            for p in processed_persons:
                iou = _iou(self.last_bbox, p["bbox"])
                if iou > best_iou: best_iou, best_p = iou, p
            if best_iou > 0.3:
                self.last_bbox = best_p["bbox"]
                return best_p, reset, None
        
        cands = [( _bbox_area(p.get("bbox")), p) for p in processed_persons]
        cands.sort(key=lambda x:x[0], reverse=True)
        if cands:
            p = cands[0][1]
            self.last_bbox = p["bbox"]
            return p, reset, None
            
        return None, reset, None

# ===================== Main Manager =====================
class StreamInferManager:
    def __init__(self):
        self.device = _torch_device()
        self.buffers = defaultdict(deque)
        self.handlers = {}
        self._user_resolutions: Dict[str, Tuple[float, float]] = {}
        self._user_locations: Dict[str, str] = {}

        self.bin_stage = _StageModels("binary", self.device)
        self.mul_stage = _StageModels("multi", self.device)
        self._load_models()

        self._in_fall = defaultdict(bool)
        self._start_hits = defaultdict(int)
        self._recover_hits = defaultdict(int)
        self._fall_peak = defaultdict(float)
        self._fall_start_ts = {}

        self._curr_action = defaultdict(lambda: None)
        self._action_start_time = {}
        self._action_peak = defaultdict(float)
        self._cand_action = defaultdict(lambda: None)
        self._cand_hits = defaultdict(int)

        self._locks = defaultdict(asyncio.Lock)
        self._trackers = defaultdict(TargetTracker)
        self._pending = defaultdict(dict)
        self._last_real_frame = defaultdict(lambda: None)
        self._last_multi_tail_seq = defaultdict(lambda: None)
        self._next_infer_tail_seq = defaultdict(lambda: None)

        self._metrics = defaultdict(lambda: {
            "recv_count": 0, "net_ms_sum": 0.0, "infer_ms_sum": 0.0, 
            "infer_count": 0, "last_report_ts": time.time(), "last_ts_ms": None, "fps_ema": None
        })

    def _load_models(self):
        self.bin_stage.load_weights(BIN_MODEL_PATH, BIN_CLASSES_PATH)
        self.mul_stage.load_weights(MULTI_MODEL_PATH, MULTI_CLASSES_PATH)
        print(f"[STREAM] Models loaded. Binary={len(self.bin_stage.class_names)}, Multi={len(self.mul_stage.class_names)}")

    def set_handlers(self, **kwargs):
        for k, v in kwargs.items():
            if v: self.handlers[k] = v

    def _spawn(self, coro, tag="task"):
        asyncio.create_task(coro)

    async def ingest(self, user_id: str, data: Dict[str, Any]):
        merged = self._merge_packet(user_id, data)
        if merged is None: return

        if user_id not in self._user_resolutions:
            w = merged.get("img_w", 0)
            h = merged.get("img_h", 0)
            if w > 0 and h > 0:
                print(f"[STREAM] Locking resolution for user {user_id}: {int(w)}x{int(h)}")
                self._user_resolutions[user_id] = (float(w), float(h))
        
        if "location" in merged and merged["location"]:
            current_loc = merged["location"]
            if self._user_locations.get(user_id) != current_loc:
                print(f"[STREAM] Location update for user {user_id}: {current_loc}")
                self._user_locations[user_id] = current_loc

        fixed_w, fixed_h = self._user_resolutions.get(user_id, (None, None))
        tracker = self._trackers[user_id]
        img_w = fixed_w if fixed_w else merged.get("img_w", 640.0) 
        
        raw_persons = merged.get("raw_persons", [])
        
        # [NEW] Check update result for warning
        target, reset, warning = tracker.update(raw_persons, img_w)
        
        # [NEW] Send warning if exists
        if warning:
            asyncio.create_task(ws_manager.send(user_id, {
                "type": "warning",
                "code": warning,
                "timestamp": _now_str()
            }))
        
        if reset:
            self.buffers[user_id].clear()
            self._last_real_frame[user_id] = None
        
        if target:
            merged["bbox"] = target.get("bbox")
            merged["kps"] = target.get("keypoints")
            merged.pop("raw_persons", None)
        else:
            return 

        _, kps, _, _ = _extract_basic_frame(merged, fixed_w=fixed_w, fixed_h=fixed_h)
        if not kps: return

        buf = self.buffers[user_id]
        fid = int(merged.get("frame_id") or 0)
        
        if fixed_w: merged["img_w"] = fixed_w
        if fixed_h: merged["img_h"] = fixed_h

        if not INTERP_DOUBLE:
            merged["frame_seq"] = fid
            self._insert_buf(buf, merged)
        else:
            prev = self._last_real_frame[user_id]
            if prev and prev.get("frame_id") is not None:
                synth = _interp_frame(prev, merged, 0.5, fixed_w=fixed_w, fixed_h=fixed_h)
                synth["frame_seq"] = prev["frame_id"] * 2 + 1
                self._insert_buf(buf, synth)
            
            merged["frame_seq"] = fid * 2
            self._insert_buf(buf, merged)
            self._last_real_frame[user_id] = merged

        tail_seq = buf[-1]["frame_seq"]
        next_seq = self._next_infer_tail_seq[user_id]
        if next_seq is None: 
            self._next_infer_tail_seq[user_id] = tail_seq
            next_seq = tail_seq
        
        if len(buf) >= BIN_WINDOW and tail_seq >= next_seq:
            clip = list(buf)[-BIN_WINDOW:]
            await self._run_two_stage(user_id, clip, fixed_w, fixed_h)
            self._next_infer_tail_seq[user_id] = tail_seq + BIN_STRIDE
            
        self._pending[user_id].pop(str(fid), None)

    def _insert_buf(self, buf, item):
        ks = item["frame_seq"]
        if not buf: 
            buf.append(item)
            return
        if buf[-1]["frame_seq"] < ks:
            buf.append(item)
        elif buf[-1]["frame_seq"] == ks:
            buf[-1] = item
        else:
            for i in range(len(buf)-1, -1, -1):
                if buf[i]["frame_seq"] < ks:
                    buf.insert(i+1, item)
                    return
                if buf[i]["frame_seq"] == ks:
                    buf[i] = item
                    return
            buf.appendleft(item)

    def _merge_packet(self, uid, pkt):
        t = pkt.get("type")
        fid = str(pkt.get("frame_id"))
        
        loc = pkt.get("location")
        w = h = None
        if "image_size" in pkt:
            w = pkt["image_size"].get("width")
            h = pkt["image_size"].get("height")
        elif "img_w" in pkt:
            w = pkt.get("img_w")
            h = pkt.get("img_h")

        if t == "frame":
            ps = pkt.get("persons", [])
            return {
                "type":"frame", "frame_id": int(fid), "ts_ms": pkt.get("ts_ms"),
                "raw_persons": ps, "bbox": ps[0].get("bbox") if ps else None,
                "kps": ps[0].get("keypoints") if ps else [],
                "detections": pkt.get("detections", []),
                "img_w": w or 640, "img_h": h or 480,
                "location": loc 
            }
        
        pend = self._pending[uid].setdefault(fid, {"frame_id": int(fid), "ts_ms": pkt.get("ts_ms")})
        if loc: pend["location"] = loc
        if w: pend["img_w"] = w
        if h: pend["img_h"] = h

        if t == "pose":
            pend["raw_persons"] = pkt.get("persons", [])
            if "raw_persons" in pend: return pend.copy() 
        if t == "object":
            pend["detections"] = pkt.get("detections", [])
        return None

    async def _run_two_stage(self, user_id, clip, fixed_w, fixed_h):
        async with self._locks[user_id]:
            parsed_bin = []
            for rec in clip:
                b, k, w, h = _extract_basic_frame(rec, fixed_w=fixed_w, fixed_h=fixed_h)
                d = rec.get("detections", [])
                parsed_bin.append((b, k, d, w, h))
            
            t0 = time.perf_counter()
            bin_out = self.bin_stage.infer(parsed_bin)
            bin_pred = bin_out["pred"]
            
            mul_out = None
            if bin_pred != "fall":
                mul_out = self.mul_stage.infer(parsed_bin)

            res = {
                "type": "inference",
                "track_info": {"bbox": self._trackers[user_id].last_bbox},
                "binary": {
                    "pred": bin_pred, "probs": bin_out["probs"], 
                    "class_names": self.bin_stage.class_names
                }
            }
            if mul_out:
                res["multi"] = {
                    "pred": mul_out["pred"], "probs": mul_out["probs"],
                    "class_names": self.mul_stage.class_names
                }
                res["event_name"] = mul_out["pred"]
            
            asyncio.create_task(ws_manager.send(user_id, res))
            
            await self._update_fall_state(user_id, bin_pred, bin_out["probs"], clip, res)
            if mul_out:
                await self._update_action_state(user_id, mul_out, clip, res)

    async def _update_fall_state(self, uid, pred, probs, clip, payload):
        try:
            fall_idx = self.bin_stage.class_names.index("fall")
            score = probs[fall_idx]
        except:
            score = 0.0

        in_fall = self._in_fall[uid]
        if not in_fall:
            if pred == "fall" and score >= FALL_START_THR:
                self._start_hits[uid] += 1
            else:
                self._start_hits[uid] = 0
            
            if self._start_hits[uid] >= FALL_START_HITS:
                self._in_fall[uid] = True
                self._start_hits[uid] = 0
                self._fall_start_ts[uid] = _now_str()
                self._fall_peak[uid] = score
                
                curr_loc = self._user_locations.get(uid, "Unknown")
                
                if "on_fall_start" in self.handlers:
                    asyncio.create_task(self.handlers["on_fall_start"](
                        user_id=uid, start_time=self._fall_start_ts[uid],
                        result=payload, clip={"start":clip[0], "end":clip[-1]},
                        location=curr_loc
                    ))
                if self._curr_action[uid]:
                    await self._force_recover_action(uid, reason="fall_start")

        else: 
            if score > self._fall_peak[uid]: self._fall_peak[uid] = score
            if score <= FALL_RECOVER_THR:
                self._recover_hits[uid] += 1
            else:
                self._recover_hits[uid] = 0
            
            if self._recover_hits[uid] >= FALL_RECOVER_HITS:
                self._in_fall[uid] = False
                self._recover_hits[uid] = 0
                if "on_fall_recover" in self.handlers:
                    asyncio.create_task(self.handlers["on_fall_recover"](
                        user_id=uid, start_time=self._fall_start_ts[uid],
                        end_time=_now_str(), peak_score=self._fall_peak[uid],
                        result=payload
                    ))

    async def _update_action_state(self, uid, out, clip, payload):
        pred = out["pred"]
        score = out["probs"][out["pred_idx"]]
        if pred not in ENABLED_ACTIONS:
            return

        curr = self._curr_action[uid]
        cand = self._cand_action[uid]
        
        if curr is None:
            if score >= ACTION_START_THR:
                if cand == pred:
                    self._cand_hits[uid] += 1
                else:
                    self._cand_action[uid] = pred
                    self._cand_hits[uid] = 1
            else:
                self._cand_action[uid] = None
                self._cand_hits[uid] = 0
            
            if self._cand_hits[uid] >= ACTION_START_HITS:
                self._curr_action[uid] = pred
                self._action_start_time[uid] = _now_str()
                self._action_peak[uid] = score
                self._cand_action[uid] = None
                self._cand_hits[uid] = 0
                if "on_state_event_start" in self.handlers:
                    asyncio.create_task(self.handlers["on_state_event_start"](
                        user_id=uid, event_name=pred, start_time=self._action_start_time[uid],
                        peak_score=score, clip={"start":clip[0], "end":clip[-1]}, payload=payload
                    ))
        else:
            if curr == pred:
                if score > self._action_peak[uid]: self._action_peak[uid] = score
                self._cand_action[uid] = None
            else:
                if score >= ACTION_START_THR:
                    if cand == pred:
                        self._cand_hits[uid] += 1
                    else:
                        self._cand_action[uid] = pred
                        self._cand_hits[uid] = 1
                
                if self._cand_hits[uid] >= ACTION_START_HITS:
                    await self._force_recover_action(uid, reason="switch")
                    self._curr_action[uid] = pred
                    self._action_start_time[uid] = _now_str()
                    self._action_peak[uid] = score
                    self._cand_action[uid] = None
                    self._cand_hits[uid] = 0
                    if "on_state_event_start" in self.handlers:
                        asyncio.create_task(self.handlers["on_state_event_start"](
                            user_id=uid, event_name=pred, start_time=self._action_start_time[uid],
                            peak_score=score, clip={"start":clip[0], "end":clip[-1]}, payload=payload
                        ))

    async def _force_recover_action(self, uid, reason=""):
        act = self._curr_action[uid]
        if not act: return
        if "on_state_event_recover" in self.handlers:
            asyncio.create_task(self.handlers["on_state_event_recover"](
                user_id=uid, event_name=act, 
                start_time=self._action_start_time.get(uid),
                end_time=_now_str(),
                peak_score=self._action_peak[uid],
                prev_action_name=act
            ))
        self._curr_action[uid] = None
        self._action_start_time.pop(uid, None)
        self._action_peak[uid] = 0.0

    async def force_recover(self, user_id, reason="manual"):
        await self._force_recover_action(user_id, reason)
        self._in_fall[user_id] = False
        self._buffers = defaultdict(deque) 
        if user_id in self.buffers: del self.buffers[user_id]
        if user_id in self._user_resolutions: del self._user_resolutions[user_id]
        if user_id in self._trackers: del self._trackers[user_id]
        if user_id in self._user_locations: del self._user_locations[user_id]
        self._pending[user_id].clear()
        self._last_real_frame[user_id] = None
        print(f"[STREAM] Force recovered {user_id}: {reason}")

stream_infer_manager = StreamInferManager()