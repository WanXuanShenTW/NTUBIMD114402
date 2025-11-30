# app/utils/stream_infer_manager.py
# -*- coding: utf-8 -*-
"""
Stream Inference Manager — WebSocket → 聚合(frame_id) → 前處理 → 二階段 CNN+LSTM → 事件/回傳

本版本重點：
1) ✅ 不再在本檔內自寫簡化工具函式；「全部」改用對應 loader 的**正式函式**
- RelationMap：使用 loader.RelationMapConfig / loader.rasterize_frame
- Motion + Mask：使用 loader.compute_motion_feats_with_mask_from_parsed
- 模型：使用 loader.CNNLSTM（binary 與 multi 各自對應自己的 loader）
2) ✅ 設定「統一在核心檔」：H/W、是否畫骨架線、物件通道、KF/Mask 門檻、視窗/步幅、事件門檻…
- 仍可用環境變數覆寫，方便部署時調整
3) ✅ 權重通道自動對齊（依 ckpt 第一層 conv 的 in_channels 反推 include_bone 與物件通道數）
4) ✅ 以 frame_seq 控頻：每差滿 STRIDE 才觸發一次推論（避免浪費）
5) ✅ 跌倒（binary）與多動作（multi）的開始/恢復狀態機 + WS 非阻塞回傳
"""

import os
import asyncio
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict, deque
from datetime import datetime
import time

import numpy as np
import torch

# === 導入兩個 loader（請確保檔案位於 app/utils/ 下）===
from . import binary_cnn_lstm_loader as bl
from . import multi_cnn_lstm_loader as ml

# WS 管理（保持原專案結構）
from .ws_connection_manager import ws_manager


# ===================== 統一設定（可用環境變數覆寫） =====================
# RelationMap 與物件
H = int(os.getenv("SC_REL_H", "64"))
W = int(os.getenv("SC_REL_W", "64"))
INCLUDE_BONE_LINES = bool(int(os.getenv("SC_INCLUDE_BONE", "1")))
# 物件通道（逗號分隔），若不用物件請設為空字串 ""
OBJECT_CLASSES = [
    s.strip()
    for s in os.getenv("SC_OBJECT_CLASSES", "bed,chair,bench").split(",")
    if s.strip()
]

# Keypoint 熱圖與 Mask
KP_CONF_TH = float(os.getenv("SC_KP_CONF_TH", "0.6"))
SIGMA_KP = float(os.getenv("SC_SIGMA_KP", "2.0"))

# Motion
USE_MOTION = bool(int(os.getenv("SC_USE_MOTION", "1")))
MOTION_DIM = int(os.getenv("SC_MOTION_DIM", "9"))

# LSTM / 時序
LSTM_HIDDEN = int(os.getenv("SC_LSTM_H", "256"))
BIDIRECTIONAL = bool(int(os.getenv("SC_BIDIR", "0")))
TEMPORAL_POOL = os.getenv("SC_TEMPORAL_POOL", "attn")  # "last" | "mean" | "attn"
DROPOUT = float(os.getenv("SC_DROPOUT", "0.3"))

# 二階段視窗/步幅（建議兩階相同：10x5）
BIN_WINDOW = int(os.getenv("SC_BIN_WINDOW", "10"))
BIN_STRIDE = int(os.getenv("SC_BIN_STRIDE", "5"))
MULTI_WINDOW = int(os.getenv("SC_MULTI_WINDOW", "10"))
MULTI_STRIDE = int(os.getenv("SC_MULTI_STRIDE", "5"))

# 權重/類別檔路徑（可覆寫）
BIN_MODEL_PATH = os.getenv("SC_BIN_MODEL_PATH", "models/binary/best.pt")
BIN_CLASSES_PATH = os.getenv("SC_BIN_CLASSES_PATH", "models/binary/classes.json")
MULTI_MODEL_PATH = os.getenv("SC_MULTI_MODEL_PATH", "models/multi/best.pt")
MULTI_CLASSES_PATH = os.getenv("SC_MULTI_CLASSES_PATH", "models/multi/classes.json")

# 事件門檻與連續命中
FALL_START_THR = float(os.getenv("SC_FALL_START_THR", "0.70"))
FALL_RECOVER_THR = float(os.getenv("SC_FALL_RECOVER_THR", "0.50"))
FALL_START_HITS = int(os.getenv("SC_FALL_START_HITS", "2"))
FALL_RECOVER_HITS = int(os.getenv("SC_FALL_RECOVER_HITS", "3"))

# （多動作示例）
ACTION_START_THR = float(os.getenv("SC_ACTION_START_THR", "0.50"))
ACTION_START_HITS = int(os.getenv("SC_ACTION_START_HITS", "3"))
ENABLED_ACTIONS = [
    s.strip()
    for s in os.getenv("SC_ENABLED_ACTIONS", "walk,sitstill,liestill").split(",")
    if s.strip()
]

# 幀插值（骨架倍頻）開關：0=關，1=開（將相鄰兩幀插一幀，變成兩倍幀率）
INTERP_DOUBLE = bool(int(os.getenv("SC_INTERP_DOUBLE", "0")))

# 指標列印
METRICS_ON = bool(int(os.getenv("SC_METRICS", "0")))
METRICS_INTERVAL_MS = int(
    os.getenv("SC_METRICS_INTERVAL_MS", "5000")
)  # 每幾毫秒聚合印一次


# ===================== 小工具 =====================
def _now_str() -> str:
    try:
        from .response_util import now_str

        return now_str()
    except Exception:
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

def _torch_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _calc_in_channels(include_bone: bool, num_obj: int, num_edges: int) -> int:
    """C = 1(bbox_mask) + 1(dist) + 17(kps) + Edges + num_obj + 2(coord)"""
    e = num_edges if include_bone else 0
    return 1 + 1 + 17 + e + num_obj + 2

def _safe_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
    logits = logits - logits.max(dim=dim, keepdim=True).values
    return torch.softmax(logits, dim=dim)

# ===================== 解析單幀（容忍不同來源格式） =====================
def _extract_basic_frame(
    rec: Dict[str, Any],
) -> Tuple[Optional[list], List[dict], float, float]:
    """
    轉為 (bbox_xyxy, kps[{x,y,conf}] 17, img_w, img_h)
    支援：
      - 合包 frame: {"type":"frame","persons":[{"bbox":..,"keypoints":..}], "detections":.., "image_size":..}
      - 已扁平：{"bbox":..,"kps" or "keypoints":..,"img_w":..,"img_h":..}
      - 備援：若為 flat 51，也轉回 17 個點格式
    """
    # image size
    if "img_w" in rec and "img_h" in rec:
        img_w = float(rec.get("img_w") or 640.0)
        img_h = float(rec.get("img_h") or 480.0)
    elif isinstance(rec.get("image_size"), dict):
        img_w = float(rec["image_size"].get("width", 640.0))
        img_h = float(rec["image_size"].get("height", 480.0))
    else:
        img_w = float(rec.get("image_w", rec.get("width", rec.get("w", 640.0))))
        img_h = float(rec.get("image_h", rec.get("height", rec.get("h", 480.0))))

    # bbox
    bbox = rec.get("bbox")
    if bbox is None and isinstance(rec.get("persons"), list) and rec["persons"]:
        bbox = rec["persons"][0].get("bbox")
    if isinstance(bbox, dict):
        if all(k in bbox for k in ("x", "y", "w", "h")):
            x, y, w, h = (
                float(bbox["x"]),
                float(bbox["y"]),
                float(bbox["w"]),
                float(bbox["h"]),
            )
            bbox = [x, y, x + w, y + h]
        elif all(k in bbox for k in ("cx", "cy", "w", "h")):
            cx, cy, w, h = (
                float(bbox["cx"]),
                float(bbox["cy"]),
                float(bbox["w"]),
                float(bbox["h"]),
            )
            bbox = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
        elif all(k in bbox for k in ("x1", "y1", "x2", "y2")):
            bbox = [
                float(bbox["x1"]),
                float(bbox["y1"]),
                float(bbox["x2"]),
                float(bbox["y2"]),
            ]
        else:
            bbox = None
    elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        bbox = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
    else:
        bbox = None

    # keypoints
    kps = rec.get("kps")
    if kps is None and isinstance(rec.get("persons"), list) and rec["persons"]:
        kps = rec["persons"][0].get("keypoints")
    if isinstance(kps, list) and kps and isinstance(kps[0], (int, float)):
        # flat 51
        flat = kps
        if len(flat) >= 51:
            pts = []
            for i in range(17):
                x = float(flat[3 * i + 0])
                y = float(flat[3 * i + 1])
                c = float(flat[3 * i + 2])
                pts.append({"x": x, "y": y, "conf": c})
            kps = pts
        else:
            kps = []
    out_kps: List[dict] = []
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

    return bbox, out_kps, img_w, img_h

# ===================== 幀插值（可選；不做影像，只做骨架/框） =====================
def _lin(v1: Optional[float], v2: Optional[float], a: float) -> float:
    if (v1 is None) and (v2 is None):
        return 0.0
    if v1 is None:
        return float(v2)
    if v2 is None:
        return float(v1)
    return float(v1) * (1.0 - a) + float(v2) * a

def _interp_frame(
    prev_rec: Dict[str, Any], curr_rec: Dict[str, Any], alpha: float = 0.5
) -> Dict[str, Any]:
    p_bbox, p_kps, p_w, p_h = _extract_basic_frame(prev_rec)
    c_bbox, c_kps, c_w, c_h = _extract_basic_frame(curr_rec)

    # bbox
    if p_bbox and c_bbox:
        bbox = [
            _lin(p_bbox[0], c_bbox[0], alpha),
            _lin(p_bbox[1], c_bbox[1], alpha),
            _lin(p_bbox[2], c_bbox[2], alpha),
            _lin(p_bbox[3], c_bbox[3], alpha),
        ]
    else:
        bbox = c_bbox or p_bbox or None

    # kps（17 個點）
    kps: List[dict] = []
    for i in range(17):
        px = py = pc = None
        cx = cy = cc = None
        if i < len(p_kps):
            px = p_kps[i].get("x")
            py = p_kps[i].get("y")
            pc = p_kps[i].get("conf", 1.0)
        if i < len(c_kps):
            cx = c_kps[i].get("x")
            cy = c_kps[i].get("y")
            cc = c_kps[i].get("conf", 1.0)
        kps.append(
            {
                "x": _lin(px, cx, alpha),
                "y": _lin(py, cy, alpha),
                "conf": _lin(pc, cc, alpha),
            }
        )

    img_w = c_w or p_w or 640.0
    img_h = c_h or p_h or 480.0

    ts_mid = None
    try:
        p_ts = prev_rec.get("ts_ms")
        c_ts = curr_rec.get("ts_ms")
        if isinstance(p_ts, (int, float)) and isinstance(c_ts, (int, float)):
            ts_mid = int(round((float(p_ts) + float(c_ts)) / 2.0))
    except Exception:
        ts_mid = None

    return {
        "type": "frame_interp",
        "frame_id": prev_rec.get("frame_id"),
        "ts_ms": ts_mid,
        "bbox": bbox,
        "kps": kps,
        "detections": [],
        "img_w": img_w,
        "img_h": img_h,
        "synth": True,
    }

# ===================== 單階段打包（完全使用 loader 提供的 API） =====================
class _StageModels:
    def __init__(self, stage: str, device: torch.device):
        assert stage in ("binary", "multi")
        self.stage = stage
        self.m = bl if stage == "binary" else ml  # 對應 loader 模組
        self.device = device

        # === 使用「核心設定」，不依賴 loader 常數 ===
        self.H = int(H)
        self.W = int(W)
        self.INCLUDE_BONE = bool(INCLUDE_BONE_LINES)
        self.OBJECT_CLASSES = list(OBJECT_CLASSES)  # 可能在載入權重時被截斷/擴充
        # 邊數取對應 loader 定義（通常相同 COCO_EDGES）
        self.NUM_EDGES = (
            len(getattr(self.m, "COCO_EDGES", [])) if self.INCLUDE_BONE else 0
        )

        self.USE_MOTION = bool(USE_MOTION)
        self.MOTION_DIM = int(MOTION_DIM)

        # 先以目前設定建一次（load_weights 會依 ckpt in_ch 自動調整，並以 class_names 重建）
        in_ch = _calc_in_channels(
            self.INCLUDE_BONE, len(self.OBJECT_CLASSES), self.NUM_EDGES
        )
        CNNLSTM = getattr(self.m, "CNNLSTM")
        self.model = CNNLSTM(
            in_ch=in_ch,
            num_classes=1,  # 先給 1，稍後依 class_names 重建
            cnn_out=256,
            lstm_h=LSTM_HIDDEN,
            lstm_layers=2,
            bidirectional=BIDIRECTIONAL,
            temporal_pool=TEMPORAL_POOL,
            dropout=DROPOUT,
            motion_dim=(self.MOTION_DIM if self.USE_MOTION else 0),
        ).to(self.device)

        self.class_names: List[str] = []

    def load_weights(self, model_path: str, classes_path: Optional[str] = None):
        """讀取 checkpoint，並根據第一層 conv 權重的輸入通道數自動對齊"""
        sd = torch.load(model_path, map_location=self.device)
        state = None
        class_names = None

        if isinstance(sd, dict) and "model_state" in sd:
            state = sd["model_state"]
            class_names = sd.get("class_names")
        elif isinstance(sd, dict):
            state = sd
        else:
            raise RuntimeError(
                f"[{self.stage}] Unsupported checkpoint format: {type(sd)}"
            )

        # 類別
        if class_names is None:
            if classes_path and os.path.isfile(classes_path):
                import json

                with open(classes_path, "r", encoding="utf-8") as f:
                    class_names = json.load(f)
            else:
                guess = os.path.join(os.path.dirname(model_path), "classes.json")
                if os.path.isfile(guess):
                    import json

                    with open(guess, "r", encoding="utf-8") as f:
                        class_names = json.load(f)
        if not class_names:
            raise RuntimeError(f"[{self.stage}] class_names not found for {model_path}")

        # checkpoint 的第一層輸入通道數（優先找 SpaceCNN 起始 conv 的權重）
        first_key = None
        for k in state.keys():
            if k.endswith("cnn.net.0.weight"):
                first_key = k
                break
        if first_key is None:
            for k in state.keys():
                if getattr(state[k], "ndim", 0) == 4:
                    first_key = k
                    break
        if first_key is None:
            raise RuntimeError(
                f"[{self.stage}] Cannot locate first conv weight key in checkpoint"
            )

        expected_in = int(state[first_key].shape[1])

        # 目前設定的 in_ch
        curr_in = _calc_in_channels(
            self.INCLUDE_BONE, len(self.OBJECT_CLASSES), self.NUM_EDGES
        )

        # 若不一致，嘗試反推 include_bone 與物件通道數以對齊 expected_in
        if curr_in != expected_in:
            base_obj_names = list(OBJECT_CLASSES)  # ← 以「核心設定」為基底
            edges_full = len(getattr(self.m, "COCO_EDGES", []))
            for include_bone_try in (True, False):
                edges_try = edges_full if include_bone_try else 0
                base_fixed = 1 + 1 + 17 + edges_try + 2  # 除了物件通道以外
                num_obj_try = expected_in - base_fixed
                if 0 <= num_obj_try <= 128:
                    self.INCLUDE_BONE = include_bone_try
                    self.NUM_EDGES = edges_try
                    if len(base_obj_names) >= num_obj_try:
                        self.OBJECT_CLASSES = base_obj_names[:num_obj_try]
                    else:
                        extra = [
                            f"obj{i}" for i in range(len(base_obj_names), num_obj_try)
                        ]
                        self.OBJECT_CLASSES = base_obj_names + extra
                    break

        # 依對齊後的通道數重建模型（維持核心 LSTM 與 pool 設定）
        in_ch = _calc_in_channels(
            self.INCLUDE_BONE, len(self.OBJECT_CLASSES), self.NUM_EDGES
        )
        CNNLSTM = getattr(self.m, "CNNLSTM")
        self.model = CNNLSTM(
            in_ch=in_ch,
            num_classes=len(class_names),
            cnn_out=256,
            lstm_h=LSTM_HIDDEN,
            lstm_layers=2,
            bidirectional=BIDIRECTIONAL,
            temporal_pool=TEMPORAL_POOL,
            dropout=DROPOUT,
            motion_dim=(self.MOTION_DIM if self.USE_MOTION else 0),
        ).to(self.device)

        # 嚴格載入（若你需要對舊→新鍵名做映射，請在外層先行處理）
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.class_names = list(class_names)

    @torch.no_grad()
    def infer(
        self, parsed_window: List[Tuple[dict, list, list, float, float]]
    ) -> Dict[str, Any]:
        """
        對一個視窗做推論。
        parsed_window: list of (bbox, kps, dets, img_w, img_h)，長度 = WINDOW
        """
        # === Motion + Mask（完全交由 loader 計算）===
        motion_feats, valid_mask = self.m.compute_motion_feats_with_mask_from_parsed(
            parsed_window, conf_th=KP_CONF_TH
        )

        # === RelationMap → 張量（完全交由 loader 生成）===
        cfg = self.m.RelationMapConfig(
            H=self.H,
            W=self.W,
            sigma_kp=SIGMA_KP,
            kp_conf_th=KP_CONF_TH,
            include_bone_lines=self.INCLUDE_BONE,
            object_classes=self.OBJECT_CLASSES,
        )

        frames = [
            self.m.rasterize_frame(bbox, kps, dets, img_w, img_h, cfg)
            for (bbox, kps, dets, img_w, img_h) in parsed_window
        ]

        x = (
            torch.from_numpy(np.stack(frames)).unsqueeze(0).float().to(self.device)
        )  # (1,T,C,H,W)
        M = (
            torch.from_numpy(motion_feats).unsqueeze(0).float().to(self.device)
            if (self.USE_MOTION and self.MOTION_DIM > 0)
            else None
        )
        mask = torch.from_numpy(valid_mask).unsqueeze(0).float().to(self.device)

        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if M is not None:
            M = torch.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)

        logits = self.model(x, motion=M, mask=mask)
        prob = _safe_softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(prob.argmax())
        pred_name = self.class_names[pred_idx]
        return {
            "class_names": self.class_names,
            "probs": prob.tolist(),
            "pred_idx": pred_idx,
            "pred": pred_name,
        }

# ===================== [新增] 目標追蹤與幾何工具 =====================
def _bbox_area(bbox):
    # bbox: [x1, y1, x2, y2]
    if not bbox:
        return 0.0
    w = max(0, bbox[2] - bbox[0])
    h = max(0, bbox[3] - bbox[1])
    return w * h

def _iou(box1, box2):
    if not box1 or not box2:
        return 0.0
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    b1_area = _bbox_area(box1)
    b2_area = _bbox_area(box2)
    return inter_area / (b1_area + b2_area - inter_area + 1e-6)

class TargetTracker:
    def __init__(self, kp_conf_th=0.5, min_kps=3, iou_track_th=0.3, reset_timeout=5.0):
        self.last_bbox = None
        self.kp_conf_th = kp_conf_th
        self.min_kps = min_kps
        self.iou_track_th = iou_track_th

        # 重置機制狀態
        self.reset_timeout = reset_timeout
        self.last_seen_ts = time.time()  # 上次看到目標的時間
        self.last_exit_zone = None  # 離開時的區域 (0=左, 1=中, 2=右)

    def _get_zone(self, bbox, img_w):
        if not bbox or not img_w:
            return 1
        cx = (bbox[0] + bbox[2]) / 2.0
        if cx < img_w * 0.25:
            return 0  # Left
        if cx > img_w * 0.75:
            return 2  # Right
        return 1  # Center

    def update(self, persons: List[dict], img_w: float) -> Tuple[Optional[dict], bool]:
        """
        輸入: 本幀 person list
        輸出: (被鎖定的 person, 是否需要重置緩衝區)
        """
        now = time.time()
        should_reset = False

        # 1. 時間逾時重置 (Temporal Reset)
        # 如果距離上次看到人已經超過 timeout，強制重置
        if (now - self.last_seen_ts) > self.reset_timeout:
            self.last_bbox = None
            self.last_exit_zone = None
            should_reset = True
            # 這裡不 return，因為這次可能剛好有人進來，要接著判斷

        if not persons:
            # 本幀無人，不需要更新 last_seen_ts (讓它變舊)
            # 如果之前有人，記錄他離開的位置
            if self.last_bbox:
                self.last_exit_zone = self._get_zone(self.last_bbox, img_w)
            return None, should_reset

        # 有人，更新看見時間
        self.last_seen_ts = now

        # 2. Tracking (IOU)
        if self.last_bbox is not None:
            best_p = None
            best_iou = -1.0
            for p in persons:
                bbox = p.get("bbox")
                if bbox:
                    val = _iou(self.last_bbox, bbox)
                    if val > best_iou:
                        best_iou = val
                        best_p = p

            if best_iou >= self.iou_track_th and best_p:
                self.last_bbox = best_p.get("bbox")
                return best_p, should_reset
            else:
                # 追蹤丟失 (Tracking Lost)
                self.last_exit_zone = self._get_zone(self.last_bbox, img_w)
                self.last_bbox = None
                # 繼續往下走 Selection 邏輯

        # 3. Selection (重新鎖定)
        candidates = []
        for p in persons:
            kps = p.get("keypoints", [])
            bbox = p.get("bbox")
            if not bbox:
                continue

            # 簡單計算有效點 (這裡用簡易版以免拖慢伺服器)
            vk = sum(
                1
                for k in kps
                if (k.get("conf", k.get("confidence", 1.0)) >= self.kp_conf_th)
            )
            if vk >= self.min_kps:
                area = _bbox_area(bbox)
                candidates.append((area, p))

        if not candidates:
            return None, should_reset

        # 選面積最大的 (最近的)
        candidates.sort(key=lambda x: x[0], reverse=True)
        selected = candidates[0][1]

        # 4. 空間重置判斷 (Spatial Reset)
        # 如果是重新鎖定 (Re-entry)，且進入方向與上次離開方向不同 (e.g. 左出右進)
        current_zone = self._get_zone(selected.get("bbox"), img_w)
        if self.last_exit_zone is not None:
            # 如果離開和進入區域不同 (例如 0 vs 2)，視為不同人，重置 LSTM
            if self.last_exit_zone != current_zone:
                should_reset = True
                # print(f"[Tracker] Zone Reset: {self.last_exit_zone} -> {current_zone}")

        self.last_bbox = selected.get("bbox")
        # 重新鎖定成功，清除離開狀態
        self.last_exit_zone = None

        return selected, should_reset

# ===================== 中樞：StreamInferManager =====================
class StreamInferManager:
    """
    伺服器端推論中樞：
      - ingest(user_id, data): 收幀 → 聚合 → 緩衝 → 觸發推論
      - set_handlers(...)    : 注入事件 hooks
      - force_recover(...)   : 中斷/關閉時清理狀態
    """

    def __init__(self):
        self.device = _torch_device()

        # 原始幀緩衝：移除 maxlen，避免 insert() 於滿載時噴 IndexError
        self.buffers: Dict[str, deque] = defaultdict(lambda: deque())

        # 事件處理器（由 pose_routes.py 設定）
        self.handlers: Dict[str, Any] = {}

        # 二階段模型
        self.bin_stage = _StageModels("binary", device=self.device)
        self.mul_stage = _StageModels("multi", device=self.device)
        self._load_models()

        # 跌倒狀態機（連續命中 + 遲滯）
        self._in_fall = defaultdict(lambda: False)  # 是否目前處於 fall 事件中
        self._start_hits = defaultdict(int)  # 連續達 START_THR 次數
        self._recover_hits = defaultdict(int)  # 連續低於 RECOVER_THR 次數
        self._fall_start_ts = {}  # 事件開始時間字串
        self._fall_peak = defaultdict(float)  # 事件期間最高分

        # 每個使用者的推論鎖，避免同一 user 併發執行 _run_two_stage 造成競態
        self._locks = defaultdict(asyncio.Lock)

        # 保留最近一次二元分類（若其他邏輯需要）
        self.state_bin_pred: Dict[str, str] = defaultdict(lambda: "non_fall")

        # 拆包聚合暫存（user_id -> frame_id -> partial dict）
        self._pending: Dict[str, Dict[Any, Dict[str, Any]]] = defaultdict(dict)

        # [新增] 每個使用者的目標追蹤器
        self._trackers: Dict[str, TargetTracker] = defaultdict(lambda: TargetTracker())

        # 記住上一個「實際」幀（非插值），供插值用
        self._last_real_frame: Dict[str, Optional[Dict[str, Any]]] = defaultdict(
            lambda: None
        )

        # ===== 多動作狀態機（目前先針對 walk，可擴充）=====
        self._curr_action = defaultdict(lambda: None)  # 目前穩定中的動作（event）
        self._action_start_time = {}  # 動作開始時間
        self._action_peak = defaultdict(float)  # 目前動作期間最高分
        self._cand_action = defaultdict(lambda: None)  # 切換候選動作
        self._cand_hits = defaultdict(int)  # 候選動作連續命中次數

        # 用 frame_seq 控 multi 去重＆觸發頻率
        self._last_multi_tail_seq = defaultdict(lambda: None)
        self._last_infer_tail_seq = defaultdict(lambda: None)
        self._next_infer_tail_seq = defaultdict(lambda: None)

        # 指標：每 user 聚合統計
        self._metrics = defaultdict(
            lambda: {
                "recv_count": 0,
                "net_ms_sum": 0.0,
                "infer_ms_sum": 0.0,
                "infer_count": 0,
                "last_report_ts": time.time(),
                "last_ts_ms": None,  # 上一筆 ts_ms（估算輸入 FPS）
                "fps_ema": None,  # 輸入 FPS 的 EMA
                "budget_ms": None,  # BIN_STRIDE/fps_ema 推導的理論預算
            }
        )

    # --- fire-and-forget helpers ---
    def _spawn(self, coro, tag: str = "task"):
        async def _runner():
            try:
                await coro
            except Exception as e:
                print(f"[ASYNC][{tag}][ERROR] {e}")

        asyncio.create_task(_runner())

    def _spawn_chain(self, coros, tag: str = "chain"):
        async def _runner():
            for idx, c in enumerate(coros):
                try:
                    await c
                except Exception as e:
                    print(f"[ASYNC][{tag}][{idx}][ERROR] {e}")

        asyncio.create_task(_runner())

    def _insert_sorted_unique(self, buf: deque, rec: dict, key: str = "frame_seq"):
        """
        將單幀以 key（預設 frame_seq）為排序鍵插入 deque：
        - 若已存在同鍵，直接覆蓋該位置（不丟棄）
        - 否則按升冪插入正確位置，確保 buf[-1] 永遠是最新幀
        備註：deque.insert 是 O(n)，但我們的視窗長度很小（~40），可接受。
        """
        ks = rec.get(key)
        if ks is None:
            buf.append(rec)
            return
        n = len(buf)
        for i in range(n - 1, -1, -1):
            vi = buf[i].get(key)
            if vi == ks:
                buf[i] = rec
                return
            if (vi is not None) and (vi < ks):
                buf.insert(i + 1, rec)
                return
        buf.appendleft(rec)

    def _load_models(self):
        self.bin_stage.load_weights(BIN_MODEL_PATH, BIN_CLASSES_PATH)
        self.mul_stage.load_weights(MULTI_MODEL_PATH, MULTI_CLASSES_PATH)
        print(
            f"[STREAM] Loaded models. binary={len(self.bin_stage.class_names)} classes, "
            f"multi={len(self.mul_stage.class_names)} classes, device={self.device}"
        )

    def set_handlers(
        self,
        on_fall_start=None,
        on_fall_recover=None,
        on_state_event_start=None,
        on_state_event_recover=None,
    ):
        if on_fall_start:
            self.handlers["on_fall_start"] = on_fall_start
        if on_fall_recover:
            self.handlers["on_fall_recover"] = on_fall_recover
        if on_state_event_start:
            self.handlers["on_state_event_start"] = on_state_event_start
        if on_state_event_recover:
            self.handlers["on_state_event_recover"] = on_state_event_recover

        # 只做可觀測性：不改任務邏輯
        try:
            names = [
                k
                for k in (
                    "on_fall_start",
                    "on_fall_recover",
                    "on_state_event_start",
                    "on_state_event_recover",
                )
                if k in self.handlers
            ]
            print(f"[STREAM][HANDLERS] registered: {', '.join(names)}")
        except Exception:
            pass

    async def ingest(self, user_id: str, data: Dict[str, Any]):
        """
        接收 WS 一筆。
        流程：聚合 -> 追蹤(過濾人/重置緩衝) -> 緩衝 -> 觸發推論
        """
        merged = self._merge_packet(user_id, data)
        if merged is None:
            return  # 等待另一半

        # ====== [新增] Target Tracker & Reset Logic ======
        tracker = self._trackers[user_id]
        raw_persons = merged.get("raw_persons", [])
        img_w = merged.get("img_w", 640.0)

        # 1. 執行追蹤更新
        target_person, should_reset = tracker.update(raw_persons, img_w)

        # 2. 如果觸發重置 (逾時 or 空間不連續)，清空該使用者的緩衝區
        if should_reset:
            print(f"[STREAM][RESET] user={user_id} trigger reset (timeout or re-entry diff zone).")
            self.buffers[user_id].clear()
            self._last_real_frame[user_id] = None
            # 視需要重置狀態機 (選用，視業務邏輯而定)
            # self.force_recover(user_id, reason="tracker_reset") 

        # 3. 根據追蹤結果過濾數據
        if target_person:
            # 鎖定目標：將 merged 的 bbox/kps 替換為目標的數據
            merged["bbox"] = target_person.get("bbox")
            merged["kps"]  = target_person.get("keypoints")
            # 移除 raw_persons 節省記憶體
            merged.pop("raw_persons", None)
        else:
            # 無目標：
            # 若 Tracker 認為沒人(或被過濾光)，此幀應視為無效或空幀
            # 這裡選擇：不放入 buffer (丟棄此幀)，或者放入空數據
            # 為了避免 LSTM 斷掉，通常若只是一兩幀丟失可不入 buffer (靠時間差算 FPS)
            # 但若要讓 motion 計算正確，這裡直接 return 即可 (視為沒抓到人)
            return

        # 確保只有在有 kps 時才入 buffer（避免全 0）
        _, kps, _, _ = _extract_basic_frame(merged)
        if not kps:
            return

        # ====== [新增] 時間丟包 + 網路延遲度量 ======
        m = self._metrics[user_id]
        try:
            ts_ms = merged.get("ts_ms")
            if isinstance(ts_ms, (int, float)):
                now_ms = int(time.time() * 1000)
                lag = now_ms - int(ts_ms)

                # 網路/端上排隊延遲統計
                if lag >= 0:
                    m["net_ms_sum"] += lag
                    m["recv_count"] += 1

                # 估算輸入 FPS（用 ts_ms 差；EMA）
                last_ts = m["last_ts_ms"]
                m["last_ts_ms"] = int(ts_ms)
                if last_ts is not None and int(ts_ms) > int(last_ts):
                    inst_fps = 1000.0 / float(int(ts_ms) - int(last_ts))
                    if m["fps_ema"] is None:
                        m["fps_ema"] = inst_fps
                    else:
                        m["fps_ema"] = 0.2 * inst_fps + 0.8 * m["fps_ema"]
                    if m["fps_ema"] > 0:
                        m["budget_ms"] = (BIN_STRIDE * 1000.0) / m["fps_ema"]
        except Exception:
            pass

        # === frame_seq 與插值/倍幀 ===
        buf = self.buffers[user_id]

        fid_raw = merged.get("frame_id")
        try:
            fid_int = int(fid_raw) if fid_raw is not None else None
        except Exception:
            fid_int = None

        if not INTERP_DOUBLE:
            merged["frame_seq"] = fid_int
            # 亂序不丟棄：插入到正確位置（相同序號則覆蓋）
            self._insert_sorted_unique(buf, merged, key="frame_seq")

        else:
            # 先嘗試插入「插值幀」（夾在上一張實際幀與本次實際幀之間）
            prev_real = self._last_real_frame[user_id]
            if (
                prev_real is not None
                and (prev_real.get("frame_id") is not None)
                and (fid_int is not None)
            ):
                try:
                    prev_id = int(prev_real["frame_id"])
                except Exception:
                    prev_id = None
                if prev_id is not None:
                    synth = _interp_frame(prev_real, merged, alpha=0.5)
                    synth_seq = (
                        2 * prev_id + 1
                    )  # 奇數序：夾在 2*prev_id 與 2*fid_int 之間
                    synth["frame_seq"] = synth_seq
                    self._insert_sorted_unique(buf, synth, key="frame_seq")

            # 再插入「本次實際幀」（偶數序）
            real_seq = (2 * fid_int) if (fid_int is not None) else None
            merged["frame_seq"] = real_seq
            self._insert_sorted_unique(buf, merged, key="frame_seq")

            # 記住本次實際幀，供下次插值用
            self._last_real_frame[user_id] = merged

        # === 用 frame_seq 控制觸發頻率：尾序號每差滿 BIN_STRIDE 就推一次 ===
        tail_seq = buf[-1].get("frame_seq") if buf else None
        next_seq = self._next_infer_tail_seq[user_id]

        # 初始化：第一次看到尾序號時，先把目標設在「當前尾序號」
        if next_seq is None and tail_seq is not None:
            self._next_infer_tail_seq[user_id] = int(tail_seq)
            next_seq = self._next_infer_tail_seq[user_id]

        if (
            (len(buf) >= BIN_WINDOW)
            and (tail_seq is not None)
            and (next_seq is not None)
            and (int(tail_seq) >= int(next_seq))
        ):
            try:
                clip = list(buf)[-BIN_WINDOW:]
                await self._run_two_stage(user_id, clip)
                # 每次成功觸發後，下一次需等到「這次尾序號 + BIN_STRIDE」
                self._next_infer_tail_seq[user_id] = int(tail_seq) + BIN_STRIDE
            except Exception as e:
                print(f"[STREAM][ERROR] user={user_id} run_two_stage: {e}")

        # 清理已處理的 pending（保守）
        fid = merged.get("frame_id")
        if fid is not None:
            self._pending[user_id].pop(fid, None)

        # ====== 週期性輸出 metrics 彙總 ======
        if METRICS_ON:
            mm = self._metrics[user_id]
            now_sec = time.time()
            if (now_sec - mm["last_report_ts"]) * 1000.0 >= METRICS_INTERVAL_MS:
                avg_net = (
                    (mm["net_ms_sum"] / mm["recv_count"]) if mm["recv_count"] else 0.0
                )
                avg_infer = (
                    (mm["infer_ms_sum"] / mm["infer_count"])
                    if mm["infer_count"]
                    else 0.0
                )
                fps_in = mm["fps_ema"] if mm["fps_ema"] else 0.0
                budget = mm["budget_ms"] if mm["budget_ms"] else 0.0
                print(
                    f"[METRIC] user={user_id} "
                    f"fps_in≈{fps_in:.2f} net_ms(avg)={avg_net:.0f} "
                    f"infer_ms(avg)={avg_infer:.0f} budget≈{budget:.0f}ms "
                )
                # reset window
                mm["net_ms_sum"] = 0.0
                mm["recv_count"] = 0
                mm["infer_ms_sum"] = 0.0
                mm["infer_count"] = 0
                mm["last_report_ts"] = now_sec

    def _merge_packet(
        self, user_id: str, packet: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        把 pose/object/frame 三種格式合併成單一幀：
        - 當 type="frame"：直接入 buffer
        - 當 type="pose" ：寫入 persons[0] 的 bbox、keypoints；記錄 image_size；看到 pose 即可入 buffer
        - 當 type="object"：只補 detections，等待 pose 再入 buffer
        回傳：若可入 buffer 就回傳合併後的單幀 dict；否則 None
        """
        t = packet.get("type")
        fid = packet.get("frame_id")
        ts = packet.get("timestamp_ms") or packet.get("ts_ms")

        if t == "frame":
            persons = packet.get("persons") or []
            # [修改] 預設拿第一個，但保留原始列表供 Tracker 使用
            first = persons[0] if persons else {}
            bbox = first.get("bbox")
            kps  = first.get("keypoints") or first.get("kps") or []
            img_size = packet.get("image_size") or {}
            dets = packet.get("detections") or []
            return {
                "type": "frame",
                "frame_id": fid,
                "ts_ms": ts,
                "raw_persons": persons,  # <--- [新增] 保留原始列表
                "bbox": bbox,            # 預設值
                "kps": kps,              # 預設值
                "detections": dets,
                "img_w": (img_size.get("width")  or packet.get("img_w") or 640),
                "img_h": (img_size.get("height") or packet.get("img_h") or 480),
            }

        # 拆包：用 pending 先收
        pend = self._pending[user_id].setdefault(fid, {"frame_id": fid, "ts_ms": ts})
        if isinstance(packet.get("image_size"), dict):
            pend["img_w"] = packet["image_size"].get("width", pend.get("img_w", 640))
            pend["img_h"] = packet["image_size"].get("height", pend.get("img_h", 480))

        if t == "pose":
            persons = packet.get("persons") or []
            # [修改] 將 persons 存入 pending
            pend["raw_persons"] = persons 
            
            first = persons[0] if persons else {}
            if "bbox" in first:
                pend["bbox"] = first["bbox"]
            if "keypoints" in first or "kps" in first:
                pend["kps"] = first.get("keypoints") or first.get("kps") or []
            
            # policy：有 pose 即可入 buffer
            ready = ("raw_persons" in pend) or ("kps" in pend)
            return pend.copy() if ready else None

        if t == "object":
            dets = packet.get("detections") or []
            pend["detections"] = dets
            return None

        # 其他未知格式，嘗試直接解析
        if "kps" in packet or "keypoints" in packet or "persons" in packet:
            persons = packet.get("persons") or []
            first = persons[0] if persons else {}
            return {
                "type": packet.get("type"),
                "frame_id": fid, "ts_ms": ts,
                "raw_persons": persons, # [新增]
                "bbox": packet.get("bbox", first.get("bbox") if first else None),
                "kps":  packet.get("kps",  first.get("keypoints") if first else None),
                "img_w": packet.get("img_w", 640), "img_h": packet.get("img_h", 480),
                "detections": packet.get("detections", []),
            }
        return None

    async def _run_two_stage(self, user_id: str, clip_bin: List[Dict[str, Any]]):
        async with self._locks[user_id]:
            # Prepare clip5 for webhook/state events (latest 5 frames)
            _clip5 = clip_bin[-5:] if len(clip_bin) >= 5 else clip_bin
            clip_meta = {
                "start": _clip5[0] if _clip5 else (clip_bin[0] if clip_bin else None),
                "end": _clip5[-1] if _clip5 else (clip_bin[-1] if clip_bin else None),
                "size": len(_clip5),
                "win": {"window": MULTI_WINDOW, "stride": MULTI_STRIDE},
            }
            t0_total = time.perf_counter()

            # head/tail（用於日誌）
            head_seq = clip_bin[0].get("frame_seq") if clip_bin else None
            tail_seq = clip_bin[-1].get("frame_seq") if clip_bin else None

            parsed_bin = []
            for rec in clip_bin:
                bbox, kps, img_w, img_h = _extract_basic_frame(rec)
                parsed_bin.append((bbox, kps, [], img_w, img_h))

            # ===== Binary 計時 =====
            t0_bin = time.perf_counter()
            bin_out = self.bin_stage.infer(parsed_bin)
            t1_bin = time.perf_counter()
            bin_ms = (t1_bin - t0_bin) * 1000.0
            bin_pred = bin_out["pred"]
            bin_probs = bin_out["probs"]
            class_names_bin = self.bin_stage.class_names
            fall_idx = class_names_bin.index("fall") if "fall" in class_names_bin else 1
            fall_score = float(bin_probs[fall_idx])

            live_tracker = self._trackers[user_id]
            live_bbox = live_tracker.last_bbox

            result_dict = {
                "type": "inference",
                "stage": "binary",
                # [新增] 這裡放入最新的追蹤框
                "track_info": {
                    "bbox": live_bbox,  # [x1, y1, x2, y2]
                    "ts": time.time()   # 可選：後端時間戳
                },
                "binary": {
                    "class_names": class_names_bin,
                    "probs": bin_out["probs"],
                    "pred_idx": bin_out["pred_idx"],
                    "pred": bin_out["pred"],
                    "thr": FALL_START_THR,
                }
            }

            # Multi 推論（若不是跌倒則進行多類別推論）
            multi_ms = 0.0
            if bin_pred != "fall":
                tail_seq_cur = clip_bin[-1].get("frame_seq", None)
                if (
                    tail_seq_cur is None
                    or tail_seq_cur != self._last_multi_tail_seq[user_id]
                ):
                    parsed_mul = parsed_bin  # 同一窗即可
                    t0_mul = time.perf_counter()
                    mul_out = self.mul_stage.infer(parsed_mul)
                    t1_mul = time.perf_counter()
                    multi_ms = (t1_mul - t0_mul) * 1000.0
                    result_dict["stage"] = "multi"
                    result_dict["multi"] = {
                        "class_names": self.mul_stage.class_names,
                        "probs": mul_out["probs"],
                        "pred_idx": mul_out["pred_idx"],
                        "pred": mul_out["pred"],
                    }
                    result_dict["event_name"] = mul_out["pred"]
                    self._last_multi_tail_seq[user_id] = tail_seq_cur

                    # ===== 多動作狀態機（以 walk 示範，可擴充）=====
                    try:
                        act_pred = mul_out["pred"]
                        act_probs = mul_out["probs"]
                        act_idx = int(mul_out["pred_idx"])
                        act_prob = float(act_probs[act_idx])

                        if act_pred in ENABLED_ACTIONS:
                            curr = self._curr_action[user_id]
                            cand = self._cand_action[user_id]
                            hits = self._cand_hits[user_id]

                            if curr is None:
                                # 尚未有穩定中的動作：累積候選
                                if act_prob >= ACTION_START_THR:
                                    if cand == act_pred:
                                        hits += 1
                                    else:
                                        cand = act_pred
                                        hits = 1
                                else:
                                    cand = None
                                    hits = 0

                                # 啟動新動作
                                if hits >= ACTION_START_HITS:
                                    start_time = _now_str()
                                    self._curr_action[user_id] = cand
                                    self._action_start_time[user_id] = start_time
                                    self._action_peak[user_id] = act_prob
                                    self._cand_action[user_id] = None
                                    self._cand_hits[user_id] = 0
                                    if self.handlers.get("on_state_event_start"):
                                        self._spawn(
                                            self.handlers["on_state_event_start"](
                                                user_id=user_id,
                                                event_name=cand,
                                                start_time=start_time,
                                                peak_score=float(act_prob),
                                                prev_action_name="none",
                                                curr_action_name=cand,
                                                clip=clip_meta,
                                                payload=result_dict,
                                            ),
                                            tag=f"state_start:{user_id}:{cand}",
                                        )
                                else:
                                    self._cand_action[user_id] = cand
                                    self._cand_hits[user_id] = hits

                            else:
                                # 目前已有動作 curr
                                if act_pred == curr:
                                    # 同一動作持續：更新峰值，清空候選
                                    if act_prob > self._action_peak[user_id]:
                                        self._action_peak[user_id] = act_prob
                                    self._cand_action[user_id] = None
                                    self._cand_hits[user_id] = 0
                                else:
                                    # 嘗試切換：新動作達門檻且連續命中，才讓舊動作 recover & 新動作 start
                                    if act_prob >= ACTION_START_THR:
                                        if cand == act_pred:
                                            hits += 1
                                        else:
                                            cand = act_pred
                                            hits = 1
                                    else:
                                        cand = None
                                        hits = 0

                                    if hits >= ACTION_START_HITS:
                                        # 先 recover 舊動作
                                        prev = curr
                                        prev_start = self._action_start_time.get(
                                            user_id
                                        )
                                        prev_peak = float(
                                            self._action_peak.get(user_id, 0.0)
                                        )
                                        end_time = _now_str()
                                        tasks = []
                                        if self.handlers.get("on_state_event_recover"):
                                            tasks.append(
                                                self.handlers["on_state_event_recover"](
                                                    user_id=user_id,
                                                    event_name=prev,
                                                    start_time=prev_start,
                                                    end_time=end_time,
                                                    peak_score=prev_peak,
                                                    prev_action_name=prev,
                                                    curr_action_name=cand,
                                                    clip=clip_meta,
                                                    payload=result_dict,
                                                )
                                            )
                                        # 再 start 新動作
                                        self._curr_action[user_id] = cand
                                        self._action_start_time[user_id] = end_time
                                        self._action_peak[user_id] = act_prob
                                        self._cand_action[user_id] = None
                                        self._cand_hits[user_id] = 0
                                        if self.handlers.get("on_state_event_start"):
                                            tasks.append(
                                                self.handlers["on_state_event_start"](
                                                    user_id=user_id,
                                                    event_name=cand,
                                                    start_time=end_time,
                                                    peak_score=float(act_prob),
                                                    prev_action_name=prev,
                                                    curr_action_name=cand,
                                                    clip=clip_meta,
                                                    payload=result_dict,
                                                )
                                            )
                                        if tasks:
                                            self._spawn_chain(
                                                tasks,
                                                tag=f"state_switch:{user_id}:{prev}->{cand}",
                                            )
                                    else:
                                        self._cand_action[user_id] = cand
                                        self._cand_hits[user_id] = hits
                    except Exception as _e:
                        print(f"[STREAM][STATE_ACTION][ERROR] user={user_id} err={_e}")

            # 取本次視窗的頭尾 frame_id 與單調序號 frame_seq（由 ingest() 填好）
            head_id = clip_bin[0].get("frame_id") if clip_bin else None
            tail_id = clip_bin[-1].get("frame_id") if clip_bin else None
            head_seq = clip_bin[0].get("frame_seq") if clip_bin else None
            tail_seq = clip_bin[-1].get("frame_seq") if clip_bin else None

            print(
                f"user={user_id},frames={head_id}-{tail_id},frames_seq={head_seq}-{tail_seq},predict={result_dict}"
            )

            # 不阻塞推論鎖：送出結果改為背景執行
            self._spawn(ws_manager.send(user_id, result_dict), tag=f"ws_send:{user_id}")

            # ===== 跌倒狀態機 =====
            in_fall = self._in_fall[user_id]
            start_hits = self._start_hits[user_id]
            recover_hits = self._recover_hits[user_id]
            peak_score = self._fall_peak[user_id]

            if in_fall and fall_score > peak_score:
                peak_score = fall_score
                self._fall_peak[user_id] = peak_score

            # 處理 fall_start 事件
            if not in_fall:
                if bin_pred == "fall" and fall_score >= FALL_START_THR:
                    start_hits += 1
                else:
                    start_hits = 0
                self._start_hits[user_id] = start_hits

                if start_hits >= FALL_START_HITS:
                    self._in_fall[user_id] = True
                    self._recover_hits[user_id] = 0
                    self._fall_peak[user_id] = fall_score
                    start_time = _now_str()
                    self._fall_start_ts[user_id] = start_time

                    # 供 webhook 使用(改自clip5)
                    _clip = clip_bin
                    clip_meta = {
                        "start": _clip[0] if _clip else clip_bin[0],
                        "end": _clip[-1] if _clip else clip_bin[-1],
                        "size": len(_clip),
                        "win": {"window": BIN_WINDOW, "stride": BIN_STRIDE},
                    }
                    curr_act = self._curr_action.get(user_id)
                if self.handlers.get("on_fall_start"):
                    self._spawn(
                        self.handlers["on_fall_start"](
                            user_id=user_id,
                            start_time=start_time,
                            result=result_dict,
                            prev_action_name=curr_act,
                            curr_action_name="fall",
                            clip=clip_meta,
                        ),
                        tag=f"fall_start:{user_id}",
                    )
                    # 若此時有正在進行的動作（如 walk），先把它 recover，curr 指向 fall
                    try:
                        curr_act = self._curr_action.get(user_id)
                        if curr_act:
                            prev_start = self._action_start_time.get(user_id)
                            prev_peak = float(self._action_peak.get(user_id, 0.0))
                            if self.handlers.get("on_state_event_recover"):
                                self._spawn(
                                    self.handlers["on_state_event_recover"](
                                        user_id=user_id,
                                        event_name=curr_act,
                                        start_time=prev_start,
                                        end_time=start_time,
                                        peak_score=prev_peak,
                                        prev_action_name=curr_act,
                                        curr_action_name="fall",
                                        payload=result_dict,
                                    ),
                                    tag=f"state_recover_on_fall:{user_id}:{curr_act}",
                                )
                            # 清掉動作狀態
                            self._curr_action[user_id] = None
                            self._action_start_time.pop(user_id, None)
                            self._action_peak[user_id] = 0.0
                            self._cand_action[user_id] = None
                            self._cand_hits[user_id] = 0
                    except Exception as _e:
                        print(f"[STREAM][STATE_ACTION][ERROR] user={user_id} err={_e}")

            # 處理 fall_recover 事件
            else:
                if fall_score <= FALL_RECOVER_THR:
                    recover_hits += 1
                else:
                    recover_hits = 0
                self._recover_hits[user_id] = recover_hits

                if recover_hits >= FALL_RECOVER_HITS:
                    self._in_fall[user_id] = False
                    self._start_hits[user_id] = 0
                    self._recover_hits[user_id] = 0

                    start_time = self._fall_start_ts.get(user_id)
                    end_time = _now_str()
                    peak = self._fall_peak[user_id]

                    if self.handlers.get("on_fall_recover"):
                        self._spawn(
                            self.handlers["on_fall_recover"](
                                user_id=user_id,
                                start_time=start_time,
                                end_time=end_time,
                                peak_score=float(peak) if peak is not None else None,
                                result=result_dict,
                                score=float(fall_score),
                                reason="below_recover_threshold",
                            ),
                            tag=f"fall_recover:{user_id}",
                        )

                    self._fall_start_ts.pop(user_id, None)
                    self._fall_peak[user_id] = 0.0

            self.state_bin_pred[user_id] = bin_out["pred"]

            t1_total = time.perf_counter()
            total_ms = (t1_total - t0_total) * 1000.0

            m = self._metrics[user_id]
            m["infer_ms_sum"] += total_ms
            m["infer_count"] += 1

            budget = m.get("budget_ms")
            budget_str = f"{budget:.0f}ms" if budget else "n/a"

            status = "ok"

            print(
                f"[PERF] user={user_id} frames_seq={head_seq}-{tail_seq} "
                f"bin={bin_ms:.1f}ms multi={multi_ms:.1f}ms total={total_ms:.1f}ms "
                f"budget≈{budget_str} status={status}"
            )

    async def force_recover(self, user_id: str, reason: str = "manual"):
        """
        斷線或出錯時呼叫，清除該 user 狀態（以及必要時觸發 recover 事件）
        """
        # 若有正在進行的動作，也補一個 recover
        try:
            curr_act = self._curr_action.get(user_id)
            if curr_act and self.handlers.get("on_state_event_recover"):
                self._spawn(
                    self.handlers["on_state_event_recover"](
                        user_id=user_id,
                        event_name=curr_act,
                        start_time=self._action_start_time.get(user_id),
                        end_time=_now_str(),
                        peak_score=float(self._action_peak.get(user_id, 0.0)),
                        prev_action_name=curr_act,
                        curr_action_name="none",
                        payload=None,
                    ),
                    tag=f"force_recover:{user_id}:{curr_act}",
                )
        except Exception as e:
            print(f"[STREAM][HOOK][on_state_event_recover][ERROR] user={user_id} {e}")

        # 清空 fall 與動作狀態
        self._in_fall[user_id] = False
        self._start_hits[user_id] = 0
        self._recover_hits[user_id] = 0
        self._fall_start_ts.pop(user_id, None)
        self._fall_peak[user_id] = 0.0

        self._curr_action[user_id] = None
        self._action_start_time.pop(user_id, None)
        self._action_peak[user_id] = 0.0
        self._cand_action[user_id] = None
        self._cand_hits[user_id] = 0

        if user_id in self.buffers:
            try:
                del self.buffers[user_id]
            except Exception:
                pass

        # 讓下一段從新節點重新對齊 STRIDE
        self._last_infer_tail_seq[user_id] = None
        self._last_multi_tail_seq[user_id] = None
        self._next_infer_tail_seq[user_id] = None
        self._pending[user_id].clear()
        self._last_real_frame[user_id] = None

        if user_id in self._trackers:
            del self._trackers[user_id]

        self.state_bin_pred[user_id] = "non_fall"
        print(f"[STREAM] force_recover user={user_id} reason={reason}")


# === Singleton ===
stream_infer_manager = StreamInferManager()
