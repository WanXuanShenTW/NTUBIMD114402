# app/utils/stream_infer_manager.py
# -*- coding: utf-8 -*-
"""
Stream Inference Manager — WS → 聚合(frame_id) → 前處理 → 二階段 CNN+LSTM → 事件/回傳
重點：
- 先把 WS 拆包的 pose/object 以 frame_id 聚合成一個完整 frame（含 bbox、kps、detections、image_size）
- 與 loader 對齊的 RelationMap / Motion / Mask；自動對齊 in_channels（骨架線 + 物件通道數）
- 二階段：Binary(10x5) → Multi(10x5)
- 帶遲滯 + 連續命中 的跌倒開始/恢復狀態機
- ✅ 針對同一 user 的推論採用 asyncio.Lock 串行化，避免並行造成重複 start 或 recover hits 被洗掉
"""

import os
import asyncio
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict, deque
from datetime import datetime

import numpy as np
import torch

# 依照專案結構（utils 目錄）引入兩個 loader
from ..utils import binary_cnn_lstm_loader as bl
from ..utils import multi_cnn_lstm_loader as ml

from ..utils.ws_connection_manager import ws_manager


def _now_str() -> str:
    try:
        from ..utils.response_util import now_str
        return now_str()
    except Exception:
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


# ====== 模型路徑與視窗設定（可用環境變數覆寫） ======
BIN_MODEL_PATH     = os.getenv("SC_BIN_MODEL_PATH",     "models/binary/best.pt")
BIN_CLASSES_PATH   = os.getenv("SC_BIN_CLASSES_PATH",   "models/binary/classes.json")
MULTI_MODEL_PATH   = os.getenv("SC_MULTI_MODEL_PATH",   "models/multi/best.pt")
MULTI_CLASSES_PATH = os.getenv("SC_MULTI_CLASSES_PATH", "models/multi/classes.json")

BIN_WINDOW         = int(getattr(bl, "WINDOW", 10))
BIN_STRIDE         = int(getattr(bl, "STRIDE", 5))
MULTI_WINDOW       = int(getattr(ml, "WINDOW", 10))   # 二階段都用 10
MULTI_STRIDE       = int(getattr(ml, "STRIDE", 5))

# ====== 事件觸發/恢復參數（可用環境變數覆寫） ======
FALL_START_THR     = float(os.getenv("SC_FALL_START_THR", getattr(bl, "DECISION_THR", 0.60)))
FALL_RECOVER_THR   = float(os.getenv("SC_FALL_RECOVER_THR", 0.45))
FALL_START_HITS    = int(os.getenv("SC_FALL_START_HITS", "2"))
FALL_RECOVER_HITS  = int(os.getenv("SC_FALL_RECOVER_HITS", "2"))


def _torch_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _calc_in_channels(include_bone: bool, num_obj: int, num_edges: int) -> int:
    """C = 1(bbox_mask) + 1(dist) + 17(kps) + Edges + num_obj + 2(coord)"""
    e = (num_edges if include_bone else 0)
    return 1 + 1 + 17 + e + num_obj + 2


def _safe_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
    logits = logits - logits.max(dim=dim, keepdim=True).values
    return torch.softmax(logits, dim=dim)


def _extract_basic_frame(rec: Dict[str, Any]) -> Tuple[Optional[list], List[dict], float, float]:
    """
    將一幀資料轉成 (bbox_xyxy, kps[{x,y,conf}], img_w, img_h)，容忍多種來源：
    - 支援合包：type=="frame" 且 root 有 persons[0], detections, image_size
    - 支援拆包後合併：我們上層 ingest 已先聚合為扁平欄位：bbox, kps, detections, img_w/img_h
    - 備援：root 即有 bbox/keypoints 的情況
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
        if all(k in bbox for k in ("x","y","w","h")):
            x,y,w,h = float(bbox["x"]), float(bbox["y"]), float(bbox["w"]), float(bbox["h"])
            bbox = [x, y, x+w, y+h]
        elif all(k in bbox for k in ("cx","cy","w","h")):
            cx,cy,w,h = float(bbox["cx"]), float(bbox["cy"]), float(bbox["w"]), float(bbox["h"])
            bbox = [cx-w/2, cy-h/2, cx+w/2, cy+h/2]
        elif all(k in bbox for k in ("x1","y1","x2","y2")):
            bbox = [float(bbox["x1"]), float(bbox["y1"]), float(bbox["x2"]), float(bbox["y2"])]
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
                x = float(flat[3*i+0]); y = float(flat[3*i+1]); c = float(flat[3*i+2])
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
                x = float(p[0]); y = float(p[1]); c = float(p[2]) if len(p) > 2 else 1.0
                out_kps.append({"x": x, "y": y, "conf": c})

    return bbox, out_kps, img_w, img_h


# ====== 簡易前處理（本檔內實作，避免依賴 loader 內部工具） ======
class RelationMapConfig:
    def __init__(self, H: int, W: int, sigma_kp: float = 2.0, kp_conf_th: float = 0.2,
                 include_bone_lines: bool = False, object_classes: Optional[list] = None):
        self.H = int(H); self.W = int(W)
        self.sigma_kp = float(sigma_kp)
        self.kp_conf_th = float(kp_conf_th)
        self.include_bone_lines = bool(include_bone_lines)
        self.object_classes = list(object_classes or [])

def _gauss2d(h, w, cx, cy, sigma):
    yy, xx = np.mgrid[0:h, 0:w]
    return np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma * sigma))

def _rasterize_frame_simple(bbox, kps: list, dets: list, img_w: float, img_h: float, cfg: RelationMapConfig,
                            num_edges: int, want_in_ch: int) -> np.ndarray:
    H, W = cfg.H, cfg.W
    chans = []

    # ch0: bbox mask
    m = np.zeros((H, W), dtype=np.float32)
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        x1, y1, x2, y2 = bbox[:4]
        xs = int(max(0, min(W-1, round(x1 / max(1.0, img_w) * W))))
        xe = int(max(0, min(W-1, round(x2 / max(1.0, img_w) * W))))
        ys = int(max(0, min(H-1, round(y1 / max(1.0, img_h) * H))))
        ye = int(max(0, min(H-1, round(y2 / max(1.0, img_h) * H))))
        if xe > xs and ye > ys:
            m[ys:ye, xs:xe] = 1.0
    chans.append(m)

    # ch1: distance-to-centroid
    if kps:
        xs = [p["x"] for p in kps if p.get("conf", 0.0) >= cfg.kp_conf_th]
        ys = [p["y"] for p in kps if p.get("conf", 0.0) >= cfg.kp_conf_th]
        if xs and ys:
            cx = float(sum(xs)/len(xs)); cy = float(sum(ys)/len(ys))
            cxg = cx / max(1.0, img_w) * W
            cyg = cy / max(1.0, img_h) * H
            dmap = _gauss2d(H, W, cxg, cyg, sigma=max(1.0, cfg.sigma_kp*1.5))
            dmap = dmap / (dmap.max() + 1e-6)
        else:
            dmap = np.zeros((H, W), dtype=np.float32)
    else:
        dmap = np.zeros((H, W), dtype=np.float32)
    chans.append(dmap.astype(np.float32))

    # 17 x keypoint heatmaps
    for i in range(17):
        if i < len(kps) and kps[i].get("conf", 0.0) >= cfg.kp_conf_th:
            x = kps[i]["x"] / max(1.0, img_w) * W
            y = kps[i]["y"] / max(1.0, img_h) * H
            hm = _gauss2d(H, W, x, y, sigma=cfg.sigma_kp)
            hm = hm / (hm.max() + 1e-6)
        else:
            hm = np.zeros((H, W), dtype=np.float32)
        chans.append(hm.astype(np.float32))

    # edges channels（僅保型，填 0）
    for _ in range(int(num_edges)):
        chans.append(np.zeros((H, W), dtype=np.float32))

    # object channels（僅保型，填 0）
    for _ in range(len(getattr(cfg, "object_classes", []) or [])):
        chans.append(np.zeros((H, W), dtype=np.float32))

    # 2 x coord grids
    yy, xx = np.mgrid[0:H, 0:W]
    chans.append((xx / max(1.0, W-1)).astype(np.float32))
    chans.append((yy / max(1.0, H-1)).astype(np.float32))

    x = np.stack(chans, axis=0)  # (C,H,W)
    # 若通道少於想要的 in_ch，後面補 0；若多於，裁切（保證形狀對齊權重）
    if x.shape[0] < want_in_ch:
        pad = np.zeros((want_in_ch - x.shape[0], H, W), dtype=np.float32)
        x = np.concatenate([x, pad], axis=0)
    elif x.shape[0] > want_in_ch:
        x = x[:want_in_ch]

    return x

def _compute_motion_and_mask_simple(parsed_window: List[Tuple[dict, list, list, float, float]],
                                    motion_dim: int, kp_conf_th: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    """
    回傳：
      motion_feats: (T, motion_dim)
      valid_mask:  (T,)  只要該幀有任一 kp conf>=th 則為 1，否則 0
    簡化 motion：以 keypoints centroid 的速度/加速度為主，並 zero-pad 到 motion_dim。
    """
    T = len(parsed_window)
    feats = np.zeros((T, max(1, motion_dim)), dtype=np.float32)
    mask = np.zeros((T,), dtype=np.float32)

    cents = []
    for t, (_bbox, kps, _dets, img_w, img_h) in enumerate(parsed_window):
        xs = [p["x"] for p in (kps or []) if p.get("conf", 0.0) >= kp_conf_th]
        ys = [p["y"] for p in (kps or []) if p.get("conf", 0.0) >= kp_conf_th]
        if xs and ys:
            cx = float(sum(xs)/len(xs)) / max(1.0, img_w)
            cy = float(sum(ys)/len(ys)) / max(1.0, img_h)
            cents.append((cx, cy))
            mask[t] = 1.0
        else:
            cents.append((None, None))

    prev = None
    vprev = None
    for t, (cx, cy) in enumerate(cents):
        vx = vy = ax = ay = spd = asp = 0.0
        if cx is not None and prev is not None:
            vx = cx - prev[0]
            vy = cy - prev[1]
            spd = (vx*vx + vy*vy) ** 0.5
        if cx is not None and vprev is not None:
            ax = vx - vprev[0]
            ay = vy - vprev[1]
            asp = (ax*ax + ay*ay) ** 0.5

        # 取 6 維特徵：cx, cy, vx, vy, ax, ay，再加 spd, asp（共 8），不足補 0
        vec = [cx or 0.0, cy or 0.0, vx, vy, ax, ay, spd, asp]
        vec = (vec + [0.0]*motion_dim)[:motion_dim]
        feats[t] = np.array(vec, dtype=np.float32)

        vprev = (vx, vy) if cx is not None else None
        prev  = (cx, cy) if cx is not None else prev

    return feats, mask


class _StageModels:
    """
    打包單一 stage（binary / multi）所需的設定與模型
    """
    def __init__(self, stage: str, device: torch.device):
        assert stage in ("binary", "multi")
        self.stage = stage
        self.m = bl if stage == "binary" else ml
        self.device = device

        # 前處理 & 模型結構設定
        self.H = int(getattr(self.m, "H", 64))
        self.W = int(getattr(self.m, "W", 64))
        self.INCLUDE_BONE = bool(getattr(self.m, "INCLUDE_BONE_LINES", True))
        self.OBJECT_CLASSES = list(getattr(self.m, "OBJECT_CLASSES", []))
        self.NUM_EDGES = len(getattr(self.m, "COCO_EDGES", [])) if self.INCLUDE_BONE else 0
        self.USE_MOTION = bool(getattr(self.m, "_USE_MOTION", True))
        self.MOTION_DIM = int(getattr(self.m, "_MOTION_DIM", 9))

        # 先以目前設定建一次（稍後 load_weights 會依 checkpoint in_ch 自動調整）
        in_ch = _calc_in_channels(self.INCLUDE_BONE, len(self.OBJECT_CLASSES), self.NUM_EDGES)
        lstm_h = int(getattr(self.m, "LSTM_HIDDEN", 256))
        bidir = bool(getattr(self.m, "BIDIRECTIONAL", False))
        pool = getattr(self.m, "TEMPORAL_POOL", "attn")
        dropout = float(getattr(self.m, "DROPOUT", 0.3))
        CNNLSTM = getattr(self.m, "CNNLSTM")

        self.model = CNNLSTM(
            in_ch=in_ch,
            num_classes=1,
            cnn_out=256,
            lstm_h=lstm_h,
            lstm_layers=2,
            bidirectional=bidir,
            temporal_pool=pool,
            dropout=dropout,
            motion_dim=(self.MOTION_DIM if self.USE_MOTION else 0),
        ).to(self.device)

        self.class_names: List[str] = []

    def load_weights(self, model_path: str, classes_path: Optional[str] = None):
        """
        讀取 checkpoint，並根據第一層 conv 權重的輸入通道數自動對齊
        """
        sd = torch.load(model_path, map_location=self.device)
        state = None
        class_names = None

        if isinstance(sd, dict) and "model_state" in sd:
            state = sd["model_state"]
            class_names = sd.get("class_names")
        elif isinstance(sd, dict):
            state = sd
        else:
            raise RuntimeError(f"[{self.stage}] Unsupported checkpoint format: {type(sd)}")

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

        # checkpoint 的第一層輸入通道數
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
            raise RuntimeError(f"[{self.stage}] Cannot locate first conv weight key in checkpoint")

        expected_in = int(state[first_key].shape[1])

        # 目前設定的 in_ch
        curr_in = _calc_in_channels(self.INCLUDE_BONE, len(self.OBJECT_CLASSES), self.NUM_EDGES)

        # 若不一致，嘗試反推 include_bone 與物件通道數以對齊 expected_in
        if curr_in != expected_in:
            base_obj_names = list(getattr(self.m, "OBJECT_CLASSES", []))
            for include_bone_try in (True, False):
                edges_try = len(getattr(self.m, "COCO_EDGES", [])) if include_bone_try else 0
                base_fixed = 1 + 1 + 17 + edges_try + 2  # 除了物件通道以外
                num_obj_try = expected_in - base_fixed
                if 0 <= num_obj_try <= 64:
                    self.INCLUDE_BONE = include_bone_try
                    self.NUM_EDGES = edges_try
                    if len(base_obj_names) >= num_obj_try:
                        self.OBJECT_CLASSES = base_obj_names[:num_obj_try]
                    else:
                        extra = [f"obj{i}" for i in range(len(base_obj_names), num_obj_try)]
                        self.OBJECT_CLASSES = base_obj_names + extra
                    break

        # 依對齊後的通道數重建模型
        in_ch = _calc_in_channels(self.INCLUDE_BONE, len(self.OBJECT_CLASSES), self.NUM_EDGES)
        lstm_h = int(getattr(self.m, "LSTM_HIDDEN", 256))
        bidir = bool(getattr(self.m, "BIDIRECTIONAL", False))
        pool = getattr(self.m, "TEMPORAL_POOL", "attn")
        dropout = float(getattr(self.m, "DROPOUT", 0.3))
        CNNLSTM = getattr(self.m, "CNNLSTM")

        self.model = CNNLSTM(
            in_ch=in_ch,
            num_classes=len(class_names),
            cnn_out=256,
            lstm_h=lstm_h,
            lstm_layers=2,
            bidirectional=bidir,
            temporal_pool=pool,
            dropout=dropout,
            motion_dim=(self.MOTION_DIM if self.USE_MOTION else 0),
        ).to(self.device)

        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.class_names = list(class_names)

    @torch.no_grad()
    def infer(self, parsed_window: List[Tuple[dict, list, list, float, float]]) -> Dict[str, Any]:
        """
        對一個視窗做推論（使用本檔內的簡易前處理）。
        parsed_window: list of (bbox, kps, dets, img_w, img_h)，長度 = WINDOW
        """
        # Motion + Mask（若 motion_dim=0 則輸出 None）
        motion_feats, valid_mask = _compute_motion_and_mask_simple(
            parsed_window,
            motion_dim=(self.MOTION_DIM if self.USE_MOTION else 0),
            kp_conf_th=float(getattr(self.m, "KP_CONF_TH", 0.2))  # 放寬一點降低 0 熱圖機率
        )

        # RelationMap 轉張量（以簡易版 rasterize，並確保通道數 == in_ch）
        want_in_ch = _calc_in_channels(self.INCLUDE_BONE, len(self.OBJECT_CLASSES), self.NUM_EDGES)
        cfg = RelationMapConfig(
            H=self.H, W=self.W,
            sigma_kp=float(getattr(self.m, "SIGMA_KP", 2.0)),
            kp_conf_th=float(getattr(self.m, "KP_CONF_TH", 0.2)),
            include_bone_lines=self.INCLUDE_BONE,
            object_classes=self.OBJECT_CLASSES,
        )
        frames = [
            _rasterize_frame_simple(bbox, kps, dets, img_w, img_h, cfg, self.NUM_EDGES, want_in_ch)
            for (bbox, kps, dets, img_w, img_h) in parsed_window
        ]

        x = torch.from_numpy(np.stack(frames)).unsqueeze(0).float().to(self.device)  # (1,T,C,H,W)
        M = (
            torch.from_numpy(motion_feats).unsqueeze(0).float().to(self.device)
            if (self.USE_MOTION and self.MOTION_DIM > 0) else None
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


class StreamInferManager:
    """
    伺服器端推論中樞：
      - ingest(user_id, data): 收幀 → 聚合 → 緩衝 → 觸發推論
      - set_handlers(...)    : 注入事件 hooks
      - force_recover(...)   : 中斷/關閉時清理狀態
    """
    def __init__(self):
        self.device = _torch_device()

        # 每個 user 的原始幀緩衝（至少保留 10 幀）
        self.buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max(40, MULTI_WINDOW+20)))

        # 事件處理器（由 pose_routes.py 設定）
        self.handlers: Dict[str, Any] = {}

        # 二階段模型
        self.bin_stage = _StageModels("binary", device=self.device)
        self.mul_stage = _StageModels("multi",  device=self.device)
        self._load_models()

        # 跌倒狀態機（連續命中 + 遲滯）
        self._in_fall        = defaultdict(lambda: False)  # 是否目前處於 fall 事件中
        self._start_hits     = defaultdict(int)            # 連續達 START_THR 次數
        self._recover_hits   = defaultdict(int)            # 連續低於 RECOVER_THR 次數
        self._fall_start_ts  = {}                          # 事件開始時間字串
        self._fall_peak      = defaultdict(float)          # 事件期間最高分

        # 每個使用者的推論鎖，避免同一 user 併發執行 _run_two_stage 造成競態
        self._locks = defaultdict(asyncio.Lock)

        # 去重：記住上次 multi 視窗的末幀 frame_id，避免同窗重算
        self._last_multi_tail_frame_id = defaultdict(lambda: None)

        # 保留最近一次二元分類（若其他邏輯需要）
        self.state_bin_pred: Dict[str, str] = defaultdict(lambda: "non_fall")

        # ★ 新增：拆包聚合暫存（user_id -> frame_id -> partial dict）
        self._pending: Dict[str, Dict[Any, Dict[str, Any]]] = defaultdict(dict)

    def _load_models(self):
        self.bin_stage.load_weights(BIN_MODEL_PATH, BIN_CLASSES_PATH)
        self.mul_stage.load_weights(MULTI_MODEL_PATH, MULTI_CLASSES_PATH)
        print(f"[STREAM] Loaded models. binary={len(self.bin_stage.class_names)} classes, "
              f"multi={len(self.mul_stage.class_names)} classes, device={self.device}")

    def set_handlers(
        self,
        on_fall_start=None,
        on_fall_recover=None,
        on_state_event_start=None,
        on_state_event_recover=None,
    ):
        if on_fall_start:          self.handlers["on_fall_start"] = on_fall_start
        if on_fall_recover:        self.handlers["on_fall_recover"] = on_fall_recover
        if on_state_event_start:   self.handlers["on_state_event_start"] = on_state_event_start
        if on_state_event_recover: self.handlers["on_state_event_recover"] = on_state_event_recover

    def _merge_packet(self, user_id: str, packet: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        把 pose/object/frame 三種格式合併成單一幀：
        - 當 type="pose"：寫入 persons[0] 的 bbox、keypoints；記錄 image_size；不直接入 buffer，等待 object 或直接以 pose 入 buffer（policy: 看到 pose 即入 buffer）
        - 當 type="object"：寫入 detections；若該 frame 已入 buffer 則忽略，否則等待 pose
        - 當 type="frame"：直接入 buffer
        回傳：若可入 buffer 就回傳合併後的單幀 dict；否則 None
        """
        t = packet.get("type")
        fid = packet.get("frame_id")
        ts  = packet.get("timestamp_ms") or packet.get("ts_ms")

        if t == "frame":
            # 已經是合包
            persons = packet.get("persons") or []
            first = persons[0] if persons else {}
            bbox = first.get("bbox")
            kps  = first.get("keypoints") or first.get("kps") or []
            img_size = packet.get("image_size") or {}
            dets = packet.get("detections") or []
            return {
                "type": "frame",
                "frame_id": fid,
                "ts_ms": ts,
                "bbox": bbox,
                "kps": kps,
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
            first = persons[0] if persons else {}
            if "bbox" in first:
                pend["bbox"] = first["bbox"]
            if "keypoints" in first or "kps" in first:
                pend["kps"] = first.get("keypoints") or first.get("kps") or []
            # policy：有 pose 即可入 buffer（沒有物件也可推論）
            ready = ("kps" in pend)  # bbox 可無
            return pend.copy() if ready else None

        if t == "object":
            dets = packet.get("detections") or []
            pend["detections"] = dets
            # 只有 object 不入 buffer，等 pose
            return None

        # 其他未知格式，嘗試直接解析
        if "kps" in packet or "keypoints" in packet or "persons" in packet:
            # 嘗試從 packet 抽出需要欄位
            persons = packet.get("persons") or []
            first = persons[0] if persons else {}
            bbox = packet.get("bbox", first.get("bbox") if first else None)
            kps  = packet.get("kps",  first.get("keypoints") if first else None)
            return {
                "type": packet.get("type"),
                "frame_id": fid, "ts_ms": ts,
                "bbox": bbox, "kps": kps,
                "img_w": packet.get("img_w", 640), "img_h": packet.get("img_h", 480),
                "detections": packet.get("detections", []),
            }
        return None

    async def ingest(self, user_id: str, data: Dict[str, Any]):
        """
        接收 WS 一筆（可能是 pose、object、或 frame 合包）。
        這裡會先聚合，再把「可用幀」放進 buffer。
        """
        merged = self._merge_packet(user_id, data)
        if merged is None:
            return  # 等待另一半

        # 確保只有在有 kps 時才入 buffer（避免全 0）
        _, kps, _, _ = _extract_basic_frame(merged)
        if not kps:
            return

        # 入 buffer（避免重複 frame_id）
        buf = self.buffers[user_id]
        if len(buf) == 0 or (buf[-1].get("frame_id") != merged.get("frame_id")):
            buf.append(merged)

        n = len(buf)
        if (n >= BIN_WINDOW) and ((n % BIN_STRIDE) == 0):
            try:
                clip10 = list(buf)[-BIN_WINDOW:]
                await self._run_two_stage(user_id, clip10)
            except Exception as e:
                print(f"[STREAM][ERROR] user={user_id} run_two_stage: {e}")

        # 清理已處理的 pending
        fid = merged.get("frame_id")
        self._pending[user_id].pop(fid, None)

    async def _run_two_stage(self, user_id: str, clip_bin: List[Dict[str, Any]]):
        async with self._locks[user_id]:
            parsed_bin = []
            for rec in clip_bin:
                bbox, kps, img_w, img_h = _extract_basic_frame(rec)
                parsed_bin.append((bbox, kps, [], img_w, img_h))

            # Binary 推論
            bin_out = self.bin_stage.infer(parsed_bin)
            bin_pred = bin_out["pred"]
            bin_probs = bin_out["probs"]
            class_names_bin = self.bin_stage.class_names
            fall_idx = class_names_bin.index("fall") if "fall" in class_names_bin else 1
            fall_score = float(bin_probs[fall_idx])

            result_dict = {
                "type": "inference",
                "stage": "binary",
                "binary": {
                    "class_names": class_names_bin,
                    "probs": bin_out["probs"],
                    "pred_idx": bin_out["pred_idx"],
                    "pred": bin_out["pred"],
                    "thr": FALL_START_THR,
                }
            }

            # Multi 推論（若不是跌倒則進行多類別推論）
            if bin_pred != "fall":
                tail_id = clip_bin[-1].get("frame_id", None)
                if tail_id is None or tail_id != self._last_multi_tail_frame_id[user_id]:
                    parsed_mul = []
                    for rec in clip_bin:
                        bbox, kps, img_w, img_h = _extract_basic_frame(rec)
                        parsed_mul.append((bbox, kps, [], img_w, img_h))
                    mul_out = self.mul_stage.infer(parsed_mul)
                    result_dict["stage"] = "multi"
                    result_dict["multi"] = {
                        "class_names": self.mul_stage.class_names,
                        "probs": mul_out["probs"],
                        "pred_idx": mul_out["pred_idx"],
                        "pred": mul_out["pred"],
                    }
                    self._last_multi_tail_frame_id[user_id] = tail_id

            print(f"user={user_id},predict={result_dict}")
            await ws_manager.send(user_id, result_dict)

            # 更新狀態機
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

                    clip_meta = {"start": clip_bin[0], "end": clip_bin[-1], "win": {"window": BIN_WINDOW, "stride": BIN_STRIDE}}
                    if self.handlers.get("on_fall_start"):
                        try:
                            await self.handlers["on_fall_start"](
                                user_id=user_id, start_time=start_time, result=result_dict, clip=clip_meta
                            )
                        except Exception as e:
                            print(f"[STREAM][HOOK][on_fall_start][ERROR] user={user_id} {e}")

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
                        try:
                            await self.handlers["on_fall_recover"](
                                user_id=user_id,
                                start_time=start_time,
                                end_time=end_time,
                                peak_score=float(peak) if peak is not None else None,
                                result=result_dict,
                                score=float(fall_score),
                                reason="below_recover_threshold",
                            )
                        except Exception as e:
                            print(f"[STREAM][HOOK][on_fall_recover][ERROR] user={user_id} {e}")

                    self._fall_start_ts.pop(user_id, None)
                    self._fall_peak[user_id] = 0.0

            self.state_bin_pred[user_id] = bin_out["pred"]

    async def force_recover(self, user_id: str, reason: str = "manual"):
        """
        斷線或出錯時呼叫，清除該 user 狀態（以及必要時觸發 recover 事件）
        """
        try:
            if self._in_fall.get(user_id, False):
                if self.handlers.get("on_fall_recover"):
                    await self.handlers["on_fall_recover"](
                        user_id=user_id,
                        start_time=self._fall_start_ts.get(user_id),
                        end_time=_now_str(),
                        peak_score=float(self._fall_peak.get(user_id, 0.0)),
                        result=None,
                        score=None,
                        reason=reason,
                    )
        except Exception as e:
            print(f"[STREAM][HOOK][on_fall_recover][ERROR] user={user_id} {e}")

        self._in_fall[user_id] = False
        self._start_hits[user_id] = 0
        self._recover_hits[user_id] = 0
        self._fall_start_ts.pop(user_id, None)
        self._fall_peak[user_id] = 0.0

        if user_id in self.buffers:
            try:
                del self.buffers[user_id]
            except Exception:
                pass
        self.state_bin_pred[user_id] = "non_fall"
        print(f"[STREAM] force_recover user={user_id} reason={reason}")


# === Singleton ===
stream_infer_manager = StreamInferManager()
