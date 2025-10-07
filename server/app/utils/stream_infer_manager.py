# app/utils/stream_infer_manager.py
# -*- coding: utf-8 -*-
"""
Stream Inference Manager（WS → 前處理 → 二階段 CNN+LSTM → 事件/回傳）
- 與你提供的 loader 完整對齊（RelationMap / Motion(9) / Mask(T,) / Kalman）
- 二階段：
  1) Binary (WINDOW=10, STRIDE=5)：fall vs non_fall ；採用 binary loader 的參數與門檻
  2) Multi  (WINDOW=20, STRIDE=5)：sit/lie/walk；僅當 binary 非跌倒時才執行
- 事件 hooks：on_fall_start / on_fall_recover / on_state_event_start / on_state_event_recover
- 支援 force_recover(user_id)：斷線或錯誤時清除 user buffer 與狀態
"""

import os
import asyncio
from collections import defaultdict, deque
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass

import numpy as np
import torch

# === 重要：引用你提供的兩個 loader（請確認路徑在 app/utils/ 下）===
# 若你的檔案實際放置於其他資料夾，請相應修改 import
from ..utils import binary_cnn_lstm_loader as bl
from ..utils import multi_cnn_lstm_loader as ml

from ..utils.ws_connection_manager import ws_manager
from .kalman_filter import KalmanFilter


# 可選：專案內若有 now_str 輔助
def _now_str():
    try:
        from ..utils.response_util import now_str
        return now_str()
    except Exception:
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


# ========== 可調整設定（亦可用環境變數覆寫） ==========
BIN_MODEL_PATH     = os.getenv("SC_BIN_MODEL_PATH",     "models/binary/best.pt")
BIN_CLASSES_PATH   = os.getenv("SC_BIN_CLASSES_PATH",   "models/binary/classes.json")

MULTI_MODEL_PATH   = os.getenv("SC_MULTI_MODEL_PATH",   "models/multi/best.pt")
MULTI_CLASSES_PATH = os.getenv("SC_MULTI_CLASSES_PATH", "models/multi/classes.json")

# 允許覆寫 multi 的「視窗長度/步幅」與 binary 分開管理
BIN_WINDOW         = getattr(bl, "WINDOW", 10)
BIN_STRIDE         = getattr(bl, "STRIDE", 5)

MULTI_WINDOW       = getattr(ml, "WINDOW", 10)
MULTI_STRIDE       = getattr(ml, "STRIDE", 5)

# 回傳/事件的 fall 決策門檻（優先採 binary loader 內設定）
if hasattr(bl, "DECISION_THR"):
    FALL_DECISION_THR = float(getattr(bl, "DECISION_THR"))
else:
    FALL_DECISION_THR = 0.65  # 後備值


# ========== 工具函式 ==========
def _torch_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _calc_in_channels(include_bone: bool, num_obj: int, num_edges: int) -> int:
    """
    C = 1(bbox_mask) + 1(dist) + 17(kps) + Edges(可0) + num_obj + 2(coord)
    """
    e = num_edges if include_bone else 0
    return 1 + 1 + 17 + e + num_obj + 2

@dataclass
class RelationMapConfig:
    H: int = 64
    W: int = 64
    sigma_kp: float = 2.0
    kp_conf_th: float = 0.4
    include_bone_lines: bool = True
    object_classes: list = None
    edges: list = None  # list of (i,j) joint index for bone lines

    def __post_init__(self):
        if self.object_classes is None:
            self.object_classes = []
        if self.edges is None:
            self.edges = []


def _gaussian_heatmap(h, w, cx, cy, sigma=2.0):
    """產生中心在 (cx,cy) 的高斯熱圖（座標已是像素值）。"""
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    return g.astype(np.float32)


def rasterize_frame(bbox, kps, dets, img_w, img_h, cfg: RelationMapConfig):
    """
    產出單幀 RelationMap：C = 1(bbox) + 1(dist佔位) + 17(kps) + |edges| + len(object_classes) + 2(coord)
    - bbox: (x1,y1,x2,y2) 或 (None,..)
    - kps: list[dict{x,y,conf/score}]（取前17點）
    - dets: 物件清單（這裡通常是 []，保留通道對齊）
    """
    H, W = int(cfg.H), int(cfg.W)
    channels = []

    # 1) bbox mask
    bbox_mask = np.zeros((H, W), dtype=np.float32)
    if bbox and all(v is not None for v in bbox):
        x1, y1, x2, y2 = bbox
        # 轉到縮放座標
        sx1 = int(max(0, min(W - 1, round((x1 / max(img_w, 1.0)) * W))))
        sy1 = int(max(0, min(H - 1, round((y1 / max(img_h, 1.0)) * H))))
        sx2 = int(max(0, min(W - 1, round((x2 / max(img_w, 1.0)) * W))))
        sy2 = int(max(0, min(H - 1, round((y2 / max(img_h, 1.0)) * H))))
        if sx2 > sx1 and sy2 > sy1:
            bbox_mask[sy1:sy2, sx1:sx2] = 1.0
    channels.append(bbox_mask)

    # 2) dist 佔位通道（如果你原本有距離場，這裡可再升級；先給 0）
    channels.append(np.zeros((H, W), dtype=np.float32))

    # 3) 17 個關鍵點熱圖
    kp_maps = []
    for i in range(17):
        if i < len(kps):
            x = float(kps[i].get("x", 0.0))
            y = float(kps[i].get("y", 0.0))
            c = float(kps[i].get("conf", kps[i].get("score", 1.0)))
        else:
            x, y, c = 0.0, 0.0, 0.0
        if c >= cfg.kp_conf_th:
            sx = (x / max(img_w, 1.0)) * W
            sy = (y / max(img_h, 1.0)) * H
            hm = _gaussian_heatmap(H, W, sx, sy, sigma=cfg.sigma_kp)
        else:
            hm = np.zeros((H, W), dtype=np.float32)
        kp_maps.append(hm)
    channels.extend(kp_maps)

    # 4) 骨架線（若有 edges 且啟用）
    if cfg.include_bone_lines and cfg.edges:
        for (i, j) in cfg.edges:
            if i < len(kps) and j < len(kps):
                xi, yi, ci = kps[i].get("x", 0.0), kps[i].get("y", 0.0), float(kps[i].get("conf", kps[i].get("score", 1.0)))
                xj, yj, cj = kps[j].get("x", 0.0), kps[j].get("y", 0.0), float(kps[j].get("conf", kps[j].get("score", 1.0)))
                if ci >= cfg.kp_conf_th and cj >= cfg.kp_conf_th:
                    # 以簡易畫線方式：在兩端蓋高斯，足夠提供空間線索
                    sx_i, sy_i = (xi / max(img_w, 1.0)) * W, (yi / max(img_h, 1.0)) * H
                    sx_j, sy_j = (xj / max(img_w, 1.0)) * W, (yj / max(img_h, 1.0)) * H
                    ch = _gaussian_heatmap(H, W, sx_i, sy_i, sigma=cfg.sigma_kp) \
                       + _gaussian_heatmap(H, W, sx_j, sy_j, sigma=cfg.sigma_kp)
                else:
                    ch = np.zeros((H, W), dtype=np.float32)
            else:
                ch = np.zeros((H, W), dtype=np.float32)
            channels.append(ch)

    # 5) 物件通道（這裡 dets 大多是空；保留長度對齊）
    for _ in (cfg.object_classes or []):
        channels.append(np.zeros((H, W), dtype=np.float32))

    # 6) 2 個座標通道
    yy, xx = np.meshgrid(np.linspace(0, 1, H, dtype=np.float32),
                         np.linspace(0, 1, W, dtype=np.float32),
                         indexing='ij')
    channels.append(xx)  # X-grid
    channels.append(yy)  # Y-grid

    return np.stack(channels, axis=0).astype(np.float32)  # (C,H,W)


def compute_motion_feats_with_mask_from_parsed_local(parsed_window, motion_dim=9, kp_conf_th=0.4):
    """
    輸出：
      motion_feats: (T, motion_dim) 這裡用簡化版運動統計特徵（平均/方差/最大速度…），對齊通道數即可
      valid_mask  : (T,)           有至少一個關鍵點達門檻就視為有效
    """
    T = len(parsed_window)
    M = np.zeros((T, motion_dim), dtype=np.float32)
    mask = np.zeros((T,), dtype=np.float32)

    # 先抽出 norm 座標序列
    coords = []
    for (bbox, kps, dets, img_w, img_h) in parsed_window:
        pts = []
        valid = False
        for i in range(17):
            if i < len(kps):
                x = float(kps[i].get("x", 0.0)) / max(img_w, 1.0)
                y = float(kps[i].get("y", 0.0)) / max(img_h, 1.0)
                c = float(kps[i].get("conf", kps[i].get("score", 0.0)))
                pts.append((x, y, c))
                if c >= kp_conf_th:
                    valid = True
            else:
                pts.append((0.0, 0.0, 0.0))
        coords.append(pts)
        mask[len(coords) - 1] = 1.0 if valid else 0.0

    # 計算每一幀的運動統計（與上一幀的差）
    for t in range(1, T):
        dxs, dys, spds = [], [], []
        for i in range(17):
            x0, y0, c0 = coords[t - 1][i]
            x1, y1, c1 = coords[t][i]
            dx, dy = (x1 - x0), (y1 - y0)
            dxs.append(dx); dys.append(dy)
            spds.append((dx * dx + dy * dy) ** 0.5)
        dxs = np.array(dxs, dtype=np.float32)
        dys = np.array(dys, dtype=np.float32)
        spds = np.array(spds, dtype=np.float32)

        feats = [
            float(np.mean(np.abs(dxs))),
            float(np.mean(np.abs(dys))),
            float(np.mean(spds)),
            float(np.std(dxs)),
            float(np.std(dys)),
            float(np.std(spds)),
            float(np.max(spds)),
            float(np.median(spds)),
            float(np.percentile(spds, 75)),
        ]
        M[t, :len(feats)] = np.asarray(feats, dtype=np.float32)

    # 第一幀沒有前幀可比，保留 0 即可；mask 已在上面計算
    return M, mask

def _safe_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # 參考 multi loader 的穩定寫法
    logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
    logits = logits - logits.max(dim=dim, keepdim=True).values
    return torch.softmax(logits, dim=dim)


def _extract_basic_frame(rec: Dict[str, Any]) -> Tuple[Optional[dict], List[dict], float, float]:
    """
    支援兩種格式：
    A) 扁平： {bbox, kps|keypoints, img_w|image_w, img_h|image_h}
    B) 行動端： {persons:[{bbox, keypoints[{x,y,conf|confidence}], ...}], image_size{width,height}}
    """
    # 解析影像尺寸
    img_w = float(
        rec.get("img_w", rec.get("image_w", rec.get("image_size", {}).get("width", 640.0))) or 640.0
    )
    img_h = float(
        rec.get("img_h", rec.get("image_h", rec.get("image_size", {}).get("height", 480.0))) or 480.0
    )

    # 如果帶有 persons 就取第一個
    persons = rec.get("persons")
    if isinstance(persons, list) and persons:
        p0 = persons[0]
        bbox = p0.get("bbox")
        raw_kps = p0.get("keypoints") or p0.get("kps") or []
        kps = [
            {"x": float(pt.get("x", 0.0)),
             "y": float(pt.get("y", 0.0)),
             "conf": float(pt.get("conf", pt.get("confidence", 1.0)))}
            for pt in raw_kps[:17] if isinstance(pt, dict)
        ]
        return bbox, kps, img_w, img_h

    # 否則走扁平
    bbox = rec.get("bbox")
    raw_kps = rec.get("kps") or rec.get("keypoints") or []
    kps = [
        {"x": float(pt.get("x", 0.0)),
         "y": float(pt.get("y", 0.0)),
         "conf": float(pt.get("conf", pt.get("confidence", 1.0)))}
        for pt in raw_kps[:17] if isinstance(pt, dict)
    ]
    return bbox, kps, img_w, img_h

class _StageModels:
    """
    打包 stage 需要的：常數、前處理函式、模型與類別
    - stage='binary' 使用 bl 模組
    - stage='multi'  使用 ml 模組
    """
    def __init__(self, stage: str, device: torch.device):
        assert stage in ("binary", "multi")
        self.stage = stage
        self.m = bl if stage == "binary" else ml  # 對應的 loader 模組
        self.device = device

        # 前處理設定
        self.H = int(getattr(self.m, "H", 64))
        self.W = int(getattr(self.m, "W", 64))
        self.INCLUDE_BONE = bool(getattr(self.m, "INCLUDE_BONE_LINES", True))
        self.OBJECT_CLASSES = list(getattr(self.m, "OBJECT_CLASSES", []))
        self.NUM_EDGES = len(getattr(self.m, "COCO_EDGES", [])) if self.INCLUDE_BONE else 0

        # 建立 nn 模型（用本模組的邊數）
        in_ch = _calc_in_channels(self.INCLUDE_BONE, len(self.OBJECT_CLASSES), self.NUM_EDGES)
        self.USE_MOTION = bool(getattr(self.m, "_USE_MOTION", True))
        self.MOTION_DIM = int(getattr(self.m, "_MOTION_DIM", 9))

        # 建立 nn 模型
        lstm_h = int(getattr(self.m, "LSTM_HIDDEN", 256))
        bidir = bool(getattr(self.m, "BIDIRECTIONAL", False))
        pool = getattr(self.m, "TEMPORAL_POOL", "attn")
        dropout = float(getattr(self.m, "DROPOUT", 0.3))
        CNNLSTM = getattr(self.m, "CNNLSTM")
        self.model = CNNLSTM(
            in_ch=in_ch,
            num_classes=1,  # 先放 1，等載入權重後再替換 head 輸出維度
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
        與 loader 一致的權重/類別讀取，並「自動對齊輸入通道數」：
          - 讀取 checkpoint 第一層卷積權重的輸入通道 expected_in
          - 根據 expected_in 反推：是否要開骨架線、需要幾個物件通道
          - 依對齊後的設定重建模型，再載入 state_dict
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

        # 類別名稱
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

        # === 讀出 checkpoint 期望的輸入通道數 expected_in ===
        #   通常鍵名會長這樣：'cnn.net.0.weight'
        first_key = None
        for k in state.keys():
            if k.endswith("cnn.net.0.weight"):
                first_key = k
                break
        if first_key is None:
            # 後備方案：找第一個 4D 權重
            for k in state.keys():
                if state[k].ndim == 4:
                    first_key = k
                    break
        if first_key is None:
            raise RuntimeError(f"[{self.stage}] Cannot locate first conv weight key in checkpoint")

        expected_in = int(state[first_key].shape[1])

        # === 目前設定下的通道數（用本模組的邊數）===
        curr_in = _calc_in_channels(self.INCLUDE_BONE, len(self.OBJECT_CLASSES), self.NUM_EDGES)

        # === 若不一致，嘗試反推 include_bone 與物件通道數，對齊 expected_in ===
        if curr_in != expected_in:
            # 嘗試 include_bone=True/False 兩種情況
            base_obj_names = list(getattr(self.m, "OBJECT_CLASSES", []))
            for include_bone_try in (True, False):
                edges_try = len(getattr(self.m, "COCO_EDGES", [])) if include_bone_try else 0
                # 固定通道（不含物件）：1(bbox)+1(dist)+17(kp)+edges_try+2(coord)
                base_fixed = 1 + 1 + 17 + edges_try + 2
                num_obj_try = expected_in - base_fixed
                if 0 <= num_obj_try <= 64:  # 給個合理上限以避免怪值
                    self.INCLUDE_BONE = include_bone_try
                    self.NUM_EDGES = edges_try
                    if len(base_obj_names) >= num_obj_try:
                        self.OBJECT_CLASSES = base_obj_names[:num_obj_try]
                    else:
                        # 不足就補名（不影響語義，只是為了通道數對齊）
                        extra = [f"obj{i}" for i in range(len(base_obj_names), num_obj_try)]
                        self.OBJECT_CLASSES = base_obj_names + extra
                    break  # 命中後結束嘗試

        # === 對齊後，用正確的 in_ch 重建模型，並載入權重 ===
        in_ch = _calc_in_channels(self.INCLUDE_BONE, len(self.OBJECT_CLASSES), self.NUM_EDGES)
        lstm_h = int(getattr(self.m, "LSTM_HIDDEN", 256))
        bidir  = bool(getattr(self.m, "BIDIRECTIONAL", False))
        pool   = getattr(self.m, "TEMPORAL_POOL", "attn")
        dropout= float(getattr(self.m, "DROPOUT", 0.3))
        CNNLSTM= getattr(self.m, "CNNLSTM")

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

        # 真的載入
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.class_names = list(class_names)

    @torch.no_grad()
    def infer(self, parsed_window: List[Tuple[dict, list, list, float, float]]) -> Dict[str, Any]:
        """
        對一個「已切好的視窗」做推論。
        parsed_window: list of (bbox, kps, dets, img_w, img_h)，長度 = WINDOW
        回傳：
          { "class_names": [], "probs": [...], "pred_idx": int, "pred": str }
        """
        # Motion + Mask（改用本檔的前處理；保持維度對齊）
        if self.USE_MOTION:
            motion_feats, valid_mask = compute_motion_feats_with_mask_from_parsed_local(
                parsed_window,
                motion_dim=self.MOTION_DIM,
                kp_conf_th=float(getattr(self.m, "KP_CONF_TH", 0.4)),
            )
        else:
            T = len(parsed_window)
            motion_feats = np.zeros((T, self.MOTION_DIM), dtype=np.float32)
            valid_mask = np.ones((T,), dtype=np.float32)

        # RelationMap 轉張量
        cfg = RelationMapConfig(
            H=self.H, W=self.W,
            sigma_kp=float(getattr(self.m, "SIGMA_KP", 2.0)),
            kp_conf_th=float(getattr(self.m, "KP_CONF_TH", 0.4)),
            include_bone_lines=self.INCLUDE_BONE,
            object_classes=self.OBJECT_CLASSES,
            edges=list(getattr(self.m, "COCO_EDGES", [])) if self.INCLUDE_BONE else [],
        )
        frames = [
            rasterize_frame(bbox, kps, dets, img_w, img_h, cfg)
            for (bbox, kps, dets, img_w, img_h) in parsed_window
        ]

        x = torch.from_numpy(np.stack(frames)).unsqueeze(0).float().to(self.device)  # (1,T,C,H,W)
        M = (
            torch.from_numpy(motion_feats).unsqueeze(0).float().to(self.device)
            if self.USE_MOTION else None
        )
        mask = torch.from_numpy(valid_mask).unsqueeze(0).float().to(self.device)

        # 數值安全（NaN/Inf→0）
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
      - ingest(user_id, data): 收幀 → 緩衝 → 觸發推論
      - set_handlers(...)    : 注入事件 hooks
      - force_recover(...)   : 中斷/關閉時清理狀態
    """
    def __init__(self):
        self.device = _torch_device()
        # 每個 user 的原始幀緩衝（保留到 40 幀，足夠 20 視窗滑動）
        self.buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max(40, MULTI_WINDOW)))
        # 事件處理器（由 pose_routes.py 設定）
        self.handlers: Dict[str, Any] = {}
        # 二階段模型
        self.bin_stage = _StageModels("binary", device=self.device)
        self.mul_stage = _StageModels("multi",  device=self.device)
        self._load_models()

        # fall 狀態
        self.state_bin_pred: Dict[str, str] = defaultdict(lambda: "non_fall")
        self.kf_bin = KalmanFilter()
        self.kf_multi = KalmanFilter()

        # ✅ 新增：frame 分組與環狀排序用
        import os
        self.WRAP_SIZE = int(os.getenv("SC_WRAP_SIZE", "60"))  # 可用環境變數改
        # 用來暫存同一 frame_id 的 pose/object（完成才進 buffer）
        self._partials: Dict[str, Dict[int, Dict[str, Any]]] = defaultdict(dict)
        # 以「完成的 frame」為單位的遞增序列（unwrap 後的線性 id）
        self._last_completed_raw: Dict[str, Optional[int]] = defaultdict(lambda: None)
        self._last_completed_lin: Dict[str, int] = defaultdict(int)
        # 注意：self.buffers 照舊，存的是「完成 frame 的原始 JSON（含 pose+object 已合併）」。

    def _load_models(self):
        # 載入 binary
        self.bin_stage.load_weights(BIN_MODEL_PATH, BIN_CLASSES_PATH)
        # 載入 multi
        self.mul_stage.load_weights(MULTI_MODEL_PATH, MULTI_CLASSES_PATH)
        print(f"[STREAM] Loaded models. binary={len(self.bin_stage.class_names)} classes, "
              f"multi={len(self.mul_stage.class_names)} classes, device={self.device}")

    def _kf_smooth(self, window_kps: list, which: str = "bin") -> list:
        """
        使用 KalmanFilter（34 維向量：17*2）對一段 window 的骨架序列做平滑。
        回傳格式：list[T][17]{x,y,conf}
        """
        kf = self.kf_bin if which == "bin" else self.kf_multi
        # 每段 window 進來前，外層會先把 state/covariance 清空
        smoothed = []
        for kps in window_kps:
            flat = []
            for i in range(17):
                if i < len(kps):
                    x = float(kps[i].get("x", 0.0))
                    y = float(kps[i].get("y", 0.0))
                else:
                    x, y = 0.0, 0.0
                flat.extend([x, y])

            state = kf.predict_and_update(flat)  # 34 維向量

            pts = []
            for i in range(17):
                x = float(state[2 * i])
                y = float(state[2 * i + 1])
                c = 1.0 if i >= len(kps) else float(kps[i].get("conf", kps[i].get("score", 1.0)))
                pts.append({"x": x, "y": y, "conf": c})
            smoothed.append(pts)
        return smoothed

    def _unwrap_frame_id(self, user_id: str, fid: int) -> int:
        """
        將環狀 frame_id（1..WRAP_SIZE）展平成線性遞增 id。
        規則：以「上一次完成的 raw fid」為錨，取往前的最小正位移。
        例如 WRAP=60，錨=59：fid=60→+1；fid=1→+2；fid=2→+3；...
        """
        wrap = max(1, int(self.WRAP_SIZE))
        prev_raw = self._last_completed_raw[user_id]
        prev_lin = self._last_completed_lin[user_id]
        if prev_raw is None:
            # 第一個完成的 frame，設錨
            self._last_completed_raw[user_id] = fid
            self._last_completed_lin[user_id] = 0
            return 0
        # 0..wrap-1 的前進位移
        delta = (fid - prev_raw) % wrap
        new_lin = prev_lin + delta
        # 更新錨
        self._last_completed_raw[user_id] = fid
        self._last_completed_lin[user_id] = new_lin
        return new_lin

    def set_handlers(
        self,
        on_fall_start=None,
        on_fall_recover=None,
        on_state_event_start=None,
        on_state_event_recover=None,
    ):
        if on_fall_start:         self.handlers["on_fall_start"] = on_fall_start
        if on_fall_recover:       self.handlers["on_fall_recover"] = on_fall_recover
        if on_state_event_start:  self.handlers["on_state_event_start"] = on_state_event_start
        if on_state_event_recover:self.handlers["on_state_event_recover"] = on_state_event_recover

    async def ingest(self, user_id: str, data: Dict[str, Any]):
        """
        接收一幀骨架 JSON（由 ws_message_dispatcher 呼叫）
        典型資料：
          {
            "frame_id": 42,
            "ts_ms": 1759508034919,
            "bbox": [x1,y1,x2,y2] 或 {x,y,w,h} 或 None,
            "kps": [{"x":..,"y":..,"conf":..}, ...],
            "img_w": 640, "img_h": 480  # 選填
          }
        """
        # 解析 frame_id 與訊息型別
        fid = int(data.get("frame_id", data.get("fid", -1)))
        msg_type = str(data.get("type", "")).lower()

        # 1) 一包就含 pose（persons）且可能也含物件（detections）的情況：直接當完成幀
        has_persons = isinstance(data.get("persons"), list) and len(data["persons"]) > 0
        has_dets    = isinstance(data.get("detections"), list)

        if has_persons:
            # 直接視為完成幀：用現有 _extract_basic_frame 取 bbox/kps，dets 暫時不進 relation map（占位）
            buf = self.buffers[user_id]
            # 以 unwrap 後的線性 id 表示時間順序
            lin = self._unwrap_frame_id(user_id, fid)
            # 同步保留原始資料，以便未來要用 dets
            buf.append({"_lin": lin, **data})

        else:
            # 2) 分開傳（pose/object）：先分組
            partials = self._partials[user_id]
            slot = partials.get(fid)
            if slot is None:
                slot = {}
                partials[fid] = slot
            if msg_type == "pose":
                slot["pose"] = data
            elif msg_type == "object":
                slot["object"] = data
            else:
                # 沒標 type，但帶 persons 就當 pose；帶 detections 就當 object
                if isinstance(data.get("persons"), list):
                    slot["pose"] = data
                elif isinstance(data.get("detections"), list):
                    slot["object"] = data

            # 湊齊了就變成完成幀
            if "pose" in slot and "object" in slot:
                buf = self.buffers[user_id]
                lin = self._unwrap_frame_id(user_id, fid)
                # 合併為一包（保留 persons/detections 給未來使用）
                merged = {
                    "type": "frame",
                    "frame_id": fid,
                    "timestamp_ms": slot["pose"].get("timestamp_ms") or slot["object"].get("timestamp_ms"),
                    "image_size": slot["pose"].get("image_size") or slot["object"].get("image_size"),
                    "persons": slot["pose"].get("persons", []),
                    "detections": slot["object"].get("detections", []),
                    "_lin": lin,
                }
                buf.append(merged)
                # 用完就刪
                try:
                    del partials[fid]
                except Exception:
                    pass

        # 3) 視窗觸發（以「完成幀數量」來判斷）
        buf = self.buffers[user_id]
        n = len(buf)
        if (n >= BIN_WINDOW) and ((n % BIN_STRIDE) == 0):
            try:
                # 取最後 BIN_WINDOW 個完成幀 → 交給 _run_two_stage
                clip10 = list(buf)[-BIN_WINDOW:]
                await self._run_two_stage(user_id, clip10)
            except Exception as e:
                print(f"[STREAM][ERROR] user={user_id} run_two_stage: {e}")


    async def _run_two_stage(self, user_id: str, clip_bin: List[Dict[str, Any]]):
        """
        - 先對 10 幀做 binary 推論
        - 若非跌倒，檢查是否有 20 幀，則再跑 multi 並一起回傳
        - 若為跌倒且分數過門檻 → 觸發 on_fall_start
        """
        # 準備 binary parsed（共用無物件通道：dets=[]；若你有物件，也可改成傳入）
        parsed_bin = []
        for rec in clip_bin:
            bbox, kps, img_w, img_h = _extract_basic_frame(rec)
            parsed_bin.append((bbox, kps, [], img_w, img_h))

        # Kalman（與 loader 對齊：採用 bl 的 KF）
        # ✅ 每段 window 開始前重置 Binary KF，避免跨片段汙染
        # Kalman（改用本檔 KF；每段進來先重置狀態，避免跨片段汙染）
        if getattr(bl, "ENABLE_KALMAN", True):
            print("Initializing Kalman filter state.")
            self.kf_bin.state = None
            self.kf_bin.covariance = None
            window_kps = [k for (_, k, _, _, _) in parsed_bin]
            smoothed = self._kf_smooth(window_kps, which="bin")
            parsed_bin = [(bbox, smoothed[i], dets, w, h) for i, (bbox, _, dets, w, h) in enumerate(parsed_bin)]

        # Binary 推論
        bin_out = self.bin_stage.infer(parsed_bin)
        bin_pred = bin_out["pred"]
        bin_probs = bin_out["probs"]
        class_names_bin = self.bin_stage.class_names

        # 估算 fall 機率 （優先找 "fall" 類別，沒有就取索引 1）
        if "fall" in class_names_bin:
            fall_idx = class_names_bin.index("fall")
        else:
            # 二元多半為 [non_fall, fall]
            fall_idx = 1 if len(class_names_bin) >= 2 else 0
        fall_score = float(bin_probs[fall_idx])

        result_dict: Dict[str, Any] = {
            "type": "inference",
            "stage": "binary",
            "binary": {
                "class_names": class_names_bin,
                "probs": bin_probs,
                "pred_idx": bin_out["pred_idx"],
                "pred": bin_pred,
                "thr": FALL_DECISION_THR,
            }
        }

        # 若為非跌倒 → 嘗試 multi（需要 20 幀）
        # 注意：這裡「不延後」等 20 幀，而是**如果當下 buffer 足夠**就一起回傳
        if bin_pred != "fall":
            buf = self.buffers[user_id]
            if len(buf) >= MULTI_WINDOW and ((len(buf) % MULTI_STRIDE) == 0):
                clip20 = list(buf)[-MULTI_WINDOW:]
                parsed_mul = []
                for rec in clip20:
                    bbox, kps, img_w, img_h = _extract_basic_frame(rec)
                    parsed_mul.append((bbox, kps, [], img_w, img_h))

                # Kalman：用 bl 的 KF（與離線相容；若要改成 ml 的也可）
                # ✅ 每段 window 開始前重置 Multi KF，避免跨片段汙染
                # Kalman（改用內建 KF 實作）
                if getattr(ml, "ENABLE_KALMAN", True):
                    self.kf_multi.state = None
                    self.kf_multi.covariance = None
                    window_kps = [k for (_, k, _, _, _) in parsed_mul]
                    half_len = getattr(ml, "HALF_LEN_OVERRIDE", None) or max(1, MULTI_WINDOW // 2)

                    # ✅ 用我們在類別內新增的 _kf_smooth（which="multi" 會走 self.kf_multi）
                    smoothed = self._kf_smooth(
                        window_kps,
                        which="multi",
                    )
                    parsed_mul = [(bbox, smoothed[i], dets, w, h) for i, (bbox, _, dets, w, h) in enumerate(parsed_mul)]

                dbg_cfg = RelationMapConfig(H=self.mul_stage.H, W=self.mul_stage.W,
                            sigma_kp=float(getattr(ml, "SIGMA_KP", 2.0)),
                            kp_conf_th=float(getattr(ml, "KP_CONF_TH", 0.4)),
                            include_bone_lines=self.mul_stage.INCLUDE_BONE,
                            object_classes=self.mul_stage.OBJECT_CLASSES,
                            edges=list(getattr(ml, "COCO_EDGES", [])) if self.mul_stage.INCLUDE_BONE else [])
                x_frames = [rasterize_frame(b, k, d, w, h, dbg_cfg) for (b, k, d, w, h) in parsed_mul]
                print(f"[DEBUG][MULTI] x_sum={float(np.stack(x_frames).sum()):.3f}, T={len(x_frames)}")
                
                mul_out = self.mul_stage.infer(parsed_mul)
                result_dict["stage"] = "multi"
                result_dict["multi"] = {
                    "class_names": self.mul_stage.class_names,
                    "probs": mul_out["probs"],
                    "pred_idx": mul_out["pred_idx"],
                    "pred": mul_out["pred"],
                }

        # 發送 WS 結果
        await ws_manager.send(user_id, result_dict)

        buf = self.buffers[user_id]
        for _ in range(BIN_STRIDE):
            if buf:
                buf.popleft()
        
        # 事件：跌倒開始（過門檻才觸發）
        if (bin_pred == "fall") and (fall_score >= FALL_DECISION_THR):
            if self.handlers.get("on_fall_start"):
                clip_meta = {
                    "start": clip_bin[0],
                    "end": clip_bin[-1],
                    "win": {"window": BIN_WINDOW, "stride": BIN_STRIDE},
                }
                try:
                    await self.handlers["on_fall_start"](
                        user_id=user_id,
                        start_time=_now_str(),
                        result=result_dict,
                        clip=clip_meta,
                    )
                except Exception as e:
                    print(f"[STREAM][HOOK][on_fall_start][ERROR] user={user_id} {e}")

        # TODO（選配）：可在這裡做簡單「狀態機」：從 fall→non_fall 時觸發 on_fall_recover
        self.state_bin_pred[user_id] = bin_pred

    async def force_recover(self, user_id: str, reason: str = "manual"):
        """
        斷線或出錯時呼叫，清除該 user 狀態（以及必要時觸發 recover 事件）
        """
        try:
            # 若當前記錄為 fall，可在此觸發 recover
            if self.state_bin_pred.get(user_id) == "fall":
                if self.handlers.get("on_fall_recover"):
                    await self.handlers["on_fall_recover"](
                        user_id=user_id,
                        start_time=None,
                        end_time=_now_str(),
                        peak_score=None,
                        result=None,
                        score=None,
                        reason=reason,
                    )
        except Exception as e:
            print(f"[STREAM][HOOK][on_fall_recover][ERROR] user={user_id} {e}")

        # 清除狀態
        if user_id in self.buffers:
            try:
                del self.buffers[user_id]
            except Exception:
                pass
        self.state_bin_pred[user_id] = "non_fall"
        print(f"[STREAM] force_recover user={user_id} reason={reason}")


# === Singleton ===
stream_infer_manager = StreamInferManager()
