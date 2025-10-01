"""
Minimal binary model loader for server runtime.
- Builds CNNLSTM with the given in_ch / motion_dim / LSTM params
- Loads weights (best.pt / epXXX*.pt) with strict=False
- Resolves class_names from checkpoint or classes.json
- Returns (model.eval().to(device), class_names, stats)

This file intentionally contains NO preprocessing / KF / JSON runner.
Keep all runtime logic inside stream_infer_manager.
"""
from __future__ import annotations
import os, json
from typing import List, Tuple, Dict, Any, Optional

import torch
import torch.nn as nn

try:
    # same package (utils)
    from .cnn_lstm_utils import CNNLSTM
except Exception:
    CNNLSTM = None


def _read_class_names(weight_obj: Any, classes_path: Optional[str]) -> List[str]:
    # 1) from checkpoint
    if isinstance(weight_obj, dict):
        for k in ("class_names", "classes", "labels"):
            v = weight_obj.get(k)
            if isinstance(v, list) and all(isinstance(x, (str, int)) for x in v):
                return [str(x) for x in v]
    # 2) from classes.json
    if classes_path and os.path.isfile(classes_path):
        with open(classes_path, "r", encoding="utf-8") as f:
            return json.load(f)
    # 3) give up
    raise RuntimeError("Class names not found. Provide a checkpoint with class_names or a valid classes.json")


def build_model(*,
                in_ch: int,
                motion_dim: int = 9,
                lstm_h: int = 256,
                lstm_layers: int = 2,
                bidirectional: bool = False,
                temporal_pool: str = "attn",
                dropout: float = 0.3,
                weight_path: str,
                classes_path: Optional[str] = None,
                device: Optional[torch.device] = None,
                strict: bool = False,
               ) -> Tuple[nn.Module, List[str], Dict[str, Any]]:
    """
    Return:
      model: nn.Module (eval() + to(device))
      class_names: list[str]
      stats: dict for logging (matched/missing/unexpected, in_ch, ckpt_keys)
    """
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(weight_path, map_location=dev)

    # Pick the actual tensor state
    if isinstance(state, dict):
        if "state_dict" in state:
            tensor_state = state["state_dict"]
        elif "model_state" in state:
            tensor_state = state["model_state"]
        else:
            # assume raw state_dict
            tensor_state = state
    else:
        # could be a whole nn.Module checkpoint -> extract via state_dict()
        try:
            tensor_state = state.state_dict()
        except Exception:
            raise RuntimeError("Unsupported checkpoint format for binary model")

    class_names = _read_class_names(state if isinstance(state, dict) else {}, classes_path)
    num_classes = len(class_names)

    # Build model (prefer shared CNNLSTM definition)
    if CNNLSTM is None:
        raise RuntimeError("cnn_lstm_utils.CNNLSTM not found in utils. Please ensure it's available.")
    model = CNNLSTM(
        in_ch=in_ch,
        num_classes=num_classes,
        cnn_out=256,
        lstm_h=lstm_h,
        lstm_layers=lstm_layers,
        bidirectional=bidirectional,
        temporal_pool=temporal_pool,
        dropout=dropout,
        motion_dim=motion_dim,
    )

    # Load weights, collect stats
    incompatible = model.load_state_dict(tensor_state, strict=strict)
    # PyTorch returns IncompatibleKeys(missing_keys, unexpected_keys)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    total_in_ckpt = len(list(tensor_state.keys()))
    matched = total_in_ckpt - len(unexpected)

    stats = {
        "matched": matched,
        "missing": len(missing),
        "unexpected": len(unexpected),
        "total_in_ckpt": total_in_ckpt,
        "in_ch": in_ch,
    }

    model.eval().to(dev)
    return model, class_names, stats
