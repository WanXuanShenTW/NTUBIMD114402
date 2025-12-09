
# -*- coding: utf-8 -*-
import os, sys, random, math, csv, copy
from pathlib import Path
from typing import List, Tuple

import torch
import importlib.util
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm.auto import tqdm

# --- Load user's trainer module by path ---
# 建議用相對路徑：與 scan.py 同資料夾
from pathlib import Path
TRAINER_PATH = str(Path(__file__).resolve().with_name("binary_cnn_lstm_trainer.py"))

spec = importlib.util.spec_from_file_location("user_trainer", TRAINER_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load module from {TRAINER_PATH}")
user_trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(user_trainer)

# Short aliases
Config = user_trainer.Config
PoseObjectDataset = user_trainer.PoseObjectDataset
CNNLSTM = user_trainer.CNNLSTM
evaluate = user_trainer.evaluate
COCO_EDGES = user_trainer.COCO_EDGES

# ===================== Hardcoded Settings =====================
# 簡單開關：選擇要跑「兩個都做」或「只做 oversample」
# 可選："both" 或 "oversample"
RUN_MODE = "both"

SEED = 42
EPOCHS = 3
DO_DOWNSAMPLE_SCAN = True
DO_OVERSAMPLE_SCAN = True

# non_fall 下採樣比例（保留比例）
DOWN_RATIOS = [1.0, 0.5, 0.25, 0.125]

# fall 過採樣倍數（重複倍數）
POS_MULTS = [1, 2, 3, 4]

# 固定訓練設定（可在這邊改）
BATCH_SIZE = None        # 若為 None 則用 Config 預設
LR = None                # 若為 None 則用 Config 預設
USE_SAMPLER = False      # 關閉 WeightedRandomSampler，避免與掃描設計衝突
LOSS = None              # 'ce' 或 'focal'；None 則沿用 Config

OUT_CSV = "outputs/models/test/balance_probe.csv"

# 依 RUN_MODE 覆寫執行內容
# "oversample"：只做 oversample；"both"：兩個都做
_rm = RUN_MODE.strip().lower()
if _rm == "oversample":
    DO_DOWNSAMPLE_SCAN = False
    DO_OVERSAMPLE_SCAN = True
elif _rm == "both":
    DO_DOWNSAMPLE_SCAN = True
    DO_OVERSAMPLE_SCAN = True
else:
    raise ValueError("RUN_MODE must be 'both' or 'oversample'")
# =============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def _split_indices(n: int, split_ratio: float = 0.8):
    idx = list(range(n))
    random.shuffle(idx)
    s = int(n * split_ratio)
    return idx[:s], idx[s:]

def _labels_from_indices(ds: PoseObjectDataset, idxs: List[int]) -> List[int]:
    return [ds.windows[i][0] for i in idxs]

def _build_dataloaders(cfg_tr: Config, cfg_va: Config,
                       idx_tr: List[int], idx_va: List[int],
                       batch_size: int):
    # Two separate datasets so we can turn off aug on val by cfg
    ds_tr = PoseObjectDataset(cfg_tr)
    ds_va = PoseObjectDataset(cfg_va)
    tr_subset = Subset(ds_tr, idx_tr)
    va_subset = Subset(ds_va, idx_va)

    # Build loaders without weighted sampler
    dl_tr = DataLoader(tr_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    dl_va = DataLoader(va_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return ds_tr, ds_va, dl_tr, dl_va

def _build_model_and_loss(cfg: Config, ds, device):
    # in_ch 對齊 rasterizer
    in_ch = 2 + 17 + (len(COCO_EDGES) if cfg.include_bone_lines else 0) \
            + (len(ds.rm_cfg.object_classes) if cfg.use_objects else 0) + 2
    model = CNNLSTM(in_ch=in_ch, num_classes=len(ds.class_names), cnn_out=user_trainer._CNN_OUT,
                    lstm_h=cfg.lstm_hidden, lstm_layers=user_trainer._LSTM_LAYERS,
                    bidirectional=bool(cfg.bidirectional),
                    temporal_pool=cfg.temporal_pool, dropout=cfg.dropout, motion_dim=9).to(device)

    # Loss（若 focal，使用訓練集分佈計 alpha）
    counts = [0]*len(ds.class_names)
    # 注意：這裡用 ds.windows（訓練集）索引分佈估計
    for y,_,_,_ in ds.windows:
        counts[y] += 1
    total = float(sum(max(1,c) for c in counts))
    alpha = torch.tensor([total/max(1,c) for c in counts], dtype=torch.float32)
    alpha = (alpha/alpha.sum()).to(device)

    loss_name = cfg.loss
    criterion = torch.nn.CrossEntropyLoss() if loss_name=="ce" else user_trainer.FocalLoss(alpha=alpha, gamma=user_trainer._FOCAL_GAMMA)
    return model, criterion

def _train_one(cfg_tr: Config, cfg_va: Config,
               idx_tr: List[int], idx_va: List[int],
               epochs: int, *, label_note: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 準備 dataloader
    batch_size = BATCH_SIZE if BATCH_SIZE else cfg_tr.batch_size
    ds_tr, ds_va, dl_tr, dl_va = _build_dataloaders(cfg_tr, cfg_va, idx_tr, idx_va, batch_size)

    # 模型與 Loss
    model, criterion = _build_model_and_loss(cfg_tr, ds_tr, device)
    opt = torch.optim.AdamW(model.parameters(), lr=(LR if LR else cfg_tr.lr), weight_decay=user_trainer._WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=user_trainer._AMP)

    # 簡化的訓練 loop（與原版一致的張量介面與 evaluate）
    for ep in range(1, epochs+1):
        model.train(); tot=0; corr=0; loss_sum=0.0
        pbar = tqdm(dl_tr, total=len(dl_tr), desc=f"[{label_note}] Epoch {ep}/{epochs}", leave=False)
        for (X,M,mask), y in pbar:
            X=X.to(device); M=M.to(device); mask=mask.to(device); y=y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda', enabled=scaler.is_enabled()):
                out = model(X, motion=M, mask=mask)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            if user_trainer._GRAD_CLIP and user_trainer._GRAD_CLIP>0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), user_trainer._GRAD_CLIP)
            scaler.step(opt); scaler.update()
            loss_sum += float(loss.item()) * len(y)
            corr += (out.argmax(1) == y).sum().item(); tot += len(y)
            cur_loss = loss_sum/max(1,tot); cur_acc = corr/max(1,tot)
            pbar.set_postfix(loss=f"{cur_loss:.4f}", acc=f"{cur_acc:.3f}")
        tr_loss = loss_sum/max(1,tot); tr_acc = corr/max(1,tot)

        va_acc, va_mf1 = evaluate(model, dl_va, device, use_motion=True)
        print(f"[{label_note}] Epoch {ep}/{epochs} | train_loss={tr_loss:.4f} train_acc={tr_acc:.3f} | val_acc={va_acc:.3f} val_mf1={va_mf1:.3f}")
    return va_acc, va_mf1, len(idx_tr)

def _clone_cfg(base: Config):
    # deep copy Config instance（Config 裡多為基本型別，可直接手動複製）
    c = Config()
    for k,v in base.__dict__.items():
        setattr(c, k, copy.deepcopy(v))
    return c

def _prepare_base_cfg():
    cfg = Config()
    # 覆蓋共用設定
    if BATCH_SIZE: cfg.batch_size = BATCH_SIZE
    if LR: cfg.lr = LR
    if LOSS: cfg.loss = LOSS
    cfg.use_sampler = USE_SAMPLER  # 關掉 weighted sampler
    # 安全：val 不做擴增（用獨立 cfg_va 再關）
    return cfg

def _make_indices_by_labels(ds: PoseObjectDataset):
    pos, neg = [], []
    for i,(y,_,_,_) in enumerate(ds.windows):
        (pos if y==1 else neg).append(i)
    return neg, pos

def run_downsample_scan():
    print("==== Downsample non_fall scan ====")
    cfg_base = _prepare_base_cfg()
    set_seed(SEED)
    # 用訓練分割的索引（先在一個 Dataset 上做，再複用到 tr/va cfg）
    ds_tmp = PoseObjectDataset(cfg_base)
    n = len(ds_tmp)
    idx_tr, idx_va = _split_indices(n, 0.8)
    # 取得訓練集的正負索引
    neg_all, pos_all = [], []
    for i in idx_tr:
        y = ds_tmp.windows[i][0]
        (neg_all if y==0 else pos_all).append(i)

    rows = []
    for r in tqdm(DOWN_RATIOS, desc="Downsample scan"):
        keep = max(1, int(len(neg_all) * float(r)))
        random.shuffle(neg_all)
        idx_tr_r = pos_all + neg_all[:keep]
        random.shuffle(idx_tr_r)

        # 建立兩份 cf g：訓練打開擴增；驗證關閉擴增
        cfg_tr = _clone_cfg(cfg_base); cfg_tr.aug_for_rare = True
        cfg_va = _clone_cfg(cfg_base); cfg_va.aug_for_rare = False

        note = f"downsample={r}"
        va_acc, va_mf1, ntr = _train_one(cfg_tr, cfg_va, idx_tr_r, idx_va, EPOCHS, label_note=note)
        rows.append(("downsample", r, ntr, va_acc, va_mf1))
    return rows

def run_oversample_scan():
    print("==== Oversample fall scan ====")
    cfg_base = _prepare_base_cfg()
    set_seed(SEED)
    ds_tmp = PoseObjectDataset(cfg_base)
    n = len(ds_tmp)
    idx_tr, idx_va = _split_indices(n, 0.8)

    # 取得訓練集的正負索引
    neg_all, pos_all = [], []
    for i in idx_tr:
        y = ds_tmp.windows[i][0]
        (neg_all if y==0 else pos_all).append(i)

    rows = []
    for m in tqdm(POS_MULTS, desc="Oversample scan"):
        idx_tr_m = neg_all + (pos_all * int(m))  # 簡單重複
        random.shuffle(idx_tr_m)

        cfg_tr = _clone_cfg(cfg_base); cfg_tr.aug_for_rare = True
        cfg_va = _clone_cfg(cfg_base); cfg_va.aug_for_rare = False

        note = f"oversample_pos={m}x"
        va_acc, va_mf1, ntr = _train_one(cfg_tr, cfg_va, idx_tr_m, idx_va, EPOCHS, label_note=note)
        rows.append(("oversample", m, ntr, va_acc, va_mf1))
    return rows

def main():
    set_seed(SEED)
    print(f"[Config] RUN_MODE={RUN_MODE}  DO_DOWNSAMPLE_SCAN={DO_DOWNSAMPLE_SCAN}  DO_OVERSAMPLE_SCAN={DO_OVERSAMPLE_SCAN}")
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    all_rows = []

    if DO_DOWNSAMPLE_SCAN:
        rows = run_downsample_scan()
        all_rows.extend(rows)

    if DO_OVERSAMPLE_SCAN:
        rows = run_oversample_scan()
        all_rows.extend(rows)

    # 寫入 CSV
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mode", "ratio_or_mult", "train_size", "val_acc", "val_mf1",
                    "seed", "epochs", "use_sampler", "batch_size", "lr"])
        for mode, rom, ntr, acc, mf1 in all_rows:
            w.writerow([mode, rom, ntr, f"{acc:.6f}", f"{mf1:.6f}",
                        SEED, EPOCHS, int(USE_SAMPLER),
                        (BATCH_SIZE if BATCH_SIZE else "cfg"),
                        (LR if LR else "cfg")])
    print(f"[Done] Results saved to: {OUT_CSV}")

if __name__ == "__main__":
    main()
