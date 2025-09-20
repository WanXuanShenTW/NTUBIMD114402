import torch
import torch.nn as nn
from typing import Optional

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
    """支援 mask 的 temporal 聚合：last/mean/attn（與新版 trainer 對齊）"""
    def __init__(self, in_dim, num_classes, mode="attn", dropout=0.0):
        super().__init__()
        self.mode = mode
        self.drop = nn.Dropout(dropout) if dropout>0 else nn.Identity()
        if mode == "attn":
            self.attn = nn.Linear(in_dim,1)
        self.fc = nn.Linear(in_dim, num_classes)
    def forward(self, seq_feats: torch.Tensor, mask: Optional[torch.Tensor]=None):
        # seq_feats: (B,T,D), mask: (B,T) in {0,1}
        if (mask is not None) and (self.mode in ("mean","attn")):
            if self.mode == "mean":
                m = mask.unsqueeze(-1)  # (B,T,1)
                den = m.sum(dim=1).clamp_min(1e-6)
                g = (seq_feats * m).sum(dim=1) / den
            else:  # attn + mask
                a = self.attn(seq_feats).squeeze(-1)          # (B,T)
                a = a.masked_fill((mask<=0), float("-inf"))
                w = torch.softmax(a, dim=1).unsqueeze(-1)     # (B,T,1)
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
    """
    CNN+LSTM（新增 motion_dim 與 mask 參數，向下相容）
    - forward(x, motion=None, mask=None)
    - 若 motion_dim>0 但 motion=None，則僅用 CNN 特徵（向下相容）
    - 若傳入 mask，mean/attn 會忽略無效幀
    """
    def __init__(self, in_ch, num_classes, cnn_out=256, lstm_h=256, lstm_layers=2,
                 bidirectional=False, temporal_pool="attn", dropout=0.3, motion_dim: int = 0):
        super().__init__()
        self.cnn = SpaceCNN(in_ch, cnn_out)
        self.motion_dim = int(motion_dim) if motion_dim else 0
        lstm_in = cnn_out + self.motion_dim
        self.lstm = nn.LSTM(lstm_in, lstm_h, lstm_layers, batch_first=True, bidirectional=bidirectional)
        feat_dim = lstm_h * (2 if bidirectional else 1)
        self.head = TemporalHead(feat_dim, num_classes, mode=temporal_pool, dropout=dropout)
    def forward(self, x: torch.Tensor, motion: Optional[torch.Tensor]=None, mask: Optional[torch.Tensor]=None):
        # x: (B,T,C,H,W)  motion: (B,T,D)  mask: (B,T)
        B,T,C,H,W = x.shape
        z = self.cnn(x.view(B*T, C, H, W)).view(B, T, -1)  # (B,T,cnn_out)
        if self.motion_dim > 0 and motion is not None:
            z = torch.cat([z, motion], dim=-1)
        out,_ = self.lstm(z)                               # (B,T,H)
        return self.head(out, mask=mask)