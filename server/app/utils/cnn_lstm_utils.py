import torch
import torch.nn as nn

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
