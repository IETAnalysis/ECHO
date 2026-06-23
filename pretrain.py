import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import warnings
import math

warnings.filterwarnings("ignore")
# ✨ 全局设置
torch.set_default_dtype(torch.float32)

class Config:
    PRETRAIN_DATA_DIR = "/home/njust/data/slj/全加密/提取的特征/903-单周-CESNET-TLS-Year22-精度32/pretrain"
    SEQ_LEN, STAT_DIM, NUM_CLASSES, BATCH_SIZE = 000, 000, 000, 000
    DEVICE = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    EPOCHS, LR = 000, 000
    SAVE_PATH = "./huawei_v14_pure_concat_train.pth"


class MultiScaleResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        c_sub = out_channels // 3
        self.conv3 = nn.Conv1d(in_channels, c_sub, 3, padding=1, bias=False)
        self.conv5 = nn.Conv1d(in_channels, c_sub, 5, padding=2, bias=False)
        self.conv7 = nn.Conv1d(in_channels, out_channels - 2 * c_sub, 7, padding=3, bias=False)
        self.norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv_out = nn.Conv1d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm_out = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential(nn.Conv1d(in_channels, out_channels, 1, bias=False), nn.BatchNorm1d(out_channels)) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        out = self.relu(self.norm(torch.cat([self.conv3(x), self.conv5(x), self.conv7(x)], dim=1)))
        return self.relu(self.norm_out(self.conv_out(out)) + self.shortcut(x))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x): return x + self.pe[:, :x.size(1)]


class CosFace(nn.Module):
    def __init__(self, in_features, out_features, s=000, m=000):
        super().__init__()
        self.s, self.m = s, m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, labels=None):
        cosine = F.linear(F.normalize(x, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        if labels is None: return cosine * self.s, cosine
        one_hot = torch.zeros(cosine.size(), device=x.device).scatter_(1, labels.view(-1, 1).long(), 1)
        return (cosine - one_hot * self.m) * self.s, cosine


class TrafficUnifiedEngine(nn.Module):
    def __init__(self, seq_len=000, num_classes=000, stat_dim=000, feat_dim=000):
        super().__init__()
        self.seq_len = seq_len
        self.stem = nn.Sequential(MultiScaleResBlock1D(3, 128), MultiScaleResBlock1D(128, feat_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, feat_dim))
        self.pos_encoder = PositionalEncoding(d_model=feat_dim, max_len=100)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=feat_dim, nhead=8, dim_feedforward=1024, dropout=0.1, batch_first=True, norm_first=True), num_layers=3)
        self.stat_encoder = nn.Sequential(nn.BatchNorm1d(stat_dim), nn.Linear(stat_dim, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, feat_dim))

        # ✨ 废除花里胡哨的门控，回归暴力的 Concat
        self.fusion_fc = nn.Sequential(nn.Linear(feat_dim * 2, feat_dim), nn.BatchNorm1d(feat_dim), nn.GELU(), nn.Dropout(0.2))
        self.classifier = CosFace(feat_dim, num_classes, s=000, m=000)

    def forward(self, seq_x, stat_x, labels=None, return_cos=False):
        B = seq_x.size(0)
        iat = torch.log1p(torch.relu(seq_x[:, :, 0:1]))
        size = seq_x[:, :, 1:2] / 1500.0
        direction = (seq_x[:, :, 2:3] + 1.0) / 2.0
        seq_norm = torch.nan_to_num(torch.cat([iat, size, direction], dim=-1), nan=0.0).permute(0, 2, 1)

        seq_emb = self.stem(seq_norm).permute(0, 2, 1)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        seq_with_cls = torch.cat((cls_tokens, seq_emb), dim=1)

        seq_feat = self.transformer(self.pos_encoder(seq_with_cls))
        cls_feat = seq_feat[:, 0, :]

        stat_norm = torch.sign(torch.nan_to_num(stat_x, nan=0.0)) * torch.log1p(torch.abs(torch.nan_to_num(stat_x, nan=0.0)))
        stat_feat = self.stat_encoder(stat_norm)

        # 暴力拼接
        fused_feat = self.fusion_fc(torch.cat([cls_feat, stat_feat], dim=1))

        logits, cosines = self.classifier(fused_feat, labels)
        if return_cos: return logits, fused_feat, cosines
        return logits, fused_feat


class TrafficDataset(Dataset):
    def __init__(self, data_dir):
        self.seqs, self.stats, self.labels = [], [], []
        for f in tqdm(glob.glob(os.path.join(data_dir, "*.pt")), desc="加载数据"):
            data = torch.load(f, weights_only=False)
            self.seqs.append(data['seq'].float())
            self.stats.append(data['stats'].float())
            self.labels.append(data['labels'])
        if self.seqs: self.seqs, self.stats, self.labels = torch.cat(self.seqs, dim=0), torch.cat(self.stats, dim=0), torch.cat(self.labels, dim=0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.seqs[idx], self.stats[idx], self.labels[idx]


if __name__ == "__main__":
    cfg = Config()
    loader = DataLoader(TrafficDataset(cfg.PRETRAIN_DATA_DIR), batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    model = TrafficUnifiedEngine(cfg.SEQ_LEN, cfg.NUM_CLASSES, cfg.STAT_DIM).float().to(cfg.DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler()
    best_loss = float('inf')

    print(f"🚀 启动")
    for epoch in range(cfg.EPOCHS):
        model.train()
        running_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1:02d}/{cfg.EPOCHS}", ncols=100)

        for seq_x, stat_x, labels in pbar:
            seq_x, stat_x, labels = seq_x.to(cfg.DEVICE), stat_x.to(cfg.DEVICE), labels.to(cfg.DEVICE)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                logits, _ = model(seq_x, stat_x, labels=labels)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix({'Loss': f"{running_loss / (pbar.n + 1):.4f}"})

        scheduler.step()
        avg_loss = running_loss / len(loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), cfg.SAVE_PATH)
            print(f"  🌟 最优权重已保存 (Loss: {best_loss:.4f})")
        if (epoch + 1) % 5 == 0: torch.save(model.state_dict(), cfg.SAVE_PATH.replace('.pth', f'_ep{epoch + 1}.pth'))