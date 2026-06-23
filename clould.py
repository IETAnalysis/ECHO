import os
import glob
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import warnings
import math

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float32)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class CloudConfig:
    STREAM_BUFFER_DIR = os.path.join(BASE_DIR, "stream_buffer")
    SOURCE_ENCODER_PATH = os.path.join(BASE_DIR, "huawei_v14_pure_concat_train_ep10.pth")
    EXPORT_WEIGHTS = os.path.join(BASE_DIR, "huawei_arcface_updated.pth")
    EXPORT_THRESHOLDS = os.path.join(BASE_DIR, "huawei_energy_thresholds.pt")
    DEVICE = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    SEQ_LEN, STAT_DIM, NUM_CLASSES, BATCH_SIZE = 000, 000, 000, 000
    UDA_EPOCHS = 000


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
        self.fusion_fc = nn.Sequential(nn.Linear(feat_dim * 2, feat_dim), nn.BatchNorm1d(feat_dim), nn.GELU(), nn.Dropout(0.2))
        self.classifier = CosFace(feat_dim, num_classes, s=000, m=000)
        self.log_vars = nn.Parameter(torch.zeros(3))

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
        fused_feat = self.fusion_fc(torch.cat([cls_feat, stat_feat], dim=1))

        logits, cosines = self.classifier(fused_feat, labels)
        if return_cos: return logits, fused_feat, cosines
        return logits, fused_feat


def filter_poisoned_data(model, dataset, cfg):
    model.eval()
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    clean_seqs, clean_stats, clean_is_ood = [], [], []

    with torch.no_grad():
        for seq_x, stat_x, is_ood in loader:
            seq_x, stat_x = seq_x.to(cfg.DEVICE), stat_x.to(cfg.DEVICE)
            _, _, cosines = model(seq_x, stat_x, return_cos=True)
            max_cos, _ = torch.max(cosines, dim=1)

            target_mask = (~is_ood.to(cfg.DEVICE)) & (max_cos > 0.15)
            ood_mask = is_ood.to(cfg.DEVICE)
            valid_mask = target_mask | ood_mask
            if valid_mask.any():
                clean_seqs.append(seq_x[valid_mask].cpu())
                clean_stats.append(stat_x[valid_mask].cpu())
                clean_is_ood.append(is_ood[valid_mask.cpu()])

    if not clean_seqs: return None
    final_seqs, final_stats = torch.cat(clean_seqs, dim=0), torch.cat(clean_stats, dim=0)
    final_is_ood = torch.cat(clean_is_ood, dim=0)
    return TensorDataset(final_seqs, final_stats, final_is_ood)


def cloud_cosine_uda(model, raw_loader, cfg):
    for name, param in model.named_parameters(): param.requires_grad = True

    backbone_params = [p for n, p in model.named_parameters() if 'classifier' not in n and 'log_vars' not in n]
    head_params = [p for n, p in model.named_parameters() if 'classifier' in n]
    log_var_params = [p for n, p in model.named_parameters() if 'log_vars' in n]

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 2e-5},
        {'params': head_params, 'lr': 2e-6},
        {'params': log_var_params, 'lr': 1e-3, 'weight_decay': 0.0}
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.UDA_EPOCHS, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler()
    class_ema_cos = torch.full((cfg.NUM_CLASSES,), 0.60, device=cfg.DEVICE)

    for epoch in range(cfg.UDA_EPOCHS):
        pseudo_weight = min(1.0, (epoch + 1) / (cfg.UDA_EPOCHS / 2)) * 0.8

        for seq_x, stat_x, is_ood in raw_loader:
            seq_x, stat_x = seq_x.to(cfg.DEVICE), stat_x.to(cfg.DEVICE)
            is_ood = is_ood.to(cfg.DEVICE)
            seq_x_aug, stat_x_aug = seq_x.clone(), stat_x.clone()
            batch_sz, seq_len, _ = seq_x.shape

            mask_indices = torch.randint(0, seq_len, (batch_sz, 1), device=cfg.DEVICE)
            for b in range(batch_sz): seq_x_aug[b, mask_indices[b], 1] = 0.0
            jitter_iat = torch.empty_like(seq_x_aug[:, :, 0]).uniform_(0.85, 1.15)
            seq_x_aug[:, :, 0] = seq_x_aug[:, :, 0] * jitter_iat

            optimizer.zero_grad()
            with torch.no_grad():
                model.eval()
                _, feat_clean, cosines_clean = model(seq_x, stat_x, return_cos=True)
                max_cos_clean, preds = torch.max(cosines_clean, dim=1)
                target_mask_clean = ~is_ood
                if target_mask_clean.any():
                    preds_target = preds[target_mask_clean]
                    max_cos_target = max_cos_clean[target_mask_clean]
                    for c in preds_target.unique():
                        c_mask = (preds_target == c)
                        class_ema_cos[c] = 0.9 * class_ema_cos[c] + 0.1 * max_cos_target[c_mask].mean()

            model.train()
            with torch.cuda.amp.autocast():
                _, feat_aug, cosines_aug = model(seq_x_aug, stat_x_aug, return_cos=True)
                loss_ce = torch.tensor(0.0, device=cfg.DEVICE)
                loss_ood_val = torch.tensor(0.0, device=cfg.DEVICE)
                loss_mse = torch.tensor(0.0, device=cfg.DEVICE)

                if target_mask_clean.any():
                    loss_mse = F.mse_loss(feat_aug[target_mask_clean], feat_clean[target_mask_clean].detach())
                    cos_clean_target = max_cos_clean[target_mask_clean]
                    pred_target = preds[target_mask_clean]

                    dynamic_thresholds = torch.clamp(class_ema_cos[pred_target] * 0.90, min=0.20)
                    confident_mask = cos_clean_target > dynamic_thresholds

                    if confident_mask.any():
                        conf_preds = pred_target[confident_mask]
                        conf_feat_aug = feat_aug[target_mask_clean][confident_mask]
                        conf_logits_with_margin, _ = model.classifier(conf_feat_aug, labels=conf_preds)

                        class_counts = torch.bincount(conf_preds, minlength=cfg.NUM_CLASSES).float()
                        class_weights = 1.0 / torch.sqrt(class_counts + 1.0)
                        sample_weights = (class_weights / class_weights.mean())[conf_preds]
                        ce = F.cross_entropy(conf_logits_with_margin, conf_preds, reduction='none', label_smoothing=0.05)
                        loss_ce = (ce * sample_weights).mean()

                if is_ood.any():
                    logits_ood = cosines_aug[is_ood] * 32.0
                    uniform_dist = torch.ones_like(logits_ood) / cfg.NUM_CLASSES
                    loss_ood_val = F.cross_entropy(logits_ood, uniform_dist)

                precision_ce = torch.exp(-model.log_vars[0])
                precision_mse = torch.exp(-model.log_vars[1])
                precision_ood = torch.exp(-model.log_vars[2])

                loss = (precision_ce * (pseudo_weight * loss_ce) + model.log_vars[0]) + \
                       (precision_mse * loss_mse + model.log_vars[1]) + \
                       (precision_ood * loss_ood_val + model.log_vars[2])

            if loss.item() > 0:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                scaler.step(optimizer)
                scaler.update()

        scheduler.step()
    return model


def calibrate_and_dispatch(model, raw_loader, cfg, base_state_dict):
    model.eval()
    class_cosines = {i: [] for i in range(cfg.NUM_CLASSES)}

    with torch.no_grad():
        for seq_x, stat_x, is_ood in raw_loader:
            target_mask = ~is_ood
            if not target_mask.any(): continue
            seq_x, stat_x = seq_x[target_mask].to(cfg.DEVICE), stat_x[target_mask].to(cfg.DEVICE)
            _, _, cosines = model(seq_x, stat_x, return_cos=True)
            max_cos, preds = torch.max(cosines, dim=1)

            for i in range(len(preds)):
                if max_cos[i].item() > 0.05:
                    class_cosines[preds[i].item()].append(max_cos[i].item())

    T_cos_dict = {}
    for c in range(cfg.NUM_CLASSES):
        cosines = class_cosines[c]
        if len(cosines) > 15:
            adaptive_t = np.mean(cosines) - 3.0 * np.std(cosines)
            T_cos_dict[c] = float(np.clip(adaptive_t, 0.08, 0.22))
        else:
            T_cos_dict[c] = 0.12

    tuned_state = model.state_dict()
    delta_state = {}
    for k in tuned_state:
        if k in base_state_dict:
            delta_state[k] = (tuned_state[k] - base_state_dict[k].to(cfg.DEVICE)).half()
        else:
            delta_state[k] = tuned_state[k].half()

    temp_w = cfg.EXPORT_WEIGHTS + ".tmp"
    temp_t = cfg.EXPORT_THRESHOLDS + ".tmp"
    torch.save(delta_state, temp_w)
    torch.save({'T_cos_dict': T_cos_dict}, temp_t)
    os.replace(temp_w, cfg.EXPORT_WEIGHTS)
    os.replace(temp_t, cfg.EXPORT_THRESHOLDS)
    print(f"🚀 [云端中心] 差分权重包及阈值生成完毕，已下发！")


if __name__ == "__main__":
    cfg = CloudConfig()
    os.makedirs(cfg.STREAM_BUFFER_DIR, exist_ok=True)

    model = TrafficUnifiedEngine(cfg.SEQ_LEN, cfg.NUM_CLASSES, cfg.STAT_DIM).float().to(cfg.DEVICE)
    base_state_dict = torch.load(cfg.SOURCE_ENCODER_PATH, map_location=cfg.DEVICE)
    model.load_state_dict(base_state_dict, strict=False)

    print("📡 [云端服务] 快照监听模式已启动，等待边缘高质量数据...")

    while True:
        files = glob.glob(os.path.join(cfg.STREAM_BUFFER_DIR, "snapshot_*.pt"))
        if files:
            files.sort(key=os.path.getctime)
            latest_file = files[-1]

            try:
                data = torch.load(latest_file, weights_only=False)
                for f in files: os.remove(f)

                is_oods = data['is_ood']
                t_count = (~is_oods).sum().item()
                o_count = is_oods.sum().item()

                if t_count >= 64 and o_count >= 64:
                    print(f"\n🔔 [云端服务] 捕获最新精英快照 (Target:{t_count}, OOD:{o_count})，触发极速演进...")
                    raw_dataset = TensorDataset(data['seqs'].float(), data['stats'].float(), data['is_ood'])
                    cleaned_dataset = filter_poisoned_data(model, raw_dataset, cfg)

                    if cleaned_dataset:
                        actual_batch_size = min(cfg.BATCH_SIZE, len(cleaned_dataset))
                        raw_loader = DataLoader(cleaned_dataset, batch_size=actual_batch_size, shuffle=True)
                        model = cloud_cosine_uda(model, raw_loader, cfg)

                        full_eval_loader = DataLoader(cleaned_dataset, batch_size=actual_batch_size, shuffle=False)
                        calibrate_and_dispatch(model, full_eval_loader, cfg, base_state_dict)

                    print("📡 [云端服务] 演进结束，继续等待下一张快照...\n")
                else:
                    print(f"⏳ [云端防御] 当前快照目标或未知特征积累不足 (T:{t_count}, O:{o_count})，暂不演进以防遗忘。")

            except Exception as e:
                pass

        time.sleep(2.0)