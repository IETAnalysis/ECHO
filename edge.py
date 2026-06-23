import os
import glob
import time
import uuid
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import warnings
import threading
import heapq
from sklearn.metrics import precision_recall_fscore_support, classification_report
import math

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float32)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_LOCK = threading.Lock()
HEAP_LOCK = threading.Lock()


class EdgeConfig:
    LIVE_TRAFFIC_DATA = "/home/njust/data/slj/全加密/提取的特征/903-单周-CESNET-TLS-Year22-精度32/traget_train_5~6"
    UNKNOWN_DATA_DIR = "/home/njust/data/slj/全加密/提取的特征/903-单周-CESNET-TLS-Year22-精度32/USTC2016"
    DOWNLOADED_WEIGHTS = os.path.join(BASE_DIR, "huawei_arcface_updated.pth")
    DOWNLOADED_THRESHOLDS = os.path.join(BASE_DIR, "huawei_energy_thresholds.pt")
    FALLBACK_WEIGHTS = os.path.join(BASE_DIR, "huawei_v14_pure_concat_train_ep10.pth")
    STREAM_BUFFER_DIR = os.path.join(BASE_DIR, "stream_buffer")
    DEVICE = 'cuda:3' if torch.cuda.is_available() else 'cpu'
    SEQ_LEN, STAT_DIM, NUM_CLASSES, BATCH_SIZE = 000, 000, 000, 000
    QUOTA_PER_CLASS = 000


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
        self.classifier = CosFace(feat_dim, num_classes, s=32.0, m=0.20)

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


class TrafficDataset(Dataset):
    def __init__(self, data_dir):
        self.seqs, self.stats, self.labels = [], [], []
        for f in glob.glob(os.path.join(data_dir, "*.pt")):
            data = torch.load(f, weights_only=False)
            self.seqs.append(data['seq'].float())
            self.stats.append(data['stats'].float())
            self.labels.append(data['labels'])
        if self.seqs:
            self.seqs, self.stats, self.labels = torch.cat(self.seqs, dim=0), torch.cat(self.stats, dim=0), torch.cat(self.labels, dim=0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.seqs[idx], self.stats[idx], self.labels[idx]

def snapshot_upload_worker(cfg, class_heaps, ood_heap):
    os.makedirs(cfg.STREAM_BUFFER_DIR, exist_ok=True)
    while True:
        time.sleep(10.0)
        final_seqs, final_stats, final_is_ood = [], [], []

        with HEAP_LOCK:
            for c, heap in class_heaps.items():
                for item in heap:
                    final_seqs.append(item[2].unsqueeze(0))
                    final_stats.append(item[3].unsqueeze(0))
                    final_is_ood.append(False)
            for item in ood_heap:
                final_seqs.append(item[2].unsqueeze(0))
                final_stats.append(item[3].unsqueeze(0))
                final_is_ood.append(True)

        if final_seqs:
            seqs = torch.cat(final_seqs, dim=0)
            stats = torch.cat(final_stats, dim=0)
            is_oods = torch.tensor(final_is_ood, dtype=torch.bool)
            save_path = os.path.join(cfg.STREAM_BUFFER_DIR, f"snapshot_{int(time.time())}.pt")
            torch.save({'seqs': seqs, 'stats': stats, 'is_ood': is_oods}, save_path)


def hot_update_worker(model, thresholds_dict, cfg, base_state):
    current_mtime = 0
    if os.path.exists(cfg.DOWNLOADED_WEIGHTS):
        current_mtime = os.path.getmtime(cfg.DOWNLOADED_WEIGHTS)

    while True:
        time.sleep(3.0)
        if not os.path.exists(cfg.DOWNLOADED_WEIGHTS): continue
        mtime = os.path.getmtime(cfg.DOWNLOADED_WEIGHTS)
        if mtime > current_mtime:
            print("\n⚙️ [系统] 收到云端最新快照抗体，正在热替换内存权重...")
            try:
                delta_state = torch.load(cfg.DOWNLOADED_WEIGHTS, map_location=cfg.DEVICE)
                with MODEL_LOCK:
                    current_state = {k: v.clone() for k, v in base_state.items()}
                    for k in delta_state:
                        if k in current_state:
                            target_dtype = current_state[k].dtype
                            current_state[k] += delta_state[k].to(device=cfg.DEVICE, dtype=target_dtype)
                        else:
                            current_state[k] = delta_state[k].float().to(cfg.DEVICE)
                    model.load_state_dict(current_state, strict=False)
                    if os.path.exists(cfg.DOWNLOADED_THRESHOLDS):
                        new_t = torch.load(cfg.DOWNLOADED_THRESHOLDS)
                        thresholds_dict.update(new_t)
                current_mtime = mtime
                print("⚙️ [系统] 热更新完成！边缘探针已加载最新抗体！\n")
            except Exception as e:
                pass


# 支持“混合真乱序”数据流的推断引擎
def edge_live_inference_mixed(model, mixed_loader, thresholds_dict, cfg, class_heaps, ood_heap):
    model.eval()
    sample_counter = 0

    # 全局统计变量
    tot_unk, rej_unk, acc_wrg_unk = 0, 0, 0
    tot_tgt, rej_tgt, acc_cor_tgt, raw_cor_tgt = 0, 0, 0, 0
    y_true_list, y_pred_list = [], []

    # 局部窗口监控变量
    local_tot_unk, local_rej_unk = 0, 0
    window_size = 500  # 每处理 500 个 batch 打印一次瞬时战报

    print("\n🛡️ 边缘常驻防护启动 (混合流量全景流式探测)...")
    with torch.no_grad():
        for batch_idx, (seq_x, stat_x, labels, is_ood_batch) in enumerate(tqdm(mixed_loader, desc="正在处理并发混合流量", ncols=100)):
            seq_x = seq_x.to(cfg.DEVICE, dtype=torch.float32)
            stat_x = stat_x.to(cfg.DEVICE, dtype=torch.float32)
            labels = labels.to(cfg.DEVICE)
            is_ood_batch = is_ood_batch.to(cfg.DEVICE)

            with MODEL_LOCK:
                T_cos_dict = thresholds_dict.get('T_cos_dict', {i: 0.2 for i in range(cfg.NUM_CLASSES)})
                _, _, cosines = model(seq_x, stat_x, return_cos=True)

            max_cos, preds = torch.max(cosines, dim=1)
            batch_thresholds = torch.tensor([T_cos_dict.get(p.item(), 0.25) for p in preds], device=cfg.DEVICE)
            rej_mask = max_cos < batch_thresholds
            acc_mask = ~rej_mask

            # 向量化统计
            ood_idx = torch.nonzero(is_ood_batch, as_tuple=True)[0]
            tgt_idx = torch.nonzero(~is_ood_batch, as_tuple=True)[0]

            # 更新全局统计
            tot_unk += len(ood_idx)
            tot_tgt += len(tgt_idx)

            rej_unk += rej_mask[ood_idx].sum().item()
            acc_wrg_unk += acc_mask[ood_idx].sum().item()

            rej_tgt += rej_mask[tgt_idx].sum().item()
            correct_mask = (preds == labels)
            raw_cor_tgt += correct_mask[tgt_idx].sum().item()
            acc_cor_tgt += (acc_mask[tgt_idx] & correct_mask[tgt_idx]).sum().item()

            acc_tgt_mask = acc_mask & (~is_ood_batch)
            if acc_tgt_mask.any():
                y_true_list.extend(labels[acc_tgt_mask].cpu().numpy())
                y_pred_list.extend(preds[acc_tgt_mask].cpu().numpy())

            # 更新局部统计并进行周期性打印
            local_tot_unk += len(ood_idx)
            local_rej_unk += rej_mask[ood_idx].sum().item()

            if (batch_idx + 1) % window_size == 0:
                local_tnr = (local_rej_unk / max(1, local_tot_unk)) * 100
                tqdm.write(f"📊 [监控] 第 {(batch_idx + 1):04d} 批次 - 当前窗口瞬时未知阻断率 (TNR): {local_tnr:.2f}%")
                local_tot_unk, local_rej_unk = 0, 0

            # 提取高质量快照样本 (Heap)
            with HEAP_LOCK:
                # OOD 恶人榜
                if len(ood_idx) > 0:
                    for i in ood_idx:
                        fit_val = max_cos[i].item()
                        if fit_val > 0.10:
                            if len(ood_heap) < cfg.QUOTA_PER_CLASS * 20:
                                heapq.heappush(ood_heap, (fit_val, sample_counter, seq_x[i].cpu().clone(), stat_x[i].cpu().clone()))
                            elif fit_val > ood_heap[0][0]:
                                heapq.heapreplace(ood_heap, (fit_val, sample_counter, seq_x[i].cpu().clone(), stat_x[i].cpu().clone()))
                            sample_counter += 1

                # 目标分类精英榜
                if len(tgt_idx) > 0:
                    for i in tgt_idx:
                        if correct_mask[i]:
                            pred_c = preds[i].item()
                            fit_val = max_cos[i].item()
                            heap = class_heaps[pred_c]
                            if len(heap) < cfg.QUOTA_PER_CLASS:
                                heapq.heappush(heap, (fit_val, sample_counter, seq_x[i].cpu().clone(), stat_x[i].cpu().clone()))
                            elif fit_val > heap[0][0]:
                                heapq.heapreplace(heap, (fit_val, sample_counter, seq_x[i].cpu().clone(), stat_x[i].cpu().clone()))
                            sample_counter += 1

    # 计算详细指标
    metrics = {}
    if len(y_true_list) > 0:
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true_list, y_pred_list, average='macro', zero_division=0)
        # 也可计算加权平均
        p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true_list, y_pred_list, average='weighted', zero_division=0)
        metrics = {
            'p_macro': p_macro, 'r_macro': r_macro, 'f1_macro': f1_macro,
            'p_weighted': p_weighted, 'r_weighted': r_weighted, 'f1_weighted': f1_weighted
        }

    accepted_target_tot = tot_tgt - rej_tgt
    accepted_acc = (acc_cor_tgt / accepted_target_tot) * 100 if accepted_target_tot > 0 else 0.0
    overall_target_acc = (acc_cor_tgt / max(1, tot_tgt)) * 100  # 综合放行成功率（目标样本正确分类占比）
    tnr = (rej_unk / max(1, tot_unk)) * 100

    print("\n" + "=" * 80)
    print("🏆 终端探针终极战报 (全局混合流式并发版)")
    print("-" * 80)
    print("【安检门防御能力】")
    print(f"🛡️ 未知应用阻断率 (TNR): {tnr:.2f}%")
    print(f"✅ 目标应用综合放行成功率 (目标样本正确分类占比): {overall_target_acc:.2f}%")

    print("\n【放行流量的纯分类能力】")
    print(f"🎯 放行流量分类准确率 (Accuracy): {accepted_acc:.2f}%")
    if metrics:
        print(f"📊 放行流量 Macro 精确率 (Precision): {metrics['p_macro'] * 100:.2f}%")
        print(f"📊 放行流量 Macro 召回率 (Recall):    {metrics['r_macro'] * 100:.2f}%")
        print(f"📊 放行流量 Macro F1-score:           {metrics['f1_macro'] * 100:.2f}%")
        print(f"📊 放行流量 Weighted 精确率: {metrics['p_weighted'] * 100:.2f}%")
        print(f"📊 放行流量 Weighted 召回率: {metrics['r_weighted'] * 100:.2f}%")
        print(f"📊 放行流量 Weighted F1-score: {metrics['f1_weighted'] * 100:.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    cfg = EdgeConfig()
    model = TrafficUnifiedEngine(cfg.SEQ_LEN, cfg.NUM_CLASSES, cfg.STAT_DIM).float().to(cfg.DEVICE)
    thresholds = {'T_cos_dict': {i: 0.2 for i in range(cfg.NUM_CLASSES)}}

    class_heaps = {i: [] for i in range(cfg.NUM_CLASSES)}
    ood_heap = []

    base_state_dict = torch.load(cfg.FALLBACK_WEIGHTS, map_location=cfg.DEVICE)
    model.load_state_dict(base_state_dict, strict=False)

    if os.path.exists(cfg.DOWNLOADED_WEIGHTS):
        try:
            delta_state = torch.load(cfg.DOWNLOADED_WEIGHTS, map_location=cfg.DEVICE)
            current_state = {k: v.clone() for k, v in base_state_dict.items()}
            for k in delta_state:
                if k in current_state:
                    current_state[k] += delta_state[k].to(device=cfg.DEVICE, dtype=current_state[k].dtype)
                else:
                    current_state[k] = delta_state[k].float().to(cfg.DEVICE)
            model.load_state_dict(current_state, strict=False)
            if os.path.exists(cfg.DOWNLOADED_THRESHOLDS):
                thresholds = torch.load(cfg.DOWNLOADED_THRESHOLDS)
        except Exception:
            pass

    uploader_thread = threading.Thread(target=snapshot_upload_worker, args=(cfg, class_heaps, ood_heap), daemon=True)
    updater_thread = threading.Thread(target=hot_update_worker, args=(model, thresholds, cfg, base_state_dict), daemon=True)
    uploader_thread.start()
    updater_thread.start()

    total_params = sum(p.numel() for p in model.parameters())
    param_size_mb = total_params * 4 / (1024 ** 2)
    print("\n" + "=" * 50)
    print(" 🛠️ 边缘终端实时运行体检")
    print("=" * 50)
    print(f"📊 模型参数: {total_params / 1e6:.2f} M | 内存占用: {param_size_mb:.2f} MB\n")

    target_dataset = TrafficDataset(cfg.LIVE_TRAFFIC_DATA)
    unk_dataset = TrafficDataset(cfg.UNKNOWN_DATA_DIR)

    all_seqs = torch.cat([target_dataset.seqs, unk_dataset.seqs], dim=0)
    all_stats = torch.cat([target_dataset.stats, unk_dataset.stats], dim=0)
    all_labels = torch.cat([target_dataset.labels, unk_dataset.labels], dim=0)

    # 构建 is_ood 标签掩码
    all_is_ood = torch.cat([
        torch.zeros(len(target_dataset), dtype=torch.bool),
        torch.ones(len(unk_dataset), dtype=torch.bool)
    ], dim=0)

    # 封装为超级混合 Loader
    mixed_dataset = TensorDataset(all_seqs, all_stats, all_labels, all_is_ood)
    mixed_loader = DataLoader(mixed_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)

    edge_live_inference_mixed(model, mixed_loader, thresholds, cfg, class_heaps, ood_heap)