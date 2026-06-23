import os
import json
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import warnings
from scipy.stats import skew, kurtosis

warnings.filterwarnings("ignore")

# ✨ 全局设置：强制默认张量为 32 位
torch.set_default_dtype(torch.float32)

RAW_YEAR22_DIR = "/home/njust/data/slj/全加密/数据集/CESNET-TLS-Year22"
OUTPUT_SAVE_PATH = "/home/njust/data/slj/全加密/提取的特征/903-单周-CESNET-TLS-Year22-精度32/pretrain"

SEQ_LEN = 000
STAT_DIM = 000
MAX_SAMPLES_PER_CLASS = 000
MIN_SAMPLES_PER_CLASS = 000

# ✨ 削减毒瘤字段，只需要 APP 和包含微观时序的 PPI
USE_COLS = ['APP', 'PPI']


def safe_calc(func, data, default=0.0):
    return float(func(data)) if len(data) > 0 else default


def parse_row_worker(row_dict):
    app = str(row_dict['APP'])
    try:
        ppi_list = json.loads(str(row_dict['PPI']))
        times, dirs, sizes = [float(x) for x in ppi_list[0]], [int(x) for x in ppi_list[1]], [float(x) for x in ppi_list[2]]

        if len(times) < 5: return None
        min_len = min(len(times), len(dirs), len(sizes))
        times, dirs, sizes = times[:min_len], dirs[:min_len], sizes[:min_len]

        seq_features, fwd_lens, bwd_lens, bursts = [], [], [], []
        current_burst_dir = dirs[0]
        current_burst_size = 0

        for i in range(min_len):
            iat, direction, length = times[i], dirs[i], sizes[i]
            if direction == 1:
                fwd_lens.append(length)
            else:
                bwd_lens.append(length)
            if len(seq_features) < SEQ_LEN: seq_features.append([iat, length, direction])

            if direction == current_burst_dir:
                current_burst_size += length
            else:
                bursts.append(current_burst_size)
                current_burst_dir = direction
                current_burst_size = length

        bursts.append(current_burst_size)
        while len(seq_features) < SEQ_LEN: seq_features.append([0.0, 0.0, 0.0])

        stats = np.zeros(STAT_DIM, dtype=np.float32)
        total_pkts = len(sizes)

        stats[0:5] = [safe_calc(np.mean, sizes), safe_calc(np.std, sizes), safe_calc(np.max, sizes), safe_calc(np.min, sizes), safe_calc(np.median, sizes)]
        stats[5:10] = [safe_calc(np.mean, fwd_lens), safe_calc(np.std, fwd_lens), safe_calc(np.max, fwd_lens), safe_calc(np.min, fwd_lens), safe_calc(np.median, fwd_lens)]
        stats[10:15] = [safe_calc(np.mean, bwd_lens), safe_calc(np.std, bwd_lens), safe_calc(np.max, bwd_lens), safe_calc(np.min, bwd_lens), safe_calc(np.median, bwd_lens)]

        stats[15:20] = [safe_calc(np.mean, times), safe_calc(np.std, times), safe_calc(np.max, times), safe_calc(np.min, times), safe_calc(np.median, times)]
        stats[20:24] = [safe_calc(np.mean, bursts), safe_calc(np.std, bursts), safe_calc(np.max, bursts), safe_calc(np.median, bursts)]

        stats[24] = len(fwd_lens) / total_pkts if total_pkts > 0 else 0
        stats[25] = sum(fwd_lens) / sum(sizes) if sum(sizes) > 0 else 0

        for i in range(min(5, total_pkts)): stats[26 + i] = sizes[i]
        stats[31] = sum([sizes[i] for i in range(min(10, total_pkts)) if dirs[i] == 1])
        stats[32] = sum([sizes[i] for i in range(min(10, total_pkts)) if dirs[i] == -1])

        if len(fwd_lens) > 0:
            f_arr = np.array(fwd_lens)
            stats[33:38] = [np.sum(f_arr < 100) / len(f_arr), np.sum((f_arr >= 100) & (f_arr < 300)) / len(f_arr), np.sum((f_arr >= 300) & (f_arr < 800)) / len(f_arr),
                            np.sum((f_arr >= 800) & (f_arr < 1200)) / len(f_arr), np.sum(f_arr >= 1200) / len(f_arr)]
        if len(bwd_lens) > 0:
            b_arr = np.array(bwd_lens)
            stats[38:43] = [np.sum(b_arr < 100) / len(b_arr), np.sum((b_arr >= 100) & (b_arr < 300)) / len(b_arr), np.sum((b_arr >= 300) & (b_arr < 800)) / len(b_arr),
                            np.sum((b_arr >= 800) & (b_arr < 1200)) / len(b_arr), np.sum(b_arr >= 1200) / len(b_arr)]

        if len(times) > 0:
            t_arr = np.array(times)
            stats[43:47] = [np.sum(t_arr < 0.001) / len(t_arr), np.sum((t_arr >= 0.001) & (t_arr < 0.01)) / len(t_arr), np.sum((t_arr >= 0.01) & (t_arr < 0.1)) / len(t_arr),
                            np.sum(t_arr >= 0.1) / len(t_arr)]

        stats[47] = safe_calc(skew, sizes)
        stats[48] = safe_calc(kurtosis, sizes)

        first_fwd_idx = dirs.index(1) if 1 in dirs else -1
        first_bwd_idx = dirs.index(-1) if -1 in dirs else -1
        if first_fwd_idx != -1 and first_bwd_idx != -1 and first_bwd_idx > first_fwd_idx:
            stats[49] = sum(times[first_fwd_idx + 1: first_bwd_idx + 1])

        stats = np.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)
        return {'app': app, 'seq': np.array(seq_features, dtype=np.float32), 'stats': stats}
    except Exception:
        return None


def main():
    if not os.path.exists(OUTPUT_SAVE_PATH): os.makedirs(OUTPUT_SAVE_PATH)
    data_buffer, finished_apps, label_map = {}, set(), {}

    week_folders = sorted([f for f in os.listdir(RAW_YEAR22_DIR) if f.startswith('WEEK-')])
    target_weeks = [w for w in week_folders if 1 <= int(w.split('-')[-1]) <= 2]

    # 优化 1：降低单次处理量和并发数，防止内存撑爆 (可根据你的服务器配置微调)
    CHUNK_SIZE = 20000
    MAX_WORKERS = 8

    # 优化 2：把进程池开在最外层，全局复用，避免反复创建销毁引发死锁
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for week in target_weeks:
            week_path = os.path.join(RAW_YEAR22_DIR, week)
            for day in os.listdir(week_path):
                day_path = os.path.join(week_path, day)
                if not os.path.isdir(day_path): continue

                for file in os.listdir(day_path):
                    if file.endswith(".csv"):
                        csv_path = os.path.join(day_path, file)
                        print(f"📄 正在处理文件: {csv_path}")  # 增加日志，知道卡在哪

                        try:
                            # 迭代读取 Chunk
                            for chunk in pd.read_csv(csv_path, chunksize=CHUNK_SIZE, usecols=USE_COLS):
                                chunk = chunk[~chunk['APP'].isin(finished_apps)]
                                if chunk.empty: continue

                                tasks = chunk.to_dict(orient='records')

                                # 将任务提交给全局进程池
                                results = list(tqdm(executor.map(parse_row_worker, tasks), total=len(tasks), leave=False))

                                for res in results:
                                    if not res or res['app'] in finished_apps: continue
                                    app = res['app']
                                    if app not in data_buffer: data_buffer[app] = []
                                    data_buffer[app].append(res)

                                    if len(data_buffer[app]) >= MAX_SAMPLES_PER_CLASS:
                                        if app not in label_map: label_map[app] = len(label_map)
                                        safe_app_name = app.replace('/', '_').replace('\\', '_')
                                        save_filename = f"App_{label_map[app]:03d}_{safe_app_name}.pt"

                                        torch.save({
                                            'seq': torch.tensor(np.array([s['seq'] for s in data_buffer[app]])),
                                            'stats': torch.tensor(np.array([s['stats'] for s in data_buffer[app]])),
                                            'labels': torch.full((len(data_buffer[app]),), label_map[app], dtype=torch.int64)
                                        }, os.path.join(OUTPUT_SAVE_PATH, save_filename))

                                        finished_apps.add(app)
                                        del data_buffer[app]
                                        print(f"\n🎉 [{app}] 提取完毕！: {save_filename}")

                        # 优化 3：不要用裸 except！把真实的报错打印出来
                        except Exception as e:
                            print(f"\n❌ 处理文件 {file} 时发生错误: {str(e)}")
                            # 这里可以选择 raise e 来彻底终止程序看完整报错栈
                            continue

    # 处理尾部数据
    for app, samples in data_buffer.items():
        if len(samples) >= MIN_SAMPLES_PER_CLASS:
            if app not in label_map: label_map[app] = len(label_map)
            safe_app_name = app.replace('/', '_').replace('\\', '_')
            save_filename = f"App_{label_map[app]:03d}_{safe_app_name}.pt"
            torch.save({'seq': torch.tensor(np.array([s['seq'] for s in samples]),dtype=torch.float32),
                        'stats': torch.tensor(np.array([s['stats'] for s in samples]),dtype=torch.float32),
                        'labels': torch.full((len(samples),), label_map[app], dtype=torch.int64)},
                       os.path.join(OUTPUT_SAVE_PATH, save_filename))
            print(f"🎉 [{app}] 提取完毕 (尾部剩余)！: {save_filename}")

    with open(os.path.join(OUTPUT_SAVE_PATH, "global_label_map.json"), "w") as f:
        json.dump(label_map, f)
    print(f"\n✅ 预训练特征提取完毕！")


if __name__ == "__main__":
    main()