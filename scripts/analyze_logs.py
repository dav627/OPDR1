#!/usr/bin/env python3
"""从本地训练日志提取关键指标并画曲线图。

用法：
    python scripts/analyze_logs.py <log_dir>
    python scripts/analyze_logs.py logs/Qwen2.5-3B-Instruct_GRPO_simulate_sft_0_0.5_wiki_turns_5

输出：
    <log_dir>/metrics.csv — 提取的指标表
    <log_dir>/loss_curve.png — loss 曲线图
    <log_dir>/reward_curve.png — reward 曲线图（RL 臂）
    <log_dir>/gpu_usage.png — GPU 显存/利用率曲线
"""

import sys
import os
import re
import csv
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')  # 无显示环境
    import matplotlib.pyplot as plt
except ImportError:
    print("请安装 matplotlib: pip install matplotlib")
    sys.exit(1)


def parse_train_log(log_path: Path) -> list:
    """从训练输出日志解析每步的指标。"""
    metrics_list = []
    current_step = None
    current_metrics = {}

    # 匹配形如 'actor/pg_loss': 0.123 的行
    pattern = re.compile(r"'([\w/]+)':\s*([\d.eE+-]+|nan|inf)")

    with open(log_path, 'r', errors='ignore') as f:
        for line in f:
            # 检测新 step 开始
            if 'global_steps' in line and '=' in line:
                if current_metrics and current_step is not None:
                    current_metrics['step'] = current_step
                    metrics_list.append(current_metrics)
                current_metrics = {}
                m = re.search(r'global_steps\s*[:=]\s*(\d+)', line)
                if m:
                    current_step = int(m.group(1))
                continue

            # 提取指标
            for match in pattern.finditer(line):
                key, value = match.group(1), match.group(2)
                try:
                    current_metrics[key] = float(value)
                except ValueError:
                    pass

    # 收尾
    if current_metrics and current_step is not None:
        current_metrics['step'] = current_step
        metrics_list.append(current_metrics)

    return metrics_list


def parse_gpu_log(gpu_log_path: Path) -> list:
    """解析 GPU 监控 CSV。"""
    rows = []
    if not gpu_log_path.exists():
        return rows
    with open(gpu_log_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append({
                    'timestamp': int(row['timestamp']),
                    'mem_used': float(row['memory.used_MiB']),
                    'mem_total': float(row['memory.total_MiB']),
                    'util': float(row['utilization_pct']),
                    'temp': float(row['temperature_C']),
                })
            except (ValueError, KeyError):
                pass
    return rows


def save_metrics_csv(metrics_list: list, out_path: Path):
    """保存指标为 CSV。"""
    if not metrics_list:
        print("无指标数据")
        return
    keys = sorted({k for m in metrics_list for k in m.keys()})
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for m in metrics_list:
            writer.writerow(m)
    print(f"指标已保存: {out_path} ({len(metrics_list)} 步)")


def plot_curves(metrics_list: list, log_dir: Path, arm: str):
    """画 loss 曲线和 reward 曲线。"""
    if not metrics_list:
        print("无数据可画图")
        return

    steps = [m.get('step', i) for i, m in enumerate(metrics_list)]

    # ── Loss 曲线 ──────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{arm} Arm — Training Curves', fontsize=14)

    # 主 loss
    ax = axes[0, 0]
    if arm == 'OPD':
        loss_key = 'actor/opd_loss'
        ax.set_ylabel('OPD Loss (KL to teacher)')
    else:
        loss_key = 'actor/pg_loss'
        ax.set_ylabel('PG Loss')
    vals = [m.get(loss_key) for m in metrics_list]
    ax.plot(steps, vals, 'b-', label=loss_key)
    ax.set_xlabel('Step')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # KL loss
    ax = axes[0, 1]
    vals = [m.get('actor/kl_loss') for m in metrics_list]
    ax.plot(steps, vals, 'r-', label='actor/kl_loss')
    ax.set_xlabel('Step')
    ax.set_ylabel('KL Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Entropy
    ax = axes[1, 0]
    vals = [m.get('actor/entropy_loss') for m in metrics_list]
    ax.plot(steps, vals, 'g-', label='actor/entropy_loss')
    ax.set_xlabel('Step')
    ax.set_ylabel('Entropy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # OPD 专属：logp_gap；GRPO：reward
    ax = axes[1, 1]
    if arm == 'OPD':
        vals = [m.get('actor/logp_gap') for m in metrics_list]
        ax.plot(steps, vals, 'm-', label='actor/logp_gap (teacher - student)')
        ax.set_ylabel('Log-prob Gap')
    else:
        # 找 reward 相关指标
        reward_keys = [k for k in metrics_list[0].keys() if 'reward' in k.lower() and 'mean' in k.lower()]
        for k in reward_keys[:3]:
            vals = [m.get(k) for m in metrics_list]
            ax.plot(steps, vals, label=k)
        ax.set_ylabel('Reward')
    ax.set_xlabel('Step')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = log_dir / 'loss_curve.png'
    plt.savefig(out_path, dpi=100)
    print(f"Loss 曲线已保存: {out_path}")
    plt.close()


def plot_gpu(gpu_rows: list, log_dir: Path):
    """画 GPU 使用曲线。"""
    if not gpu_rows:
        print("无 GPU 监控数据")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('GPU Usage During Training', fontsize=14)

    ts = [r['timestamp'] - gpu_rows[0]['timestamp'] for r in gpu_rows]

    # 显存
    ax1.plot(ts, [r['mem_used'] for r in gpu_rows], 'b-', label='Used (MiB)')
    ax1.plot(ts, [r['mem_total'] for r in gpu_rows], 'r--', label='Total (MiB)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Memory (MiB)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 利用率
    ax2.plot(ts, [r['util'] for r in gpu_rows], 'g-', label='Utilization (%)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('GPU Utilization (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = log_dir / 'gpu_usage.png'
    plt.savefig(out_path, dpi=100)
    print(f"GPU 使用曲线已保存: {out_path}")
    plt.close()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    log_dir = Path(sys.argv[1])
    if not log_dir.exists():
        print(f"日志目录不存在: {log_dir}")
        sys.exit(1)

    train_log = log_dir / 'train_output.log'
    gpu_log = log_dir / 'gpu_monitor.csv'

    # 判断 arm 类型
    dir_name = log_dir.name
    if 'OPD' in dir_name:
        arm = 'OPD'
    elif 'GRPO' in dir_name:
        arm = 'GRPO'
    else:
        arm = 'Unknown'

    print(f"分析日志目录: {log_dir}")
    print(f"训练臂: {arm}")

    # 解析训练日志
    if train_log.exists():
        print(f"\n解析训练日志: {train_log}")
        metrics_list = parse_train_log(train_log)
        save_metrics_csv(metrics_list, log_dir / 'metrics.csv')
        plot_curves(metrics_list, log_dir, arm)
    else:
        print(f"训练日志不存在: {train_log}")

    # 解析 GPU 日志
    if gpu_log.exists():
        print(f"\n解析 GPU 监控: {gpu_log}")
        gpu_rows = parse_gpu_log(gpu_log)
        plot_gpu(gpu_rows, log_dir)
    else:
        print(f"GPU 监控日志不存在: {gpu_log}")

    print("\n✅ 分析完成")


if __name__ == '__main__':
    main()
