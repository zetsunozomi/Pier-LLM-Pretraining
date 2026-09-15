# add subgroup size as we add gpus.

import numpy as np
import matplotlib.pyplot as plt

dark_orange = "#1f77b4"  # (蓝) 用于 DiLoCo 线
light_orange = "#aec7e8"

dark_blue = "#ff7f0e"    # (橙) 用于 DDP 线
light_blue = "#ffbb78"

# 1) 增加 1-GPU
gpu_labels = [1, 2, 4, 8, 16, 32,64,128]
x = list(range(len(gpu_labels)))

# --- 时间数据（秒） ---
# 柱状图用：DiLoCo 在 1-GPU 不存在 -> NaN
times_algo1_bars = [np.nan, 5889.432, 3057.892, 1602.692 , 842.425, 467.289 ,  292.383, 209.787]
# “scaling”折线用：DiLoCo 在 1-GPU 设为 （只用于连线与效率计算）
times_algo1_line = [11351.476, 5889.432, 3057.892, 1602.692 , 842.425, 467.289 ,  292.383, 209.787]

# DDP：
times_algo2 = [11351.476, 5912.502, 3126.64, 2072.53,1364.296, 969.159, 812.394, 620.094]

# --- 效率统一以 “DDP 1-GPU(1897.917s)” 为基准 ---
G0 = gpu_labels[0]   # 1
T0 = times_algo2[0]  # DDP 1-GPU

def calc_eff(times, gpus, T0, G0=1):
    eff = []
    for t, G in zip(times, gpus):
        if t is None or (isinstance(t, float) and not np.isfinite(t)):
            eff.append(np.nan)
        else:
            eff.append((T0 * G0) / (t * G) * 100.0)
    return eff

eff_1 = calc_eff(times_algo1_line, gpu_labels, T0, G0)  
eff_2 = calc_eff(times_algo2,       gpu_labels, T0, G0)

fig, ax_time = plt.subplots(figsize=(7,4))

# 柱状图：时间（每次迭代）
width = 0.20
bars1 = ax_time.bar([i - width/2 for i in x], times_algo1_bars, width, label='DiLoCo time', color=light_orange)
bars2 = ax_time.bar([i + width/2 for i in x], times_algo2,      width, label='DDP time',    color=light_blue)

ax_time.set_xlabel('number of GPU')
ax_time.set_ylabel('time per iteration (s)')
ax_time.set_xticks(x)
ax_time.set_xticklabels(gpu_labels)
ax_time.grid(True, which='both', axis='both', linestyle=':')

# 双轴：效率（“scaling”）
ax_eff = ax_time.twinx()
l3, = ax_eff.plot(x, eff_1, marker='^', linestyle='--', linewidth=2, label='DiLoCo scaling', color=dark_orange)
l4, = ax_eff.plot(x, eff_2, marker='v', linestyle='--', linewidth=2, label='DDP scaling',    color=dark_blue)
ax_eff.set_ylabel('Scaling Efficiency (%)')
ax_eff.set_ylim(0, 110)

# 合并图例
lines = [bars1, bars2, l3, l4]
labels = [l.get_label() for l in lines]
ax_time.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=2, frameon=False)
plt.title('Strong Scaling: Time & Efficiency vs. GPU (baseline: DDP 1-GPU)')
plt.tight_layout()

# 标注效率（跳过 NaN）
for xi, yi in zip(x, eff_1):
    if np.isfinite(yi):
        ax_eff.text(xi, yi + 2, f"{yi:.1f}%", ha='center', va='bottom', color=dark_orange, fontsize=9)

for xi, yi in zip(x, eff_2):
    if np.isfinite(yi):
        ax_eff.text(xi, yi - 2, f"{yi:.1f}%", ha='center', va='top', color=dark_blue, fontsize=9)

# 组内文本：用单词“gain”替换 speedup（DDP 相对 DiLoCo 的百分比；若任一为 NaN 则跳过）
for xi, t1, t2 in zip(x, times_algo1_bars, times_algo2):
    if np.isfinite(t1) and np.isfinite(t2):
        gain = (t2 / t1 - 1) * 100
        y = (t2 + t1) / 4
        ax_time.text(xi, y, f"gain {gain:.2f}%", ha='center', va='center', fontsize=9, color='black')

plt.savefig("benchmark_4_medium_perlmutter.png", dpi=200)
