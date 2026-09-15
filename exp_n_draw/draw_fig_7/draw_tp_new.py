# add subgroup size as we add gpus.

import numpy as np
import matplotlib.pyplot as plt

diloco_color = "#d34a47"
diloco_light = "#ff9897"

ddp_color = '#074f97'
ddp_light = '#5e95d0'
# 1) 增加 1-GPU
gpu_labels = [4, 128]
x = list(range(len(gpu_labels)))

# --- 时间数据（秒） ---
# DiLoCo data
times_algo1_bars = np.array([np.nan, 6801.5875])/1000
# DiLoCo data, but at gpu == 1 set to ddp
times_algo1_line = np.array([192888.0755, 6801.5875])/1000

# DDP
times_algo2 = np.array([192888.075  ,18070.80769])/1000

# Momentum Produce
times_algo3 = np.array([np.nan, 20935.875])/1000

times_algo1_bars = times_algo1_bars*0.9 + times_algo3*0.1
times_algo1_line = times_algo1_line*0.9 + times_algo3*0.1
times_algo1_line[0] = times_algo2[0]
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
scale = 1
fig, ax_time = plt.subplots(figsize=(12.2*scale,5.6*scale))


# 绘制柱状图：时间（每次迭代）
width = 0.20
offset = 0.1
bars1 = ax_time.bar([i - offset for i in x], times_algo1_bars, width, label='MALT Time',color= diloco_light)
bars2 = ax_time.bar([i + offset for i in x], times_algo2, width, label='DP+TP time', color = ddp_light)

group_centers = [(i - offset + i + offset) / 2 for i in x]
group_centers[0]+=offset
ax_time.set_xlabel('number of GPU',fontsize=24)
ax_time.set_ylabel('time per iteration (s)',fontsize=24)


xticks = x.copy()
xticks[0] += offset   # 把第一个tick右移
ax_time.set_xticks(xticks)
ax_time.set_xticklabels(gpu_labels)

ax_time.grid(True, which='both', axis='both', linestyle=':')
ax_time.set_ylim(0, 220)

# 效率双轴
ax_eff = ax_time.twinx()
print(eff_1)
l3, = ax_eff.plot(group_centers, eff_1, marker='^', linestyle='--', linewidth=2, label='MALT efficiency',color = diloco_color)
l4, = ax_eff.plot(group_centers, eff_2, marker='v', linestyle='--',  linewidth=2, label='DP+TP efficiency', color = ddp_color)
ax_eff.set_ylabel('Scaling Efficiency (%)',fontsize=24)
ax_eff.set_ylim(0, 120)  # 百分比，适当留白

# 合并图例
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
lines = [bars1, bars2, l3, l4]
labels = [l.get_label() for l in lines]
ax_time.legend(lines, labels, loc="upper center", frameon=False, fontsize=24,bbox_to_anchor=(0.5, -0.2),ncol=2)

plt.tight_layout()

ax_time.tick_params(axis='both', labelsize=24)
ax_eff.tick_params(axis='y', labelsize=24)
plt.tight_layout()

# 画完两条效率折线后，做标注
for xi, yi in zip(group_centers, eff_1):
    if np.isfinite(yi):
        ax_eff.text(xi, yi + 2,      # +2 是垂直偏移，可根据百分比比例微调
                    f"{yi:.1f}%",
                    ha='center',
                    va='bottom',
                    color=diloco_color,
                    fontsize=24)

for xi, yi in zip(group_centers, eff_2):
    if np.isfinite(yi):
        ax_eff.text(xi, yi - 2,      # -2 是垂直偏移，可根据百分比比例微调
                    f"{yi:.1f}%",
                    ha='center',
                    va='top',
                    color=ddp_color,
                    fontsize=24)
ymin, ymax = ax_time.get_ylim()
for xi, t1, t2 in zip(group_centers, times_algo1_line, times_algo2):
    print(xi)
    if xi == group_centers[0]:
        continue
    speedup = (t2 / t1) 
    y = ymin + 0.1 * (ymax - ymin)
    ax_time.text(
        xi, y,
        f"{speedup:.1f}x",
        ha='center', va='center',
        fontsize=24,
        color='black'
    )
print(f"group_centers:{group_centers}")
plt.subplots_adjust(left=0.1,right=0.90,top=0.94,bottom=0.34)
plt.savefig("tp_result.png")
