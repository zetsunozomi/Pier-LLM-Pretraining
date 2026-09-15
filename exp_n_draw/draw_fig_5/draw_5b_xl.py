import matplotlib.pyplot as plt

diloco_color = "#d34a47"
diloco_light = "#ff9897"

ddp_color = '#074f97'
ddp_light = '#5e95d0'
# 1. 定义 GPU 数量（可随时增删）
gpu_labels = [64, 128,256]  # 如果以后要加 4、64，只要在这里改列表即可
# 2. 生成等距的 x 位置
x = list(range(len(gpu_labels)))

# 3. 两条曲线的时间开销数据（单位秒）

import numpy as np
# 1: DiLoCo 
times_algo1 = np.array([1777.58, 1452.795, 865.9775])/1000
# 2 : DDP
times_algo2 = np.array([4124.96 , 3377.8025, 2971.5])/1000
# 3: MP
times_mp = np.array([4205.195, 3756.2675, 3098.14])/1000

times_algo1 = times_algo1*0.9 + times_mp*0.1

print(f"Ours speed in 64, 128, 256: {times_algo1}")
print(f"baseline speed in 64, 128, 256: {times_algo2}")
# 基线 GPU 数量
G0 = gpu_labels[0]
T0_1 = times_algo1[0]
T0_2 = times_algo2[0]

# 计算效率
eff_1 = [ (T0_1 * G0) / (t * G) * 100 for t, G in zip(times_algo1, gpu_labels) ]
eff_2 = [ (T0_2 * G0) / (t * G) * 100 for t, G in zip(times_algo2, gpu_labels) ]


fig, ax_time = plt.subplots(figsize=(13,6.2))


# 绘制柱状图：时间（每次迭代）
width = 0.20
offset = 0.1
bars1 = ax_time.bar([i - offset for i in x], times_algo1, width, label='MALT Time',color= diloco_light)
bars2 = ax_time.bar([i + offset for i in x], times_algo2, width, label='AdamW time', color = ddp_light)

ax_time.set_xlabel('number of GPU',fontsize=24)
ax_time.set_ylabel('time per iteration (s)',fontsize=24)
ax_time.set_xticks(x)
ax_time.set_xticklabels(gpu_labels)
ax_time.grid(True, which='both', axis='both', linestyle=':')

# 效率双轴
ax_eff = ax_time.twinx()
l3, = ax_eff.plot(x, eff_1, marker='^', linestyle='--', linewidth=2, label='MALT efficiency',color = diloco_color)
l4, = ax_eff.plot(x, eff_2, marker='v', linestyle='--',  linewidth=2, label='AdamW efficiency', color = ddp_color)
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
for xi, yi in zip(x, eff_1):
    ax_eff.text(xi, yi + 2,      # +2 是垂直偏移，可根据百分比比例微调
                f"{yi:.1f}%",
                ha='center',
                va='bottom',
                color=diloco_color,
                fontsize=24)

for xi, yi in zip(x, eff_2):
    ax_eff.text(xi, yi - 2,      # -2 是垂直偏移，可根据百分比比例微调
                f"{yi:.1f}%",
                ha='center',
                va='top',
                color=ddp_color,
                fontsize=24)
ymin, ymax = ax_time.get_ylim()
for xi, t1, t2 in zip(x, times_algo1, times_algo2):
    speedup = (t2 / t1) 
    y = ymin + 0.1 * (ymax - ymin)
    ax_time.text(
        xi, y,
        f"{speedup:.1f}x",
        ha='center', va='center',
        fontsize=24,
        color='black'
    )
plt.subplots_adjust(left=0.1,right=0.9,top=0.9,bottom=0.35)
plt.savefig("benchmark_xl_perlmutter_fixed.png")
