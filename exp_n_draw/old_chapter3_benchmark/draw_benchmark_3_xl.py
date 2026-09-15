import matplotlib.pyplot as plt

dark_orange = "#1f77b4"
light_orange = "#aec7e8"

dark_blue = "#ff7f0e"
light_blue = "#ffbb78"
# 1. 定义 GPU 数量（可随时增删）
gpu_labels = [64, 128, 256]  # 如果以后要加 4、64，只要在这里改列表即可
# 2. 生成等距的 x 位置
x = list(range(len(gpu_labels)))

# 3. 两条曲线的时间开销数据（单位秒）
# 1: DiLoCo 2: DDP
times_algo1 = [1746.94, 1267.894, 919.218] 
times_algo2 =  [3961.222, 3255.924,  2961.51]  

# 基线 GPU 数量
G0 = gpu_labels[0]
T0_1 = times_algo1[0]
T0_2 = times_algo2[0]

# 计算效率
eff_1 = [ (T0_1 * G0) / (t * G) * 100 for t, G in zip(times_algo1, gpu_labels) ]
eff_2 = [ (T0_2 * G0) / (t * G) * 100 for t, G in zip(times_algo2, gpu_labels) ]


fig, ax_time = plt.subplots(figsize=(7,4))


# 绘制柱状图：时间（每次迭代）
width = 0.20
bars1 = ax_time.bar([i - width/2 for i in x], times_algo1, width, label='DiLoCo time',color= light_orange)
bars2 = ax_time.bar([i + width/2 for i in x], times_algo2, width, label='DDP time', color = light_blue)

ax_time.set_xlabel('number of GPU')
ax_time.set_ylabel('time per iteration')
ax_time.set_xticks(x)
ax_time.set_xticklabels(gpu_labels)
ax_time.grid(True, which='both', axis='both', linestyle=':')

# 效率双轴
ax_eff = ax_time.twinx()
l3, = ax_eff.plot(x, eff_1, marker='^', linestyle='--', linewidth=2, label='DiLoCo efficiency',color = dark_orange)
l4, = ax_eff.plot(x, eff_2, marker='v', linestyle='--',  linewidth=2, label='DDP efficiency', color = dark_blue)
ax_eff.set_ylabel('Scaling Efficiency (%)')
ax_eff.set_ylim(0, 110)  # 百分比，适当留白

# 合并图例
lines = [bars1, bars2, l3, l4]
labels = [l.get_label() for l in lines]
ax_time.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=2, frameon=False)
plt.title('Strong Scaling: Time & Efficiency vs. GPU ')
plt.tight_layout()

# 画完两条效率折线后，做标注
for xi, yi in zip(x, eff_1):
    # 算法 A 效率值标在点上方，用 Algo1 效率色 #aec7e8
    ax_eff.text(xi, yi + 2,      # +2 是垂直偏移，可根据百分比比例微调
                f"{yi:.1f}%",
                ha='center',
                va='bottom',
                color=dark_orange,
                fontsize=9)

for xi, yi in zip(x, eff_2):
    # 算法 B 效率值标在点下方，用 Algo2 效率色 #ffbb78
    ax_eff.text(xi, yi - 2,      # -2 是垂直偏移，可根据百分比比例微调
                f"{yi:.1f}%",
                ha='center',
                va='top',
                color=dark_blue,
                fontsize=9)

for xi, t1, t2 in zip(x, times_algo1, times_algo2):
    speedup = (t2 / t1 - 1) * 100
    y = (t2 + t1) / 4  # 垂直居中位置
    ax_time.text(
        xi, y,
        f"speedup: {speedup:.2f}%",
        ha='center', va='center',
        fontsize=9,
        color='black'
    )

plt.savefig("benchmark_3_xl.png")
