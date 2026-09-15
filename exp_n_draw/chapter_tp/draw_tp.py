import matplotlib.pyplot as plt

ddp_color = '#ff9900'
diloco_color = "#d24d52"
colors=[ddp_color, diloco_color]

# 数据和标签
values = [18070.80769, 8215.01625]
labels = ['DDP', 'Pier']

plt.figure(figsize=(10, 8))  # 可以适当放大画布，不然243的字会挤爆

bars = plt.bar(labels, values, color=colors)

# 自动算 gain
t1, t2 = values
gain = t1 / t2
gain_text = f'Speedup: {gain:.1f}x'

# 在图上标注，放在柱子上方中间偏上
max_height = max(values)
plt.text(0.5, max_height , gain_text,
         ha='center', fontsize=24)

# 图例
plt.legend(bars, labels, fontsize=24)

# 坐标轴标签和标题字体大小
plt.ylabel('time per iteration', fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)

plt.savefig("tp_result.png", dpi=300, bbox_inches='tight')