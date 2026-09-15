import matplotlib.pyplot as plt
import numpy as np

# 数据
labels = ['32gpu-345M', '32node-1.5B']
# 每组包含两个柱
data1 = [880.95, 1113.11]  # ddp
data2 = [384.4575, 3377.8025]  # diloco

x = np.arange(len(labels))  # 横坐标位置
width = 0.35  # 每个柱的宽度

fig, ax = plt.subplots(figsize=(6, 4))

# 绘制柱状图
rects1 = ax.bar(x - width/2, data1, width, label='ddp', color='orange')
rects2 = ax.bar(x + width/2, data2, width, label='diloco', color='steelblue')

# 添加标签与标题
ax.set_ylabel('Time')
ax.set_xlabel('Config')
ax.set_title('Scaling')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# 显示数值标签
ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

plt.tight_layout()
plt.savefig('scaling.png')