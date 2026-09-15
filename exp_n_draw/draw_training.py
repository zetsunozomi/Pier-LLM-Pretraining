import re
import matplotlib.pyplot as plt

def parse_log(file_path):
    pattern = re.compile(r"iteration\s+(\d+)/\s*\d+\s+\|.*?lm loss:\s+([0-9.Ee+-]+)")
    iterations = []
    losses = []
    with open(file_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                iterations.append(int(match.group(1)))
                losses.append(float(match.group(2)))
    return iterations, losses

# 替换成你的日志文件路径
log_file_1 = "/pscratch/sd/s/syfan/Diloco/exp/sophia_medium_megatron_exp4(new_baseline)/log.txt"
log_file_2 = "/pscratch/sd/s/syfan/Diloco/exp/exp14_scratch/medium_subgroup32.41784708.out"
log_file_3 = "/pscratch/sd/s/syfan/Diloco/exp/exp15/medium_subgroup32.41877801.out"
log_file_4 = "/pscratch/sd/s/syfan/Diloco/exp/exp16_subgroup128/subgroup128_pretraining_exp1.42066750.out"
# 读取两个日志
iters1, loss1 = parse_log(log_file_1)
iters2, loss2 = parse_log(log_file_2)
iters3, loss3 = parse_log(log_file_3)
iters4, loss4 = parse_log(log_file_4)
# 绘图
plt.figure(figsize=(10, 5))
plt.plot(iters1, loss1, label="Medium-DDP-newbaseline", linestyle='--')
plt.plot(iters2, loss2, label="Medium-DiLoCo-exp14(sub32-from scratch)", linestyle='--')
plt.plot(iters3, loss3, label="Medium-DiLoCo-exp15(sub32-trick tuned)", linestyle='--')
plt.plot(iters4, loss4, label="Medium-DiLoCo-exp16(sub128)", linestyle='--')

plt.xlabel("Iteration")
plt.ylabel("LM Loss")
plt.xlim(0,100000)
plt.ylim(2.7,3.5)
plt.title("Training Loss Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("medium_fig.png")
