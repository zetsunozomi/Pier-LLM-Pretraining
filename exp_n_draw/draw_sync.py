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

def compute_loss_diff(data, start=5, end=10000, interval=5):
    diff_pairs = []
    diffs = []
    for i in range(start, min(end, len(data)), interval):
        diff = data[i] - data[i - 1]
        diff_pairs.append(i)
        diffs.append(diff)
    return diff_pairs, diffs

log_file = "/pscratch/sd/s/syfan/Diloco/exp/exp13_more_warmup/medium_subgroup32.41740783.out"
data = parse_log(log_file)
x, y = compute_loss_diff(data[1])
# 绘图
plt.figure(figsize=(10, 5))
plt.plot(x, y, label="Loss Diff (step+10 - step)", marker='o')
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel("Iteration (step)")
plt.ylabel("Loss Difference (after - before)")
plt.title("Loss Difference Between Steps N and N+10 (Every 50 steps)")
plt.grid(True)
plt.ylim(-1,1)
plt.legend()
plt.tight_layout()
plt.savefig("loss_diff.png")
