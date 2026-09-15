import re
import matplotlib.pyplot as plt
import numpy as np
def extract_time_per_iteration(file_path):
    """
    从日志文件中提取每次迭代的 elapsed time per iteration (ms)
    返回一个包含浮点数的列表
    """
    times = []
    pattern = r"elapsed time per iteration \(ms\):\s*([\d.]+)"
    with open(file_path, "r", encoding="utf-8") as f:  # 相当于 fopen
        for line in f:
            match = re.search(pattern, line)
            if match:
                times.append(float(match.group(1)))
    return times
def compute_interval_means(times, base_interval=50):
    """
    自动根据 base_interval 决定 offset 组数：
      - base_interval=50 → offsets=[0,10,20,30,40] (5组)
      - base_interval=100 → offsets=[0,10,...,90] (10组)
    """
    # 自动确定 offsets
    num_offsets = base_interval // 10
    offsets = [i for i in range(num_offsets)]
    
    means = []
    for offset in offsets:
        indices = list(range(offset, len(times), base_interval))
        group = [times[i] for i in indices if i < len(times)]
        means.append(np.mean(group) if group else np.nan)
    return offsets, means


def main():
    file_path = "50.log"  # 你的日志文件路径
    times = extract_time_per_iteration(file_path)
    offsets, means1 = compute_interval_means(times, base_interval=500)
    plt.figure(figsize=(8, 5))
    offsets1 = [i*10+10 for i in offsets]
    plt.plot(offsets1, means1, label = '50', marker='o', linewidth=2)

    file_path = "100.log"  # 你的日志文件路径
    times = extract_time_per_iteration(file_path)
    offsets, means2 = compute_interval_means(times, base_interval=500)
    offsets2 = [i*10+10 for i in offsets]
    plt.plot(offsets2, means2, label = '100',marker='o', linewidth=2)


    file_path = "200.log"  # 你的日志文件路径
    times = extract_time_per_iteration(file_path)
    offsets, means3 = compute_interval_means(times, base_interval=500)
    offsets3 = [i*10+10 for i in offsets]
    plt.plot(offsets3, means3, label = '200',marker='o', linewidth=2)

    file_path = "500.log"  # 你的日志文件路径
    times = extract_time_per_iteration(file_path)
    offsets, means4 = compute_interval_means(times, base_interval=500)
    offsets4 = [i*10+10 for i in offsets]
    plt.plot(offsets4, means4, label = '500',marker='o', linewidth=2)

    plt.title("Average iteration time (interval = 50)")
    plt.xlabel("Offset (iteration index start)")
    plt.ylabel("Average elapsed time (ms)")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig('show_all.png')

if __name__ == "__main__":
    main()
