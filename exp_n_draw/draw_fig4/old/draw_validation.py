import re
import matplotlib.pyplot as plt

def read_baseline_log(file_path):
    """解析 baseline 日志 (validation loss at iteration ...)。"""
    steps, losses = [], []
    pattern = re.compile(r'validation loss at iteration (\d+) \| lm loss value: ([\d.Ee+-]+)')
    with open(file_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                steps.append(int(match.group(1)))
                losses.append(float(match.group(2)))
    return steps, losses


def read_experiment_log(file_path, start_step=0, step_interval=1000):
    """解析 experiment 日志 (The loss value is: <class 'float'>, ...)"""
    losses = []
    pattern = re.compile(r"The loss value is: <class 'float'>, ([\d.]+)")
    with open(file_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                losses.append(float(match.group(1)))
    steps = [start_step + 1000 + i * step_interval for i in range(len(losses))]
    # 截掉尾部两个异常点
    return steps[:-2], losses[:-2]


def read_mixed_log(file_path1, file_path2, split_step, total_step, step_interval=1000):
    """
    前半部分用 baseline，后半部分用 experiment。
    Args:
        file_path1: baseline log
        file_path2: experiment log
        split_step: 从多少步开始切换
        total_step: 总步数
    """
    steps1, loss1 = read_baseline_log(file_path1)
    steps2, loss2 = read_experiment_log(file_path2, start_step=split_step, step_interval=step_interval)
    # 拼接
    steps = steps1 + steps2
    losses = loss1 + loss2
    # 如果超出总步数则截断
    combined = [(s, l) for s, l in zip(steps, losses) if s <= total_step]
    steps, losses = zip(*combined)
    return list(steps), list(losses)


# ---------------- 主程序 ----------------

# 配置：比例不变 (1/5)
configs = [
    {
        "label": "25000-step",
        "baseline_log": "/pscratch/sd/s/syfan/Diloco/exp/chapter2_weak_scaling_exp/exp2_subgroup32_our_impl_WS/log_merge.log",
        "experiment_log": "/pscratch/sd/s/syfan/Diloco/exp/chapter2_weak_scaling_exp/exp2_subgroup32_our_impl_WS/log_merge.log",
        "split_step": 5000,
        "total_step": 25000
    },
    {
        "label": "50000-step",
        "baseline_log": "/pscratch/sd/s/syfan/Diloco/exp/exp4/log.log",
        "experiment_log": "/pscratch/sd/s/syfan/Diloco/exp/exp4/log.log",
        "split_step": 10000,
        "total_step": 50000
    },
    {
        "label": "100000-step",
        "baseline_log": "/pscratch/sd/s/syfan/Diloco/exp/chapter2_weak_scaling_exp/sophia_sub8.log",
        "experiment_log": "/pscratch/sd/s/syfan/Diloco/exp/chapter2_weak_scaling_exp/sophia_sub8.log",
        "split_step": 20000,
        "total_step": 100000
    },
    {
        "label": "200000-step",
        "baseline_log": "/pscratch/sd/s/syfan/Diloco/exp/chapter2_weak_scaling_exp/polaris_7_sub4.log",
        "experiment_log": "/pscratch/sd/s/syfan/Diloco/exp/chapter2_weak_scaling_exp/polaris_7_sub4.log",
        "split_step": 40000,
        "total_step": 200000
    },
]

plt.figure(figsize=(12, 7))

for cfg in configs:
    steps, losses = read_mixed_log(
        cfg["baseline_log"],
        cfg["experiment_log"],
        cfg["split_step"],
        cfg["total_step"]
    )
    print(len(steps))
    plt.plot(steps, losses, label=cfg["label"], marker='o')

plt.xlabel('Step')
plt.ylabel('Validation Loss')
plt.title('Validation Loss Comparison (Dynamic Range)')
plt.ylim(2.9, 3.4)
plt.legend()
plt.grid(True)

plt.savefig('validation_WS_dynamic_full.png')
