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
    return list(steps), list(losses)


# ---------------- 主程序 ----------------

# 配置：比例不变 (1/5)
configs = [
    {
        "label": "25000-step",
        "baseline_log": "/pscratch/sd/s/syfan/Pier/exp/draw_fig4/miu_exp_weak32.53590195.out",
        "experiment_log": "/pscratch/sd/s/syfan/Pier/exp/draw_fig4/miu_exp_weak32.53590195.out",
        "split_step": 2500,
        "total_step": 25000
    },
    {
        "label": "50000-step",
        "baseline_log": "/pscratch/sd/s/syfan/Pier/exp/draw_fig4/miu_exp_weak16.53589915.out",
        "experiment_log": "/pscratch/sd/s/syfan/Pier/exp/draw_fig4/miu_exp_weak16.53589915.out",    
        "split_step": 5000,
        "total_step": 50000
    },
    {
        "label": "100000-step",
        "baseline_log": "/pscratch/sd/s/syfan/Pier/exp/draw_fig3a/miu_exp1_50_rerun_rerun.55349364.out",
        "experiment_log": "/pscratch/sd/s/syfan/Pier/exp/draw_fig3a/miu_exp1_50_rerun_rerun.55349364.out",
        "split_step": 10000,
        "total_step": 100000
    },
    {
        "label": "200000-step",
        "baseline_log": "/pscratch/sd/s/syfan/Pier/exp/draw_fig4/miu_exp_weak4.53589508.out",
        "experiment_log": "/pscratch/sd/s/syfan/Pier/exp/draw_fig4/miu_exp_weak4.53589508.out",
        "split_step": 20000,
        "total_step": 200000
    },
]

plt.figure(figsize=(12, 5.5))
# -------------------------------------32gpu
steps, losses = read_mixed_log(
    configs[0]["baseline_log"],
    configs[0]["experiment_log"],
    configs[0]["split_step"],
    configs[0]["total_step"]
)
print(len(steps))
plt.plot(steps, losses, label='32 gpus', marker='s',color='#ebb17b')
# ---------------------------------------16 gpu
steps, losses = read_mixed_log(
    configs[1]["baseline_log"],
    configs[1]["experiment_log"],
    configs[1]["split_step"],
    configs[1]["total_step"]
)
print(len(steps))
plt.plot(steps, losses, label='16 gpus', marker='s',color='#9c6b69')
# --------------------------------------8gpu
steps, losses = read_mixed_log(
    configs[2]["baseline_log"],
    configs[2]["experiment_log"],
    configs[2]["split_step"],
    configs[2]["total_step"]
)
print(len(steps))
plt.plot(steps, losses, label='8 gpus', marker='s', color='#d34a47')
# ----------------------------------4 gpu
steps, losses = read_mixed_log(
    configs[3]["baseline_log"],
    configs[3]["experiment_log"],
    configs[3]["split_step"],
    configs[3]["total_step"]
)
print(len(steps))
plt.plot(steps, losses, label='4 gpus', marker='s',color='#91bffa')


plt.xticks(range(0, 200001, 50000)) # 1 tick per 50000
plt.locator_params(axis='y', nbins=5)

plt.xlabel('Iteration',fontsize=24)
plt.ylabel('Validation Loss',fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
#plt.title('Validation Loss Comparison (Dynamic Range)')
plt.ylim(2.9, 3.6)
plt.legend(fontsize=24)
plt.grid(True)
plt.subplots_adjust(left=0.11,right=1-0.02,top=0.97,bottom=0.15)

plt.savefig('figure4.png')
