import re
import matplotlib.pyplot as plt

def read_baseline_log(file_path):
    """
    Parses validation loss values and corresponding iteration steps from a baseline log file.

    Args:
        file_path (str): The path to the log file.

    Returns:
        tuple: A tuple containing two lists (steps, losses).
    """
    steps = []
    losses = []
    # Regex for lines like 'validation loss at iteration 1000 | lm loss value: 3.32628E+00'
    pattern = re.compile(r'validation loss at iteration (\d+) \| lm loss value: ([\d.Ee+-]+)')
    
    with open(file_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                steps.append(int(match.group(1)))
                losses.append(float(match.group(2)))
                
    return steps, losses

def read_experiment_log(file_path, start_step=20000, step_interval=1000):
    """
    Parses validation loss values from an experiment log file and generates corresponding steps.

    Args:
        file_path (str): The path to the log file.
        start_step (int): The initial iteration step.
        step_interval (int): The interval between steps.

    Returns:
        tuple: A tuple containing two lists (steps, losses).
    """
    losses = []
    # Regex for lines like 'The loss value is: <class 'float'>, 3.16015625'
    pattern = re.compile(r"The loss value is: <class 'float'>, ([\d.]+)")
    
    with open(file_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                losses.append(float(match.group(1)))
    
    steps = [start_step + (i+1) * step_interval for i in range(len(losses))]
    return steps, losses
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
    steps = (steps1 + steps2)[:-2]
    losses = (loss1 + loss2)[:-2]
    return list(steps), list(losses)



# 1. Define file paths
baseline_file = '/pscratch/sd/s/syfan/Megatron-LM/4096_merge.txt'
file2 = '/pscratch/sd/s/syfan/Pier/llama_32gpu_pier_32subgroup_4096.48433767.out'
file3 = '/pscratch/sd/s/syfan/Pier/llama_32gpu_pier_32subgroup_4096_diloco.48773067.out'
# 2. Read and parse data
baseline_steps, baseline_loss = read_baseline_log(baseline_file)
steps2, loss2 = read_experiment_log(file2,start_step=0)
steps3, loss3 = read_experiment_log(file3,start_step=0)

steps2 = steps2[:-2]
loss2 = loss2[:-2]

steps3 = steps3[:-2]
loss3 = loss3[:-2]
# 3. Plotting
plt.figure(figsize=(12, 5.5))
# --- Main Program ---
configs = [
    {
        "label": "100000-step",
        "baseline_log": "/pscratch/sd/s/syfan/Pier/llama_32gpu_pier_32subgroup_4096.48433767.out",
        "experiment_log": "/pscratch/sd/s/syfan/Pier/llama_32gpu_pier_32subgroup_4096.48433767.out",
        "split_step": 6000,
        "total_step": 60000
    },

]
print(baseline_steps)
print(steps2)
plt.plot(baseline_steps, baseline_loss, label='AdamW', marker='s', color = '#FFC145', )
plt.plot(steps2, loss2, label='Pier', marker='s', color = '#5B5F97')
plt.plot(steps3, loss3, label='DiLoCo', marker='s', color = '#B8B8D1')
'''
for cfg in configs:
    steps, losses = read_mixed_log(
        cfg["baseline_log"],
        cfg["experiment_log"],
        cfg["split_step"],
        cfg["total_step"]
    )
    print(steps)
    plt.plot(steps, losses ,label='Pier', marker='s', color = '#d34a47')
'''
plt.xlabel('Iteration',fontsize=24)
plt.ylabel('Validation Loss',fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
#plt.title('Validation Loss Comparison (Dynamic Range)')
plt.ylim(2, 4)
plt.legend(fontsize=24)
plt.grid(True)
plt.subplots_adjust(left=0.09,right=1-0.02,top=0.97,bottom=0.15)
# 5. Save the chart
output_filename = 'llama.png'
plt.savefig(output_filename)