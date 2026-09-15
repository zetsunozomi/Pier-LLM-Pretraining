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
    前半部分用 baseline（step <= split_step），后半部分用 experiment（step <= total_step）。
    Args:
        file_path1: baseline log
        file_path2: experiment log
        split_step: 从多少步开始切换
        total_step: 总步数，超过此步数的数据会被截断
    """
    steps1, loss1 = read_baseline_log(file_path1)
    steps2, loss2 = read_experiment_log(file_path2, start_step=split_step, step_interval=step_interval)

    # 只保留 baseline 中 step <= split_step 的部分
    filtered1 = [(s, l) for s, l in zip(steps1, loss1) if s <= split_step]
    # 只保留 experiment 中 split_step < step <= total_step 的部分
    filtered2 = [(s, l) for s, l in zip(steps2, loss2) if split_step < s <= total_step]

    steps = [x[0] for x in filtered1] + [x[0] for x in filtered2]
    losses = [x[1] for x in filtered1] + [x[1] for x in filtered2]
    return steps, losses



# 1. Define file paths
baseline_file = '/pscratch/sd/s/syfan/Pier/exp_n_draw/draw_fig3a/ddp_8gpu_small.log'
file2 = '/pscratch/sd/s/syfan/Pier/exp_n_draw/draw_fig3a/diloco_baseline.log'

# 2. Read and parse data
baseline_steps, baseline_loss = read_baseline_log(baseline_file)
steps2, loss2 = read_experiment_log(file2,start_step=0)

steps2 = steps2[:-2]
loss2 = loss2[:-2]
# 3. Plotting
plt.figure(figsize=(12, 5.5))
# --- Main Program ---
pier_log = "/pscratch/sd/s/syfan/Pier/exp_n_draw/draw_fig3a/miu_exp1_50_rerun_rerun.55349364.out"
# Pier log 只有 "The loss value is" 格式，eval_interval=1000，从 step 1000 开始
pier_steps, pier_losses = read_experiment_log(pier_log, start_step=0, step_interval=1000)
# 截断到 100000 步以内
pier_steps, pier_losses = zip(*[(s, l) for s, l in zip(pier_steps, pier_losses) if s <= 100000])
pier_steps, pier_losses = list(pier_steps), list(pier_losses)

print(baseline_steps)
print(steps2)
print(f"Pier: {len(pier_steps)} validation points, step range [{pier_steps[0]}, {pier_steps[-1]}]")

plt.plot(baseline_steps, baseline_loss, label='AdamW', marker='s', color='#5e95d0')
plt.plot(steps2, loss2, label='DiLoCo', marker='s', color='#ebb27b')
plt.plot(pier_steps, pier_losses, label='MALT', marker='s', color='#d34a47')

plt.xlabel('Iteration',fontsize=24)
plt.ylabel('Validation Loss',fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
#plt.title('Validation Loss Comparison (Dynamic Range)')
plt.ylim(2.9, 3.5)
plt.legend(fontsize=24)
plt.grid(True)
plt.subplots_adjust(left=0.09,right=1-0.02,top=0.97,bottom=0.15)
# 5. Save the chart
output_filename = 'figure6_1_small.png'
plt.savefig(output_filename)