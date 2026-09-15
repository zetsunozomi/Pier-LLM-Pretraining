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
    return steps[:-2], losses[:-2]

# --- Main Program ---

# 1. Define file paths
baseline_file = '/pscratch/sd/s/syfan/Megatron-LM/llama_32gpu_5e4_5e5.47762164.out'
file2 = '/pscratch/sd/s/syfan/Pier/llama_32gpu_5e4_5e5_32subgroup_200.47762116.out'
file3 = '/pscratch/sd/s/syfan/Pier/llama_32gpu_5e3_5e5_32subgroup_50.47942700.out'
file4 = '/pscratch/sd/s/syfan/Pier/llama_32gpu_5e3_5e5_16subgroup_50.47942702.out'
file5 = '/pscratch/sd/s/syfan/Pier/llama_32gpu_5e3_5e4_8subgroup_50.48015877.out'
file6 = '/pscratch/sd/s/syfan/Pier/llama_32gpu_5e4_5e5_nowarmup_32subgroup_50_run6.48107978.out'
# 2. Read and parse data
baseline_steps, baseline_loss = read_baseline_log(baseline_file)
steps2, loss2 = read_experiment_log(file2,start_step=0)
steps3, loss3 = read_experiment_log(file3,start_step=0)
steps4, loss4 = read_experiment_log(file4, start_step=0)
steps5, loss5 = read_experiment_log(file5, start_step=0)
steps6, loss6 = read_experiment_log(file6, start_step=0)
print(len(baseline_steps))
print(len(steps2))
# 3. Plotting
plt.figure(figsize=(12, 7))

plt.plot(baseline_steps, baseline_loss, label='ddp', marker='s')
plt.plot(steps2, loss2, label='sub32-oldlr_50', marker='o')
plt.plot(steps3, loss3, label='sub32-newlr_50', marker='^')
plt.plot(steps4, loss4, label='sub16-newlr_50', marker='x')
plt.plot(steps5, loss5, label='sub8-newlr_50', marker='x')
plt.plot(steps6, loss6, label='no_warmup', marker='x')
# 4. Set chart properties
plt.xlabel('Step')
plt.ylabel('Validation Loss')
plt.title('Validation Loss Comparison')
plt.ylim(1, 6)
plt.legend()
plt.grid(True)

# 5. Save the chart
output_filename = 'llama.png'
plt.savefig(output_filename)