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
    
    steps = [start_step + i * step_interval for i in range(len(losses))]
    return steps, losses

# --- Main Program ---

# 1. Define file paths
file1 = '/pscratch/sd/s/syfan/Diloco/exp/chapter_4_frequency/50.log'
file2 = '/pscratch/sd/s/syfan/Diloco/exp/chapter_4_frequency/100.log'
file3 = '/pscratch/sd/s/syfan/Diloco/exp/chapter_4_frequency/200.log'
file4 = '/pscratch/sd/s/syfan/Diloco/exp/chapter_4_frequency/500.log'
#file5 = '/lus/eagle/projects/Local-LLM/shuyuanfan/Diloco/gpt2-small-logs/perlmutter_exp4/pretrain-GPT2small.40885700.out'
#file6 = '/lus/eagle/projects/Local-LLM/shuyuanfan/Diloco/gpt2-small-logs/exp31_subgroup2_weakscale/merge.txt'
# 2. Read and parse data
steps1, loss1 = read_experiment_log(file1,start_step=10000)
steps2, loss2 = read_experiment_log(file2,start_step=10000)
steps3, loss3 = read_experiment_log(file3,start_step=10000)
steps4, loss4 = read_experiment_log(file4, start_step=10000)
#steps5, loss5 = read_experiment_log(file5, start_step=10000)
#steps6, loss6 = read_experiment_log(file6, start_step=80000)
# 3. Plotting
plt.figure(figsize=(12, 7))

plt.plot(steps1, loss1, label='50', marker='s')
plt.plot(steps2, loss2, label='100', marker='o')
plt.plot(steps3, loss3, label='200', marker='^')
plt.plot(steps4, loss4, label='500 (500 didnt gain performance)', marker='x')
#plt.plot(steps5, loss5, label='subgroup16 (with lazy start)', marker='x')
#plt.plot(steps6, loss6, label='subgroup2 (with lazy start)', marker='x')

# 4. Set chart properties
plt.xlabel('Step')
plt.ylabel('Validation Loss')
plt.title('Validation Loss Comparison')
plt.ylim(2.9, 3.4)
plt.legend()
plt.grid(True)

# 5. Save the chart
output_filename = 'validation.png'
plt.savefig(output_filename)