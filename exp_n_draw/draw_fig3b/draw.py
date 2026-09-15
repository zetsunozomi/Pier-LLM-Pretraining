import re
import matplotlib.pyplot as plt


def read_baseline_log(file_path):
    """
    Parses validation loss values and corresponding iteration steps from a baseline log file.
    """
    steps = []
    losses = []

    pattern = re.compile(
        r'validation loss at iteration (\d+) \| lm loss value: ([\d.Ee+-]+)'
    )

    with open(file_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                steps.append(int(match.group(1)))
                losses.append(float(match.group(2)))

    return steps, losses


def read_experiment_log(file_path, start_step=0, step_interval=1000):
    """
    Parses validation loss values from an experiment log file and generates corresponding steps.
    """
    losses = []

    pattern = re.compile(
        r"The loss value is: <class 'float'>, ([\d.Ee+-]+)"
    )

    with open(file_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                losses.append(float(match.group(1)))

    steps = [start_step + (i + 1) * step_interval for i in range(len(losses))]

    return steps, losses


# 1. Define file paths
baseline_file = '/pscratch/sd/s/syfan/Pier/exp_n_draw/draw_fig3b/xl_ddp.log'
diloco_file = '/pscratch/sd/s/syfan/Pier/exp_n_draw/draw_fig3b/xl_diloco.log'
pier_file = '/pscratch/sd/s/syfan/Pier/exp_n_draw/draw_fig3b/vista_new_run1/merge.txt'


# 2. Read and parse data
baseline_steps, baseline_loss = read_baseline_log(baseline_file)

diloco_steps, diloco_loss = read_experiment_log(
    diloco_file,
    start_step=0
)

pier_steps, pier_loss = read_experiment_log(
    pier_file,
    start_step=0
)


# 3. Remove last two points, same as before
diloco_steps = diloco_steps[:-2]
diloco_loss = diloco_loss[:-2]

pier_steps = pier_steps[:-2]
pier_loss = pier_loss[:-2]


print("AdamW:", len(baseline_steps))
print("DiLoCo:", len(diloco_steps))
print("Pier:", len(pier_steps))


# 4. Plotting
plt.figure(figsize=(12, 5.5))

plt.plot(
    baseline_steps,
    baseline_loss,
    label='AdamW',
    marker='s',
    color='#5e95d0'
)

plt.plot(
    diloco_steps,
    diloco_loss,
    label='DiLoCo',
    marker='s',
    color='#ebb27b'
)

plt.plot(
    pier_steps,
    pier_loss,
    label='MALT',
    marker='s',
    color='#d34a47'
)

plt.xlabel('Iteration', fontsize=24)
plt.ylabel('Validation Loss', fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)

plt.ylim(2.4, 3.6)
plt.legend(fontsize=24)
plt.grid(True)

plt.subplots_adjust(left=0.09, right=1-0.02, top=0.97, bottom=0.15)


# 5. Save the chart
output_filename = 'figure6_1_XL.png'
plt.savefig(output_filename, dpi=300)
plt.close()