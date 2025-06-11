Apr 12:

Changed the implementation of create subgroup： no longer uses mydistributed.py, but use parallel_state to create the subgroup.

Problem1: The validation loss seems not divided into 2 groups correctly.

Problem2: The nccl error makes the program can't run for the second time.

May 23:

Need to use DiLoCo Baseline setting.

Model: 150M 
    We use default GPT-2 with 124M
Batchsize: 512 
    Note that it's for DDP baseline. As we have mentioned before, opendiloco uses 512 * 8 for diloco, but use 512 as global batch size for ddp. it's unfair and seems like a mistake.
    And it's confusing in the original paper.
    Since [512, 150M, 88000 steps, 16 ppl] is a set of parameters that makes sense, We will set baseline 512, ddp 4096, 
    diloco 4096 (512 for each) as our setting.
    Then we will definitely need to recover the training from checkpoint of outer optimizer states.

Settings for single GPU Baseline (We will still use 8 gpus to do it faster):
Model: 150M in paper, we use 124 default gpt-2
Dataset: C4 in paper, we use openwebtext
Batchsize: 512 batchsize in 1(8 acutally) gpu
Lr: 4e-4
warmup: 1000
Decay 0.1
Total step: 88000 in paper, we use 100000 steps

Settings for DDP GPU Baseline:
Model: 150M in paper, we use 124 default gpt-2
Dataset: C4 in paper, we use openwebtext
Batchsize: 4096 batchsize in 8 gpu
Lr: 4e-4 * 8
warmup: 2000
Decay 0.1
Total step: 88000 steps

模型参数对不上，以及数据集openwebtext vs C4的差距，会导致实验对不上：
NanoGPT 500左右的bsz + 100000步直接收敛，最低loss 3.11
paper里 500左右的bsz + 88000步能做到 16.23 的ppl 相当于loss 2.77
于此同时4096左右的bsz + 88000步能做到15.3的ppl，相当于 loss 2.73
也就是说paper里的所有数都只能稍作参考。
我还是用了100000步，把warmup翻倍到2000

6/11:
Diloco和DDP同样步数得到的模型性能非常接近。
1. 实现3D DiLoCo
2. 实验不同的replica数量：4，16，32会不会有某种趋势
3. 用两个node profile节省的时间