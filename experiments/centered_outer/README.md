# First cluster handoff: E0a correctness gate

本轮停在第一次必须使用真实 GPU 验证的位置。推荐先用 **A100 一节点四卡**，
不下载模型、不需要数据集、不提交 32/128 GPU 性能实验。40G/80G 均可运行小 fixture。
本轮没有完成整个 E0，更不能填论文的性能表。

## 已接入的修改

- 两条现有 Nesterov outer 分支共享 `centered_allreduce_update`，更新 FP32 masters 后
  调用 `commit_outer_update`，使普通 mixed-precision 模型在下一次 forward 看到新值。
  FP32 alias 和 ChainedOptimizer 也有提交接口。
- 新增 `--local-sgd-inner-average`。显式启用时，dense/CP1/普通 optimizer 的
  MyDDP 按 inner subgroup 归一化。旧配方默认仍采用 full-DP divisor；lazy-start
  补偿与所选语义一致。历史 AVG 路径也使用显式 divisor，避免子组平均的断言冲突。
- 尚无 production distributed-optimizer ownership/model-gather adapter；有 outer
  同步的训练会在开始前拒绝该配置，避免把错误坐标当成同一个参数更新。
- 现有单 slot reference executor 和独立 NumPy oracle 已收入本仓库，集群上无需
  论文目录。executor/spec/verify 初始来源为论文仓库 2026-09-15 的 reference_collective。
  源码原样复制；每次 run 记录实际源码 hash。

## 同步及提交

用你通常的 git add/commit/push 和集群 pull 流程同步本轮文件。然后在集群上：

```bash
cd /你的集群路径/Pier
# 先激活原来能够运行这个 Pier checkout 的 Python/CUDA 环境。
sbatch experiments/centered_outer/e0.sbatch
```

如果集群要求 account、partition、qos 或 constraint，在同一条 sbatch 命令添加
原来作业使用的参数。脚本不猜测这些站点配置，也不自动安装/升级 PyTorch、Apex 或 TE。
默认继承提交时的已激活环境；可用 `export PIER_PYTHON=/完整路径/bin/python` 指定。
默认工作目录为 `SLURM_SUBMIT_DIR`，所以需要从仓库根目录提交；也可设置 `PIER_ROOT`。

可选：GH200 每节点一卡，用四节点跑同一 gate：

```bash
export PIER_GPUS_PER_NODE=1
sbatch --nodes=4 --gpus-per-node=1 experiments/centered_outer/e0.sbatch
```

推荐先提交 A100 一次。默认 walltime 是 20 分钟的上限，不是耗时预测。
输出目录使用 job ID，拒绝复用旧结果目录；不会调用 sbatch 提交其他作业。

## 三个阶段

1. **megatron-dp1**：真实 MyDDP + 真实 Float16OptimizerWithFloat16Params/AdamW。
2. **megatron-dp2**：每 learner 两个复制副本；这里仍是普通 optimizer，
   **不是 distributed optimizer**。外层 helper 以固定 inner 坐标跨 learner 建组。
3. **protocol**：独立单 slot CUDA/NCCL executor，对照 NumPy 固定图 oracle。

前两个阶段每 rank 检查：

- SUM/prescale、AVG 两种梯度 collective；旧 full-DP divisor 与新 inner average；
  两次梯度累积和 lazy-start 全局平均。
- outer shard × CPU storage 四种组合；FP32 masters、momentum、BF16 model 和
  下一次真实 forward 的 bits；inner AdamW moments/step 不被 outer 改动。
- 在真实 mixed-precision optimizer 上人为省略提交，必须观察到 stale-model
  负对照，再验证修复；并做本地 optimizer snapshot 恢复检查。

四 GPU protocol 检查 48 个全局 operator 配置、3 个 toy-training 配置：
`s=1/2/4`、非整齐长度、两种 tile、device/host outer storage、连续更新、
非零 momentum、signed zero/cancellation、不等到达、固定缓冲区和 next forward。
每 rank 的重复记录不是独立实验次数。tensor payload 不是物理 wire bytes。

## 结果和交回内容

输出目录：`local/centered_outer/e0-<jobid>/`，已被现有 `.gitignore` 的 `local/` 排除。

- `summary.json`：只有所有节点、rank、阶段及矩阵完整并通过才是 `passed`。
- `manifest.json`：源码 hash、git 状态、Python、设备数、明确的未验证项。
- `node-*.json`：GPU/CUDA/NCCL 和 `nvidia-smi topo -m` 等只读采集。
- `megatron-dp1.log`、`megatron-dp2.log`、`protocol.log`：完整阶段日志。
- `megatron-dp*/rank-*.json`：每 rank 结果；失败时尽量保存 traceback。
- `protocol.json`：固定图、storage、payload 和 toy training 细目。

作业失败会保留非零退出码，并尝试输出失败 summary。作业被强制杀死、环境连 Python
都无法启动等情况可能没有 summary；此时交回 Slurm 输出，不能视为通过。

**运行后给回整个结果目录和 `pier-e0-<jobid>.out`。** 可打包后放回本机；
无需把 `.pt` fixture 纳入 git。不要只发 `passed`，后续需要实际版本和逐 rank 证据。

## 本地检查

```bash
python -m unittest discover -s tests/outer_sync -v
bash -n experiments/centered_outer/e0.sbatch experiments/centered_outer/node_entry.sh
# 需要本机 loopback socket；只验证 CPU/Gloo，不替代 GPU gate。
python experiments/centered_outer/reference/verify_executor.py \
  --world-size 4 --device cpu --output /tmp/pier-e0-cpu4.json
```

本轮本地检查详情见 `LOCAL_VALIDATION.md`。无需在集群上重跑本地 CPU 检查。

## 必须等反馈后再接的内容

生产 Qwen 坐标 adapter、successful-step/skip 同步调度、完整 per-learner checkpoint、
distributed optimizer、并发 slots、强 G/R/T/O/S/W 基线及性能计量仍在后续计划。
当前 pretrain_gpt 仍保留历史 iteration 同步时机，不能直接拿它运行 draft 的
500-step 成功步配方。当前 gate 不下载 Qwen、不开完整训练；fixture snapshot 不等于
production checkpoint，独立 executor 也不等于已经接通 Megatron 数据流。

GPU gate 若失败，下一轮先按日志修复；若通过，再接生产 adapter 和统一计数/恢复，
随后推进公平的完整周期性能比较。
