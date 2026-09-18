# E0b：真实 pretrain_gpt 接入验证

状态：`e0b-58457867-20260917-021226` **集群验收通过**，八阶段完成，
四 rank 的 s2/s4/host/resume 对 s1 共 16 项轨迹比较逐位一致，errors 为空。
75 个文件已回传，本地对 32 份 rank 报告、启动记录及 390 个源码哈希的只读复核
也通过。本轮无需重跑或再次回传，当前进入 Qwen 接入与验证准备。范围与后续见
[E0b 结果审查](E0B_REVIEW_58457867.md)。
本轮目标是验证生产训练接入；不会产生可用于论文性能比较的数据。

## 复现实验时的提交方式

本地正常 `git add / commit / push`，集群正常 `git pull`。
在仓库根目录激活原来的 `diloco` 环境后：

```bash
sbatch experiments/centered_outer/e0b.sbatch
```

已有 1 节点 4 GPU 的 Slurm 交互式分配时，在集群仓库根目录运行：

```bash
conda activate /pscratch/sd/s/syfan/conda/envs/diloco
export PIER_PYTHON=/pscratch/sd/s/syfan/conda/envs/diloco/bin/python
PIER_ROOT="$PWD" \
SLURM_OVERLAP=1 \
PIER_E0B_RUN_DIR="$PWD/local/centered_outer/e0b-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)" \
bash experiments/centered_outer/e0b.sbatch
```

`bash` 不处理 `#SBATCH` 资源申请，使用现有分配和剩余时间；
`SLURM_OVERLAP=1` 允许内部 srun 与交互式 shell 的 step 共享分配。
该 Python 路径来自已通过的 E0a manifest；默认 `nersc-python` 本次没有 torch。
修复后必须使用新输出目录重跑整套，避免混用不同源码的阶段证据。

- NERSC account `m4431`、qos `regular`、constraint `gpu`。
- 1 节点，1 Slurm task，4 GPU；task 内 torchrun 启动 4 ranks。
- walltime 01:00:00；这是上限，实际运行时间待测。
- 每个 job 使用新目录，拒绝覆盖旧结果；仅用户提交这一份 job。
- 默认继承提交时的 Python；可设置 `PIER_PYTHON`。
- 使用仓库已有的 Megatron 数据索引 helper 编译流程；不自动安装或升级软件。
- 不需要下载模型、tokenizer 或数据集；不需要修改数据路径。

## 固定配方及矩阵

实际入口是仓库根目录 `pretrain_gpt.py`，不是独立 toy optimizer。
两层 GPT、hidden 64、FFN 128、4 heads、sequence 64，BF16 model、
FP32 masters/梯度/outer state、AdamW、dropout 0.1、两次梯度累积。
NullTokenizer 的基础词表 4096，加 EOD 后按 TP 和 128 的倍数补齐，
覆盖原生 MockGPT 数据的 token 范围。融合开关关闭，TF32 关闭。

Outer：成功步周期 r=3，momentum=0.9，learning rate=0.7，tile=257 坐标；
第 3 个 attempted step 由 rank 1 注入 skip 投票，所有 ranks 在更新前统一跳过。
每次完整轨迹 11 个 attempted steps、10 个 successful steps、3 次 outer；
outer 位于 attempted 4/7/10，第 11 步验证最后一次提交的真实 consumer。

| 阶段 | TP / inner DP | cohort | state | 检查 |
|---|---|---|---|---|
| tp1-s1 | 1 / 1 | 1 | device | 连续运行基准 |
| tp1-s2 | 1 / 1 | 2 | device | 与 s1 逐步完全一致 |
| tp1-s4 | 1 / 1 | 4 | device | 与 s1 逐步完全一致 |
| tp1-host | 1 / 1 | 2 | pinned host | 与 s1 逐步完全一致 |
| split | 1 / 1 | 2 | device | 第 5 步保存并退出，outer 周期中途 |
| resume | 1 / 1 | 2 | device | 新进程恢复第 5 步，运行到第 11 步 |
| tp2-s2 | 2 / 1 | 2 | device | 不同 TP 坐标的跨 learner 分组 |
| dp2-s2 | 1 / 2 | 2 | device | inner DP2 复制式普通 optimizer |

split 的成功状态标记为 partial（符合预定提前退出），其余必须 passed。
八阶段分别启动新训练进程，因此会多次打印模型/分布式初始化。
日志的 `[E0b n/8] START/DONE/FAILED` 标出阶段切换；DONE 仅表示进程正常退出，
最终仍以 summary 验收为准。任一阶段非零退出立即停止，torchrun 不自动重试。
resume 读取 split 的完整 checkpoint，不能只加载 model weights；
调度器总训练长度在两阶段都固定为 11，恢复时不重置学习率曲线。
TP2、inner-DP2 各自在对应坐标组内对照固定图 oracle，不强求它们与 TP1
不同 learner 数/样本分组的轨迹相等。

## 本轮实现与接口

| 文件 | 作用 |
|---|---|
| `megatron/core/outer_sync/coordinates.py` | 稳定名称、形状和分片描述构成坐标 schema，直接读写现有 master 参数，处理跨参数 tile |
| `megatron/core/outer_sync/executor.py` | 生产与 E0a 共用的单 slot 执行器，只常驻 owned R/M 和六个显式 tile buffer |
| `megatron/core/outer_sync/runtime.py` | 对应坐标分组、全局 skip、成功步 clock、提交与验收 trace |
| `megatron/core/outer_sync/checkpoint.py` | 逐 rank 完整状态、hash/size 核验、原子发布、相同布局恢复 |
| `megatron/core/optimizer/optimizer.py` | 显式 master/model 配对；在任何 optimizer update 前执行 skip consensus |
| `megatron/training/training.py` | 建立 runtime，接成功步边界，绕过旧 outer 状态分配，恢复数据迭代器建立后的 RNG |
| `megatron/training/checkpointing.py` | 新 runtime 分流到完整逐 rank save/load |
| `megatron/training/arguments.py` | opt-in centered runtime 和正确性参数；默认 legacy |
| `megatron/training/initialize.py` | 无融合的 core 模型不探测未使用的 CUDA 扩展；仍编译实际需要的数据 helper |
| `experiments/centered_outer/e0b*` | 固定矩阵、实际入口包装、Slurm 和严格汇总 |

验证模式会复制小模型状态到 CPU、额外 gather 和逐步 hash，明确不在性能预算中。
生产路径不构造完整 flat master/reference；普通 optimizer 的现有 master/model 仍计入模型状态。
当前完整恢复限定 single dataloader、0 workers、同一 world/rank/坐标/配方；
不支持的 optimizer/拓扑/恢复模式会拒绝启动。

## 通过条件

1. 八阶段各四 ranks 的实际 CUDA/NCCL、入口和参数记录完整。
2. 每一步 model/master、inner state 和 loss 记录完整且有限。
3. 每次 outer 的 master/R/M bits 对照独立 NumPy 固定图 oracle 一致，
   inner optimizer state 保留；下一次真实 forward 的 model bits 已提交。
4. 第 3 步全局跳过，master/inner state 不更新；成功步与 outer 计数准确。
5. s1/s2/s4/host/resume 的每 rank 完整 loss/master/model/inner 轨迹一致，
   恢复后的前五步 trace 与 split 原始结果一致。
6. 恢复 receipt 包含第 5 步、4 成功步、40 consumed samples 和已验证的 rank 文件 hash/size。
7. manifest 的执行源码 hash 与汇总时一致；launcher 或任一 worker 非零退出不能产生 passed。

本地另有真实单 rank 非有限梯度触发全局跳步的四进程 CPU 检查；
GPU gate 中注入的是明确标记的 skip 投票，不将它描述为 GPU 数值溢出实验。

## 回传产物

`local/centered_outer/e0b-<jobid>/`：

- `manifest.json`、`node-0.json`、`summary.json`；
- 八份阶段 `*.log`；
- `case-*/launch-rank-*.json`、`case-*/rank-*.json`，
  失败时额外 `failure-rank-*.json`；
- `checkpoints/` 的完整二进制状态留在集群供本轮恢复使用，Git 忽略。

以上文本和根目录 `pier-e0b-<jobid>.out` 直接走普通 Git 回传；
不需要把结果挪到另一个目录，不需要 Git LFS。
正常 shell 失败会保留阶段日志、非零退出码和失败 summary；
超时/SIGKILL 可能没有完整 summary，不能当作通过。

## 对 draft 的贡献与边界

E0a 已通过组件检查；E0b 通过后可补“真实 Megatron 小 GPT 的训练接入、
成功步/skip/同布局恢复”这一层证据。
仍需 Qwen 的真实状态与数据配方、多 slot GPU 验证、强基线、完整周期吞吐、
真实显存峰值和物理 bytes；不支持 full-pretraining quality 结论。
E1–E4 性能、消融与扩展尚未启动，不能凭本轮填性能表。
