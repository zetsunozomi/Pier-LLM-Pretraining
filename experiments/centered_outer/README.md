# Centered outer 实验进展

**E0a 已有四卡通过证据；E0b 集群及本地原始产物复核均通过；性能实验尚未开始。**

## 当前动作：Qwen 接入与验证准备

`e0b-58457867-20260917-021226` 八阶段完成，s2/s4/host/resume 对 s1 的
16 项 per-rank 轨迹比较全部逐位一致，GPU_executed=true、errors=[]。
无需重复提交 E0b。75 个原始文件与 390 个执行源码 hash 均已本地复核，详见 [结果审查](E0B_REVIEW_58457867.md)。

当前下一份作业是 [Qwen E0c](../qwen/E0C_HANDOFF.md)：固定真权重的
FP32/BF16、TP1/TP2 完整 logits/梯度对照，申请 1 节点 4 卡 1 小时。
先在登录节点准备约 6.18 GB 的固定模型快照，再提交 `experiments/qwen/e0c.sbatch`。
这是模型转换检查，尚不是训练或性能结果；当前本地 27 项 Qwen 检查已通过。

等待 E0c 期间，G/R/W 原生 collective、T 显式 DTensor 对照和实际训练周期
计量已接入同一训练循环并完成 CPU 检查，见 [基线与计量说明](BASELINES_AND_METRICS.md)。
尚未完成 GPU 数值验收、基线调优或 E1 提交脚本，不能据此填写性能结果。

完整状态验收另增加了 [分块文件 oracle](STREAMED_ORACLE.md)：默认内存模式保留
100 万本地参数上限，显式 streamed 模式逐项核对全部坐标，并支持从 checkpoint
中的 R/M 重建与哈希校验；超过原上限的向量及恢复检查已在 CPU 上通过。
当前完整 outer-sync 测试共 38 项通过，真实 Qwen 的 CUDA 验收与存储需求仍待验证。

Qwen 训练入口也已接通 full/selective 激活重计算及 TP saved-activation 分片，
并将配置写入初始化与恢复配方；152 条逐 rank CPU 对照的完整输出和梯度逐位一致。
真实 2K 序列的显存收益尚未测量，详见 [Qwen 重计算说明](../qwen/README.md#activation-recomputation-for-the-later-training-gate)。

原始周期的 [只读汇总工具](BASELINES_AND_METRICS.md#read-only-cycle-collection) 已接入：
校验全 rank 文件、运行 UUID、声明步数、全局 token 与慢 rank 时钟，每次 run 只输出
一个按总量计算的吞吐样本；warmup、恢复前缀和尾部残缺周期留档但不入样本。
这些是计量记录检查，尚不验证真实工作负载、独立配对实验或调优公平性。

真实数据另有 [固定 FineWeb-Edu 分片准备入口](../qwen/CORPUS.md)，可显式下载并
按 Qwen tokenizer 生成有哈希和来源记录的 indexed data，也保留本地 JSONL/Parquet
入口；公共分片尚未在本机准备，E0c 仍使用原来的转换文本，不依赖这一步。

## 复现实验入口：E0b

在本地完成通常的 Git add/commit/push、集群 pull 后，从集群仓库根目录使用原有 `diloco` 环境提交：

```bash
sbatch experiments/centered_outer/e0b.sbatch
```

资源：NERSC `m4431 / regular / gpu`，**1 节点、4 GPU、1 小时上限**。
使用实际 `pretrain_gpt.py`、两层小 GPT 和 mock 数据，不下载 Qwen 或语料。
完整配方、通过标准、产物和范围见 [E0b 交接说明](E0B_HANDOFF.md)。

本轮实现：

- 稳定的 FP32 master 参数坐标映射；训练循环直接调用单 slot centered executor。
- 成功 optimizer step 计数；所有 learner 在更新前统一 skip 决策。
- outer 更新后的 master → BF16 model 提交和下一次 forward 检查。
- 完整逐 rank checkpoint：model/master、inner moments、R/M、scheduler、成功步、
  consumed samples、CPU/CUDA/TP RNG，支持相同布局下周期中途恢复。
- 八个真实训练阶段：cohort 1/2/4、host state、停止/恢复、TP2、inner-DP2。
- 严格核对逐步 loss/状态 hash、固定图 oracle、完整 rank 报告和退出状态。
- 修正 `data/` 误忽略源码；恢复与 E0a manifest 一致的 12 个 legacy/data 文件。

E0b 接入时本地 21 项 outer-sync 回归通过；独立四进程 Gloo verifier 的 24 个算子配置和 3 个 toy-training
配置通过。这些是 CPU 证据，不能替代 E0b 的 CUDA/NCCL 验证。

## Git 回传

E0b 文本证据：`local/centered_outer/e0b-<jobid>/` 下的 JSON/log，
以及根目录 `pier-e0b-<jobid>.out`，已由 Git 放行；直接用通常的 add/commit/push 和本机 pull。
checkpoint 的二进制文件用于同一个 job 内的恢复检查，保留在集群，当前无需 LFS。
不要重生成旧结果的 manifest 或 summary。

## 已接受的历史结果

- [E0a 进度审查](E0A_REVIEW_58425240.md)：作业 58425240，
  一节点四张 A100-SXM4-40GB，48 个算子配置和 3 个 toy-training 配置。
- [E0a 初次交接记录](E0A_HANDOFF.md)：旧脚本 `e0.sbatch` 的范围和使用方法。
- [验证记录](LOCAL_VALIDATION.md)：历史与本轮实际执行的检查。

E0a 尚缺一个未被当前入口引用的 `training_legacy.py` 源文件留档；
新 manifest 单列该历史文件，不把它列为当前执行依赖，也不修改旧 manifest。

## 下一步

Qwen 的原生模型、TP 权重映射和快照 preflight 已独立完成 CPU 检查，见
[Qwen 准备记录](../qwen/README.md)；本轮增加了训练入口与数据预处理/身份检查，尚无 Qwen GPU 结果。

E0b 原始产物复核已完成；继续补 Qwen 真权重 CUDA 对照和实际数据配方，再补强
基线和完整周期计量、进行 E1/E2；[E0c 启动与回传说明](../qwen/E0C_HANDOFF.md)
已准备，等待用户按正常 Git 更新后提交。
Distributed optimizer、多 slot pipeline、PP/CP/MoE、跨拓扑恢复尚无支持承诺。
当前没有 GPU 吞吐、总峰值显存、物理链路流量或最终模型质量结果；
draft 性能表的 42 个单元格继续留空。
