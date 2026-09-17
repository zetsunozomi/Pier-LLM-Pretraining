# Centered outer 实验进展

**E0a 已有四卡通过证据；E0b 已完成本地回归，等待首次真实训练 GPU 验证；性能实验尚未开始。**

## 当前提交入口：E0b

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

本地 18 项测试通过；独立四进程 Gloo verifier 的 24 个算子配置和 3 个 toy-training
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
[Qwen 准备记录](../qwen/README.md)；它们尚未接入训练入口，也没有 Qwen GPU 结果。

E0b 通过后，接 Qwen 与数据/状态初始化配方，补强基线和完整周期计量，再进行 E1/E2。
Distributed optimizer、多 slot pipeline、PP/CP/MoE、跨拓扑恢复尚无支持承诺。
当前没有 GPU 吞吐、总峰值显存、物理链路流量或最终模型质量结果；
draft 性能表的 42 个单元格继续留空。
