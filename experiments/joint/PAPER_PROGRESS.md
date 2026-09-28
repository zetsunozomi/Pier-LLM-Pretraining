# 红框结果 → 实机数据

唯一目标：逐次提交、拉回、quick check，并用可追溯的实测结果替换 draft 红框。
不再单独申请正确性验收机器；每次启动必须指出其填写位置。必要的输入、
有限值、模型写回和完整测量窗口检查保留在正式实验中。

| 红框 | 论文位置 | 所需结果 | 当前状态 |
|---|---|---|---|
| A6、A8 | Table VI；§V-G 指标 | 固定预算四方实现的 outer+commit 延迟、allocated/reserved 峰值、布局/slot/tile、reference/momentum 通信；同配置原 colocated static 对照 | **下一次只跑 joint-s1，第 1 次独立重复** |
| A7、A8、A4 | §V-G；切换存储/实现证据 | 同一模型和拓扑下预算收缩/扩张，切换成本、显存峰值及累计收益；需要解释峰值时在对应实验加 trace | 待固定预算数据回收后逐步安排 |
| B1、B2、B3、B5、B7、B9 | Table II/III、Table IV 的 32 GPU 部分、Table V anchor 及正文比例 | 新版 P 与相应 O/R/OS/G 的同配置比较、32 GPU cohort sweep；所有派生百分比由对应数据重算 | 待安排；旧数据不改名复用 |
| B4、B6 | Table IV/V 的 8 GPU、3B 部分 | 8 GPU 的 cohort sweep 和 O/R/P 比较 | 待安排 |
| B8 | Table V 的 32 GPU、1.5B 部分 | 1.5B 的 O/R/P 比较及派生比例 | 待安排 |
| A5、B10 | §V-C 验证文字 | 已有新实现的 13 阶段、40 项逐 rank 轨迹比较，含两种恢复、TP2/inner-DP2 | 已有实机证据：`out/joint-training-gate-58995490-20260927-230508.myDKhH/gate/summary.json`；无需新作业 |
| A1–A3、B11 | 摘要、引言、结论、测量协议 | 随最终实测表更新汇总、范围、采样方式和比较条件 | 数据齐后写入，不另开实验 |

## 当前唯一启动项

`sbatch experiments/joint/paper.sbatch joint-s1 1`

- 8 节点 × 4 A100 40GB；walltime **00:30:00**；Qwen2.5-3B，TP2/K16。
- 固定预算：next-phase 35 GiB，transition 39 GiB；workspace 64 MiB。
- r=50；2 个 warmup 周期 + 3 个 measured 周期，共 250 步；第 1 次独立 launch。
- 使用原有本地模型快照及 mock-token recipe，不下载新模型，不运行独立 gate。
- 输出 `out/joint-paper-fixed-.../campaign/joint-s1/run-1-P/` 的逐 rank 周期报告，
  以及上层 `summary.json`、`campaign.json` 和日志。
- 本轮取得 joint 行的首份正式样本；还不能宣称四方消融或三次重复完成。
- 历史相同 250-step static/P 工作负载每方法约 14 分钟；新版 joint 的真实耗时待本轮返回。

分方法短作业之间不能宣称同 allocation 配对；汇总使用
`qwen_summary.py --split-allocations` 并保留 allocation、case、repeat 身份。
论文最终报告跨 allocation 的独立重复统计，旧配对表仍保留自己的采样说明。
若后续某比较必须同 allocation，按完整比较所需时长申请，不削减周期来硬塞半小时。

每次 quick check 后记录：有效/失败、实际耗时、可填写的单元格、尚缺的重复或对照，
然后只给下一次的同步与启动命令。只有已有数据支持时才更新稿件数字；失败和
中断记录保留，不作为测量完成。诊断时间不计作论文加速结果。
