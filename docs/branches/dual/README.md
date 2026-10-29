# Global/Local双Decoder VAE

更新：2026-10-29T13:59:27.603099+08:00。输入范围：历史16分钟。

[返回总表](../README.md) · [模型谱系](../../MODEL_LINEAGE.md)

## 模型结构

以下图对应主要展示点的结构；分支其他可选结构见谱系说明，实际开关和精度以检查点配置为准。

![结构](architecture.svg)

```mermaid
flowchart TD
  N0["Observed book<br/>16 minutes / 2048 levels per side"]
  N1["Shared encoder<br/>price + log-gap + quantity + differences / LoG"]
  N0 --> N1
  N2["Two Gaussian fields<br/>G: 16x64; L: 16x2x128x16"]
  N1 --> N2
  N3["Price decoder<br/>ordered mid / spread / gap"]
  N2 --> N3
  N4["Quantity decoder + volume refinement<br/>quantity conditioned on predicted price"]
  N3 --> N4
```

- 瓶颈：G[B,16,64]＋L[B,16,2,128,16]；合计66560标量。
- 架构变化：共享Encoder输出Global/Local；价格/数量独立Decoder；空间/数量/时间容量对照。
- 结论边界：dual首先指双Decoder，不等于双塔Encoder；16分钟结果不代表256分钟能力。
- 原模型代码（相对OBreconstuction）：`model/dual/orderbook_vae.py`, `model/experimental/dual_long.py`。

## 各评估口径的最佳已记录检查点

“最佳”定义为本分支、相同窗口长度/测试时段/窗口数/有效计数/目标RMS下所有已保存正步数评估中挂量RMSE最低；这是回顾性选点，包含中间检查点，不冒充独立测试确认；价格、≥100BTC指标和动态相关同时报告，不声称所有指标同时最优。不同口径分别保留最佳点，不做跨口径总冠军。下表第一项只是较大测试样本口径的展示代表。

| 组别 | 步数/epoch | 口径 | 价格RMSE USDT | 挂量RMSE BTC | ≥100召回 | ≥100精确率 | F1 | 动态相关 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| volume_s42 | 1200 / None | 16分钟×256窗，['2025-01-01', '2025-04-01'] | 8.43797 | 0.74623 | 92.29% | 95.57% | 93.90% | 0.97850 |

### 最佳点1：volume_s42

- 实验：`checkpoints/dual_12h_20261024_004600`。
- 原始指标：`checkpoints/dual_12h_20261024_004600/runs/volume_s42/latest_evaluation.json`，字段/行：`result`。
- [固定指标证据](best_0_evidence.json) · [权重元数据与哈希](best_0_checkpoint.json) · [原实验计划](best_0_plan.json)。
- 权重：`checkpoints/dual_12h_20261024_004600/runs/volume_s42/evaluation_step_0001200.pt`；SHA256 `ef82f8384b19f9fe03a18d4a81732ee99c94261681b6dddfad0cd8791dddce96`。权重留在原项目，本次不向Git批量上传大检查点。

<details><summary>实际检查点结构配置</summary>

```json
{
  "model_config": {
    "levels": 2048,
    "quantity_scale": 1.0,
    "global_dim": 64,
    "local_dim": 16,
    "causal": false,
    "gap_activation": "softplus",
    "detach_price": false,
    "fix_first_mid": true
  },
  "epoch": 4,
  "state_tensor_count": 289,
  "state_scalars": 4861004
}
```

</details>

原始预测：`checkpoints/dual_12h_20261024_004600/reconstruction_20261024_111849/profiles.npz`，SHA256 `f0d9f1a759c45245eee28fbdfc8b15ffd9d4b6bf4dad9c686f04627b2adf478e`。固定第一个保存案例，分钟[0, 8, 15]，完整2048档。它是案例可视化，不替代全测试统计；没有新增模型推理。

![真实黑色，预测绿色；完整档位](best_0_btc.png)

![真实黑色，预测绿色；完整档位](best_0_log.png)

![真实黑色，预测绿色；完整档位](best_0_price.png)

[图中精确数据](best_0_display_points.npz)；未做深度平均、峰值截断或按误差挑样本。

### 若优先大单F1

保留候选中F1最高点：`checkpoints/dual_12h_20261024_004600/volume_s42`，step=1200，F1=93.90%，挂量RMSE=0.74623 BTC，价格RMSE=8.4380 USDT。它可能属于不同测试口径，不自动替代上面的最小挂量误差点。完整候选与口径见JSON。

## 所有保留候选与比较边界

![误差可视化](metrics.png)

图中最多展示10个低挂量误差点，只能在相同口径内比较；所有候选均保留在[candidates.json](candidates.json)。0步初始化不参与已训练分支排名；历史参考/未压缩模型不冒充压缩方案结果；训练拟合、合成移动峰和已见窗口不混入测试排名。历史2025Q1已重复查看，不称独立确认。

<!-- PARETO -->
## 价格与挂量的取舍

最小挂量误差可能以明显价格退化为代价；下面按每个评估口径分别列出价格/挂量二维非支配点（最多10个）。这是多目标取舍，不是因果或显著性结论。完整点仍在候选JSON中。

口径：`[16, ["2025-01-01", "2025-04-01"], 256, 16777216, 2.5669809]`

| 组别 | step / epoch | 价格RMSE | 挂量RMSE |
|---|---|---:|---:|
| volume_s42 | 1200 / None | 8.4380 | 0.74623 |
| temporal_s42 | 1400 / None | 7.9088 | 1.29029 |
| control_s42 | 1400 / None | 4.0884 | 1.29615 |
