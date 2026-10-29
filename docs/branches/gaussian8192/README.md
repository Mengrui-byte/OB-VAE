# 从头训练的唯一Gaussian8192

更新：2026-10-29T13:59:27.603099+08:00。输入范围：实测完整256分钟。

[返回总表](../README.md) · [模型谱系](../../MODEL_LINEAGE.md)

## 模型结构

以下图对应主要展示点的结构；分支其他可选结构见谱系说明，实际开关和精度以检查点配置为准。

![结构](architecture.svg)

```mermaid
flowchart TD
  N0["Observed full window<br/>B x 256 x 2 x 2048 x 2"]
  N1["16 feature channels + spatial encoder<br/>4 stride-2 stages; 2048 -> 128"]
  N0 --> N1
  N2["Bid/Ask attention + frame reduction<br/>4-frame packing / temporal attention"]
  N1 --> N2
  N3["Single Gaussian posterior<br/>mu / logvar / Z: B x 64 x 128"]
  N2 --> N3
  N4["Two independent temporal/spatial decoders<br/>nearest upsample x4 + predicted geometry"]
  N3 --> N4
  N5["Ordered price + nonnegative quantity<br/>same Z / no Encoder bypass"]
  N4 --> N5
```

- 瓶颈：唯一μ/logvar/Z均[B,64,128]。
- 架构变化：共享Encoder、一次Z采样、双Decoder；确定性AE与3档KL；最初patch4组织。
- 结论边界：这是重新初始化的新架构，不是把旧G/L直接改名；8192为标量容量，不是已证明有效信息维数。
- 原模型代码（相对OBreconstuction）：`model/autoencoder/unified_book_vae.py`, `model/encoders/unified_book.py`, `model/decoders/unified_book.py`。

## 各评估口径的最佳已记录检查点

“最佳”定义为本分支、相同窗口长度/测试时段/窗口数/有效计数/目标RMS下所有已保存正步数评估中挂量RMSE最低；这是回顾性选点，包含中间检查点，不冒充独立测试确认；价格、≥100BTC指标和动态相关同时报告，不声称所有指标同时最优。不同口径分别保留最佳点，不做跨口径总冠军。下表第一项只是较大测试样本口径的展示代表。

| 组别 | 步数/epoch | 口径 | 价格RMSE USDT | 挂量RMSE BTC | ≥100召回 | ≥100精确率 | F1 | 动态相关 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| both | 2048 / None | 256分钟×16窗，['2025-01-01', '2025-04-01'] | 81.75240 | 2.17952 | 0.00% | 未记录 | 0.00% | 0.00025 |

### 最佳点1：both

- 实验：`recons/probability_shape_20261026_181628`。
- 原始指标：`recons/probability_shape_20261026_181628/both/step2048_evaluation.json`，字段/行：`result`。
- [固定指标证据](best_0_evidence.json) · [权重元数据与哈希](best_0_checkpoint.json) · [原实验计划](best_0_plan.json)。
- 权重：`recons/probability_shape_20261026_181628/both/step2048.pt`；SHA256 `f7699d4ff127170c6d7b150bf45878bed9d493fa85544dd92512a74dcedf6cd9`。权重留在原项目，本次不向Git批量上传大检查点。

<details><summary>实际检查点结构配置</summary>

```json
{
  "unified_vae_config": {
    "levels": 2048,
    "quantity_scale": 7.517722085208762,
    "activation_checkpointing": true
  },
  "step": 2048,
  "lane": "both",
  "gradient_route_config": {
    "detach_quantity_latent": false,
    "detach_quantity_geometry": false
  },
  "state_tensor_count": 240,
  "state_scalars": 11651542
}
```

</details>

原始预测：`recons/probability_shape_20261026_181628/both/step2048_profiles.npz`，SHA256 `b0e0362968e8fdf0f91a4a647d380fcf2609ffd1a35a2206690fa87e89648c62`。固定第一个保存案例，分钟[0, 128, 255]，完整2048档。它是案例可视化，不替代全测试统计；没有新增模型推理。

![真实黑色，预测绿色；完整档位](best_0_btc.png)

![真实黑色，预测绿色；完整档位](best_0_log.png)

![真实黑色，预测绿色；完整档位](best_0_price.png)

[图中精确数据](best_0_display_points.npz)；未做深度平均、峰值截断或按误差挑样本。

### 若优先大单F1

保留候选中F1最高点：`recons/loss_allocation_20261026_150351/baseline`，step=1024，F1=0.01%，挂量RMSE=10.93081 BTC，价格RMSE=195.9187 USDT。它可能属于不同测试口径，不自动替代上面的最小挂量误差点。完整候选与口径见JSON。

## 同预算共同终点参考

该分支保留的共同终点评估中，最低挂量误差点为`recons/probability_shape_20261026_181628/both`，step=2048，挂量RMSE=2.17952 BTC、价格RMSE=81.7524 USDT。它与回顾性最低误差检查点是两个选型口径；不同来源或测试协议不能据此直接作因果比较。

## 所有保留候选与比较边界

![误差可视化](metrics.png)

图中最多展示10个低挂量误差点，只能在相同口径内比较；所有候选均保留在[candidates.json](candidates.json)。0步初始化不参与已训练分支排名；历史参考/未压缩模型不冒充压缩方案结果；训练拟合、合成移动峰和已见窗口不混入测试排名。历史2025Q1已重复查看，不称独立确认。

<!-- PARETO -->
## 价格与挂量的取舍

最小挂量误差可能以明显价格退化为代价；下面按每个评估口径分别列出价格/挂量二维非支配点（最多10个）。这是多目标取舍，不是因果或显著性结论。完整点仍在候选JSON中。

口径：`[256, ["2025-01-01", "2025-04-01"], 16, 16777216, 2.2170201]`

| 组别 | step / epoch | 价格RMSE | 挂量RMSE |
|---|---|---:|---:|
| both | 2048 / None | 81.7524 | 2.17952 |
| n1024 | 1024 / None | 78.0532 | 2.18010 |
| wasserstein | 2048 / None | 62.4286 | 2.18035 |
| baseline | 32 / None | 50.1868 | 2.42306 |
