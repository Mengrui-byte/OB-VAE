# 显式稀疏地址：非原生Gaussian诊断

更新：2026-10-29T13:59:27.603099+08:00。输入范围：完整256分钟。

[返回总表](../README.md) · [模型谱系](../../MODEL_LINEAGE.md)

## 模型结构

以下图对应主要展示点的结构；分支其他可选结构见谱系说明，实际开关和精度以检查点配置为准。

![结构](architecture.svg)

```mermaid
flowchart TD
  N0["Observed full256<br/>diagnostic content encoder"]
  N1["Price latent + explicit sparse content<br/>per-window quantity addresses / values"]
  N0 --> N1
  N2["Fixed 8192 scalar allocation<br/>m3072 leaves 8 price channels"]
  N1 --> N2
  N3["Neural price + sparse reconstruction<br/>engineered route; not pure Gaussian"]
  N2 --> N3
  N4["Diagnostic book<br/>do not compare as native model success"]
  N3 --> N4
```

- 瓶颈：8192标量预算内含显式窗口地址与数量值；不是纯学习Gaussian内容。
- 架构变化：工程化稀疏内容编码＋神经价格子空间，用于能力/指标诊断。
- 结论边界：即使指标极好，也不能与单Gaussian神经压缩作为同一任务直接排名。
- 原模型代码（相对OBreconstuction）：`model/autoencoder/sparse_address_vae.py`。

## 各评估口径的最佳已记录检查点

“最佳”定义为本分支、相同窗口长度/测试时段/窗口数/有效计数/目标RMS下所有已保存正步数评估中挂量RMSE最低；这是回顾性选点，包含中间检查点，不冒充独立测试确认；价格、≥100BTC指标和动态相关同时报告，不声称所有指标同时最优。不同口径分别保留最佳点，不做跨口径总冠军。下表第一项只是较大测试样本口径的展示代表。

| 组别 | 步数/epoch | 口径 | 价格RMSE USDT | 挂量RMSE BTC | ≥100召回 | ≥100精确率 | F1 | 动态相关 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| m3072 | 0 / None | 256分钟×16窗，['2025-01-01', '2025-04-01'] | 75.59411 | 0.99068 | 100.00% | 100.00% | 100.00% | 0.38578 |

### 最佳点1：m3072

- 实验：`recons/recall80_goal_20261027_235344/sparse_address_diagnostic`。
- 原始指标：`recons/recall80_goal_20261027_235344/sparse_address_diagnostic/m3072/step0_development.json`，字段/行：`result`。
- [固定指标证据](best_0_evidence.json) · [权重元数据与哈希](best_0_checkpoint.json) · [原实验计划](best_0_plan.json)。
- 权重：`recons/recall80_goal_20261027_235344/sparse_address_diagnostic/m3072/model.pt`；SHA256 `1a2aa768f3207594f7a1bc4e775c9bb31a20ed7f2a81487947d5b19a1f28ecdc`。这是显式稀疏诊断模型，不是原生Gaussian神经目标。

原始预测：`recons/recall80_goal_20261027_235344/sparse_address_diagnostic/m3072/step0_profiles.npz`，SHA256 `85f2f3765603cabfd56abb5ac4670f683b2d77ef9308e1653ee4f509928c76e0`。固定第一个保存案例，分钟[0, 128, 255]，完整2048档。它是案例可视化，不替代全测试统计；没有新增模型推理。

![真实黑色，预测绿色；完整档位](best_0_btc.png)

![真实黑色，预测绿色；完整档位](best_0_log.png)

![真实黑色，预测绿色；完整档位](best_0_price.png)

[图中精确数据](best_0_display_points.npz)；未做深度平均、峰值截断或按误差挑样本。

### 若优先大单F1

保留候选中F1最高点：`recons/recall80_goal_20261027_235344/sparse_address_diagnostic/m1024`，step=0，F1=100.00%，挂量RMSE=1.14978 BTC，价格RMSE=75.0689 USDT。它可能属于不同测试口径，不自动替代上面的最小挂量误差点。完整候选与口径见JSON。

## 同预算共同终点参考

该分支保留的共同终点评估中，最低挂量误差点为`recons/recall80_goal_20261027_235344/sparse_address_diagnostic/m3072`，step=0，挂量RMSE=0.99068 BTC、价格RMSE=75.5941 USDT。它与回顾性最低误差检查点是两个选型口径；不同来源或测试协议不能据此直接作因果比较。

## 所有保留候选与比较边界

![误差可视化](metrics.png)

图中最多展示10个低挂量误差点，只能在相同口径内比较；所有候选均保留在[candidates.json](candidates.json)。0步初始化不参与已训练分支排名；显式稀疏地址诊断单独标明。历史参考/未压缩模型不冒充压缩方案结果；训练拟合、合成移动峰和已见窗口不混入测试排名。历史2025Q1已重复查看，不称独立确认。

<!-- PARETO -->
## 价格与挂量的取舍

最小挂量误差可能以明显价格退化为代价；下面按每个评估口径分别列出价格/挂量二维非支配点（最多10个）。这是多目标取舍，不是因果或显著性结论。完整点仍在候选JSON中。

口径：`[256, ["2025-01-01", "2025-04-01"], 16, 16777216, 2.2170201]`

| 组别 | step / epoch | 价格RMSE | 挂量RMSE |
|---|---|---:|---:|
| m3072 | 0 / None | 75.5941 | 0.99068 |
| m2048 | 0 / None | 74.5031 | 1.05161 |

<details><summary>稀疏诊断实际结构配置</summary>

```json
{
  "unified_vae_config": {
    "levels": 2048,
    "quantity_scale": 7.517722085208762,
    "activation_checkpointing": true
  },
  "sparse_address_config": {
    "peak_slots": 3072,
    "use_addresses": true,
    "frame_ffn": 1536
  },
  "step": 0,
  "state_tensor_count": 244,
  "state_scalars": 11730238
}
```

</details>
