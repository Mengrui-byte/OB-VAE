# 局部查询V4与数量短路V5

更新：2026-10-29T13:59:27.603099+08:00。输入范围：计划256分钟。

[返回总表](../README.md) · [模型谱系](../../MODEL_LINEAGE.md)

## 模型结构

以下图对应主要展示点的结构；分支其他可选结构见谱系说明，实际开关和精度以检查点配置为准。

![结构](architecture.svg)

```mermaid
flowchart TD
  N0["Observed book<br/>256 minutes / both sides"]
  N1["Spatial-temporal encoder<br/>local region features"]
  N0 --> N1
  N2["Local queries<br/>8 queries x 8 features"]
  N1 --> N2
  N3["Compressed state<br/>B x 16 x 64"]
  N2 --> N3
  N4["Main decoder + V5 patch residual<br/>latent-expanded features only"]
  N3 --> N4
  N5["Reconstructed book<br/>price / quantity"]
  N4 --> N5
```

- 瓶颈：[B,16,64]，1024标量。
- 架构变化：8个局部空间查询、直接数量/patch残差/mass-share、pool norm消融。
- 结论边界：小样本overfit与完整测试分开；不能把量预测分支差异当作latent扩容。
- 原模型代码（相对OBreconstuction）：`model/autoencoder/geometry.py`。

## 各评估口径的最佳已记录检查点

“最佳”定义为本分支、相同窗口长度/测试时段/窗口数/有效计数/目标RMS下所有已保存正步数评估中挂量RMSE最低；这是回顾性选点，包含中间检查点，不冒充独立测试确认；价格、≥100BTC指标和动态相关同时报告，不声称所有指标同时最优。不同口径分别保留最佳点，不做跨口径总冠军。下表第一项只是较大测试样本口径的展示代表。

| 组别 | 步数/epoch | 口径 | 价格RMSE USDT | 挂量RMSE BTC | ≥100召回 | ≥100精确率 | F1 | 动态相关 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| window-anchor | 1152 / 18 | 256分钟×694窗，['2025-01-01', '2025-02-01'] | 115.59165 | 2.14957 | 未记录 | 未记录 | 未记录 | 未记录 |

### 最佳点1：window-anchor

- 实验：`checkpoints/btc_patch_quantity_v5_20261018_205710`。
- 原始指标：`checkpoints/btc_patch_quantity_v5_20261018_205710/window-anchor/metrics.jsonl`，字段/行：`line:18:validation`。
- [固定指标证据](best_0_evidence.json) · [权重元数据与哈希](best_0_checkpoint.json) · [原实验计划](best_0_plan.json)。
- 权重：`checkpoints/btc_patch_quantity_v5_20261018_205710/window-anchor/best_quantity.pt`；SHA256 `d0fbbbba232d15c48583230624bdb55502c04f3cf760b88e4e242114ce5f3c65`。权重留在原项目，本次不向Git批量上传大检查点。

<details><summary>实际检查点结构配置</summary>

```json
{
  "epoch": 18,
  "model_config": {
    "architecture": "ae_local_queries_patch_residual_v5",
    "latent_dim": 64,
    "spatial_queries": 8,
    "features_per_query": 8,
    "spatial_regions": "2 sides x 4 contiguous depth bands",
    "decoder": "local region projection and spatial-query cross-attention",
    "bottleneck": "per-time local spatial attention; no temporal mixing or input token-channel flattening",
    "latent_shape_per_sample": [
      16,
      64
    ],
    "quantity_mode": "patch-residual",
    "quantity_decoder": "baseline plus zero-initialized short patch decoder from latent-expanded features; no input bypass"
  },
  "state_tensor_count": 116,
  "state_scalars": 1479971
}
```

</details>

未找到与这个最佳点明确匹配的保存订单簿预测。本页展示真实指标/训练历史图，不借用其他检查点的曲线，不伪造订单簿预览。

## 所有保留候选与比较边界

![误差可视化](metrics.png)

图中最多展示10个低挂量误差点，只能在相同口径内比较；所有候选均保留在[candidates.json](candidates.json)。0步初始化不参与已训练分支排名；历史参考/未压缩模型不冒充压缩方案结果；训练拟合、合成移动峰和已见窗口不混入测试排名。历史2025Q1已重复查看，不称独立确认。

<!-- PARETO -->
## 价格与挂量的取舍

最小挂量误差可能以明显价格退化为代价；下面按每个评估口径分别列出价格/挂量二维非支配点（最多10个）。这是多目标取舍，不是因果或显著性结论。完整点仍在候选JSON中。

口径：`[256, ["2025-01-01", "2025-02-01"], 694, 727684864, null]`

| 组别 | step / epoch | 价格RMSE | 挂量RMSE |
|---|---|---:|---:|
| window-anchor | 1152 / 18 | 115.5916 | 2.14957 |
| window-anchor | 1600 / 25 | 105.8153 | 2.24781 |
| window-anchor | 1920 / 30 | 93.2276 | 2.24908 |
