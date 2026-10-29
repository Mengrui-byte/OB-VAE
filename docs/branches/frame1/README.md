# patch4→frame1时间组织

更新：2026-10-29T13:59:27.603099+08:00。输入范围：完整256分钟。

[返回总表](../README.md) · [模型谱系](../../MODEL_LINEAGE.md)

## 模型结构

以下图对应主要展示点的结构；分支其他可选结构见谱系说明，实际开关和精度以检查点配置为准。

![结构](architecture.svg)

```mermaid
flowchart TD
  N0["Observed full256<br/>spatial features / Bid-Ask attention"]
  N1["Frame-aligned transformer<br/>256 tokens / FFN1536"]
  N0 --> N1
  N2["Per-frame posterior coordinates<br/>256 x 32 -> reshape 64 x 128"]
  N1 --> N2
  N3["Single shared Z<br/>8192 scalars"]
  N2 --> N3
  N4["Frame temporal expansion<br/>reshape 256 x 32; MLP -> 256 features"]
  N3 --> N4
  N5["Price + quantity spatial decoders<br/>Huber / pre-Softplus logit controls"]
  N4 --> N5
```

- 瓶颈：同[B,64,128]；frame1中间为256×32再重排。
- 架构变化：patch/frame时间组织×Huber/激活前logit监督2×2。
- 结论边界：时间特征投影、注意力与Decoder展开均改变，不能把收益只归因于reshape。
- 原模型代码（相对OBreconstuction）：`model/autoencoder/time_layout_vae.py`, `model/encoders/frame_aligned.py`, `model/decoders/frame_aligned.py`。

## 各评估口径的最佳已记录检查点

“最佳”定义为本分支、相同窗口长度/测试时段/窗口数/有效计数/目标RMS下所有已保存正步数评估中挂量RMSE最低；这是回顾性选点，包含中间检查点，不冒充独立测试确认；价格、≥100BTC指标和动态相关同时报告，不声称所有指标同时最优。不同口径分别保留最佳点，不做跨口径总冠军。下表第一项只是较大测试样本口径的展示代表。

| 组别 | 步数/epoch | 口径 | 价格RMSE USDT | 挂量RMSE BTC | ≥100召回 | ≥100精确率 | F1 | 动态相关 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| frame_huber | 4096 / None | 256分钟×16窗，['2025-01-01', '2025-04-01'] | 74.70415 | 2.17945 | 0.00% | 未记录 | 0.00% | 0.00630 |

### 最佳点1：frame_huber

- 实验：`recons/time_logit_20261027_124432`。
- 原始指标：`recons/time_logit_20261027_124432/frame_huber/step4096_evaluation.json`，字段/行：`result`。
- [固定指标证据](best_0_evidence.json) · [权重元数据与哈希](best_0_checkpoint.json) · [原实验计划](best_0_plan.json)。
- 权重：`recons/time_logit_20261027_124432/frame_huber/step4096.pt`；SHA256 `df02bc4c2fd9a171b17f7dbb53b9370cef8992dd41216692c2b71dd8c0c46763`。权重留在原项目，本次不向Git批量上传大检查点。

<details><summary>实际检查点结构配置</summary>

```json
{
  "unified_vae_config": {
    "levels": 2048,
    "quantity_scale": 7.517722085208762,
    "activation_checkpointing": true
  },
  "time_layout_config": {
    "layout": "frame1",
    "frame_ffn": 1536
  },
  "step": 4096,
  "lane": "frame_huber",
  "state_tensor_count": 240,
  "state_scalars": 11735830
}
```

</details>

原始预测：`recons/time_logit_20261027_124432/frame_huber/step4096_profiles.npz`，SHA256 `94ad0acbb32ddb418b41d51f3070d6622af4a74448bd3578b130aef047213f11`。固定第一个保存案例，分钟[0, 128, 255]，完整2048档。它是案例可视化，不替代全测试统计；没有新增模型推理。

![真实黑色，预测绿色；完整档位](best_0_btc.png)

![真实黑色，预测绿色；完整档位](best_0_log.png)

![真实黑色，预测绿色；完整档位](best_0_price.png)

[图中精确数据](best_0_display_points.npz)；未做深度平均、峰值截断或按误差挑样本。

### 若优先大单F1

保留候选中F1最高点：`recons/time_logit_20261027_124432/patch_huber`，step=128，F1=0.00%，挂量RMSE=2.32337 BTC，价格RMSE=230.6441 USDT。它可能属于不同测试口径，不自动替代上面的最小挂量误差点。完整候选与口径见JSON。

## 同预算共同终点参考

该分支保留的共同终点评估中，最低挂量误差点为`recons/time_logit_20261027_124432/frame_huber`，step=4096，挂量RMSE=2.17945 BTC、价格RMSE=74.7042 USDT。它与回顾性最低误差检查点是两个选型口径；不同来源或测试协议不能据此直接作因果比较。

## 所有保留候选与比较边界

![误差可视化](metrics.png)

图中最多展示10个低挂量误差点，只能在相同口径内比较；所有候选均保留在[candidates.json](candidates.json)。0步初始化不参与已训练分支排名；显式稀疏地址诊断单独标明。历史参考/未压缩模型不冒充压缩方案结果；训练拟合、合成移动峰和已见窗口不混入测试排名。历史2025Q1已重复查看，不称独立确认。

<!-- PARETO -->
## 价格与挂量的取舍

最小挂量误差可能以明显价格退化为代价；下面按每个评估口径分别列出价格/挂量二维非支配点（最多10个）。这是多目标取舍，不是因果或显著性结论。完整点仍在候选JSON中。

口径：`[256, ["2025-01-01", "2025-04-01"], 16, 16777216, 2.2170201]`

| 组别 | step / epoch | 价格RMSE | 挂量RMSE |
|---|---|---:|---:|
| frame_huber | 4096 / None | 74.7042 | 2.17945 |
| frame_logit | 2048 / None | 74.4745 | 2.18328 |
