# Global/Local统一连续4096 PCA

更新：2026-10-29T13:59:27.603099+08:00。输入范围：历史16分钟。

[返回总表](../README.md) · [模型谱系](../../MODEL_LINEAGE.md)

## 模型结构

以下图对应主要展示点的结构；分支其他可选结构见谱系说明，实际开关和精度以检查点配置为准。

![结构](architecture.svg)

```mermaid
flowchart TD
  N0["Observed book / frozen Cross<br/>16 minutes"]
  N1["Flatten original Global / Local<br/>66560 scalar coordinates"]
  N0 --> N1
  N2["Fixed joint PCA<br/>66560 -> 4096"]
  N1 --> N2
  N3["One continuous state<br/>B x 4096"]
  N2 --> N3
  N4["Inverse PCA<br/>restore Global / Local"]
  N3 --> N4
  N5["Adapted price / quantity decoders<br/>original / Dice / shape-loss controls"]
  N4 --> N5
```

- 瓶颈：唯一连续[B,4096]；先还原Global/Local再解码。
- 架构变化：固定联合PCA，Decoder重训；Dice/无符号形态/有符号差分损失控制。
- 结论边界：原始Local测试失真约70.7%；固定预算形态监督未恢复大单。不能据此推断学习型4096编码器必然失败。
- 原模型代码（相对OBreconstuction）：`model/autoencoder/joint_pca_codec.py`, `model/experimental/joint_pca_state.py`。

## 各评估口径的最佳已记录检查点

“最佳”定义为本分支、相同窗口长度/测试时段/窗口数/有效计数/目标RMS下所有已保存正步数评估中挂量RMSE最低；这是回顾性选点，包含中间检查点，不冒充独立测试确认；价格、≥100BTC指标和动态相关同时报告，不声称所有指标同时最优。不同口径分别保留最佳点，不做跨口径总冠军。下表第一项只是较大测试样本口径的展示代表。

| 组别 | 步数/epoch | 口径 | 价格RMSE USDT | 挂量RMSE BTC | ≥100召回 | ≥100精确率 | F1 | 动态相关 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| compressed4096 | 128 / None | 16分钟×128窗，['2025-01-01', '2025-04-01'] | 12.18021 | 2.47808 | 0.00% | 未记录 | 0.00% | 0.20705 |

### 最佳点1：compressed4096

- 实验：`recons/joint4096_morphology_20261026`。
- 原始指标：`recons/joint4096_morphology_20261026/dice_morphology/compressed4096/final_evaluation.json`，字段/行：``。
- [固定指标证据](best_0_evidence.json) · [权重元数据与哈希](best_0_checkpoint.json) · [原实验计划](best_0_plan.json)。
- 权重：`recons/joint4096_morphology_20261026/dice_morphology/compressed4096/final.pt`；SHA256 `5d691d441fcfebf5e16701bd6aadb9ad2c7d06070807e3c4ff4ed9e821cc749f`。权重留在原项目，本次不向Git批量上传大检查点。

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
  "step": 128,
  "lane": "compressed4096",
  "joint_pca_config": {
    "state_dim": 4096,
    "window_size": 16
  },
  "state_tensor_count": 313,
  "state_scalars": 4906316
}
```

</details>

原始预测：`recons/joint4096_morphology_20261026/dice_morphology/compressed4096/development_step128_128windows_profiles.npz`，SHA256 `4a77d88e015cd89429562071463aaec3d075700bba659b8cd44c642327c3a036`。固定第一个保存案例，分钟[0, 8, 15]，完整2048档。它是案例可视化，不替代全测试统计；没有新增模型推理。

![真实黑色，预测绿色；完整档位](best_0_btc.png)

![真实黑色，预测绿色；完整档位](best_0_log.png)

![真实黑色，预测绿色；完整档位](best_0_price.png)

[图中精确数据](best_0_display_points.npz)；未做深度平均、峰值截断或按误差挑样本。

### 若优先大单F1

保留候选中F1最高点：`recons/joint4096_decoder_20261026/compressed4096`，step=128，F1=0.00%，挂量RMSE=2.49809 BTC，价格RMSE=12.2830 USDT。它可能属于不同测试口径，不自动替代上面的最小挂量误差点。完整候选与口径见JSON。

## 所有保留候选与比较边界

![误差可视化](metrics.png)

图中最多展示10个低挂量误差点，只能在相同口径内比较；所有候选均保留在[candidates.json](candidates.json)。0步初始化不参与已训练分支排名；显式稀疏地址诊断单独标明。历史参考/未压缩模型不冒充压缩方案结果；训练拟合、合成移动峰和已见窗口不混入测试排名。历史2025Q1已重复查看，不称独立确认。

<!-- PARETO -->
## 价格与挂量的取舍

最小挂量误差可能以明显价格退化为代价；下面按每个评估口径分别列出价格/挂量二维非支配点（最多10个）。这是多目标取舍，不是因果或显著性结论。完整点仍在候选JSON中。

口径：`[16, ["2025-01-01", "2025-04-01"], 128, 8388608, 2.5344305]`

| 组别 | step / epoch | 价格RMSE | 挂量RMSE |
|---|---|---:|---:|
| compressed4096 | 128 / None | 12.1802 | 2.47808 |
| joint4096_signed_difference_20261026 | 128 / None | 12.1687 | 2.48822 |
| compressed4096 | 128 / None | 12.1574 | 2.49261 |
