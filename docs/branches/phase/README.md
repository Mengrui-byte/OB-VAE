# 区域内16档直接细节/空间上采样路径

更新：2026-10-29T13:59:27.603099+08:00。输入范围：完整256分钟。

[返回总表](../README.md) · [模型谱系](../../MODEL_LINEAGE.md)

## 模型结构

以下图对应主要展示点的结构；分支其他可选结构见谱系说明，实际开关和精度以检查点配置为准。

![结构](architecture.svg)

```mermaid
flowchart TD
  N0["Full256 encoder<br/>original 8192 posterior"]
  N1["Single shared Z<br/>B x 64 x 128"]
  N0 --> N1
  N2["Temporal expansion<br/>256 frames x 256 features"]
  N1 --> N2
  N3["Regional spatial seed<br/>128 regions / side"]
  N2 --> N3
  N4["Original upsample + offset residual<br/>Conv1d 256->16; independent offsets"]
  N3 --> N4
  N5["Softplus quantity + price decoder<br/>no extra per-window latent"]
  N4 --> N5
```

- 瓶颈：[B,64,128]不变。
- 架构变化：区域共享修正 vs 16个独立offset；Encoder冻结/训练交叉对照。
- 结论边界：增加的是post-Z读出路径，不是Encoder跳连；Jacobian集中不等于网络没有表达能力。
- 原模型代码（相对OBreconstuction）：`model/autoencoder/phase_book_vae.py`, `model/decoders/phase_volume.py`。

## 各评估口径的最佳已记录检查点

“最佳”定义为本分支、相同窗口长度/测试时段/窗口数/有效计数/目标RMS下所有已保存正步数评估中挂量RMSE最低；这是回顾性选点，包含中间检查点，不冒充独立测试确认；价格、≥100BTC指标和动态相关同时报告，不声称所有指标同时最优。不同口径分别保留最佳点，不做跨口径总冠军。下表第一项只是较大测试样本口径的展示代表。

| 组别 | 步数/epoch | 口径 | 价格RMSE USDT | 挂量RMSE BTC | ≥100召回 | ≥100精确率 | F1 | 动态相关 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| joint_independent | 2048 / None | 256分钟×16窗，['2025-01-01', '2025-04-01'] | 62.50153 | 2.17953 | 0.00% | 未记录 | 0.00% | 0.00079 |

### 最佳点1：joint_independent

- 实验：`recons/phase_encoder_20261026_231213`。
- 原始指标：`recons/phase_encoder_20261026_231213/joint_independent/step2048_evaluation.json`，字段/行：`result`。
- [固定指标证据](best_0_evidence.json) · [权重元数据与哈希](best_0_checkpoint.json) · [原实验计划](best_0_plan.json)。
- 权重：`recons/phase_encoder_20261026_231213/joint_independent/step2048.pt`；SHA256 `d1d98e373208edac2208626746ac709cd475a8934e11e86e85448e1dd49b9412`。权重留在原项目，本次不向Git批量上传大检查点。

<details><summary>实际检查点结构配置</summary>

```json
{
  "unified_vae_config": {
    "levels": 2048,
    "quantity_scale": 7.517722085208762,
    "activation_checkpointing": true
  },
  "phase_decoder_config": {
    "detail_mode": "independent"
  },
  "step": 2048,
  "lane": "joint_independent",
  "state_tensor_count": 242,
  "state_scalars": 11655654
}
```

</details>

原始预测：`recons/phase_encoder_20261026_231213/joint_independent/step2048_profiles.npz`，SHA256 `1dcca0e8e6f7b62501d1be6bc7559df38e6a453a86e87158eb635bb50cc42d15`。固定第一个保存案例，分钟[0, 128, 255]，完整2048档。它是案例可视化，不替代全测试统计；没有新增模型推理。

![真实黑色，预测绿色；完整档位](best_0_btc.png)

![真实黑色，预测绿色；完整档位](best_0_log.png)

![真实黑色，预测绿色；完整档位](best_0_price.png)

[图中精确数据](best_0_display_points.npz)；未做深度平均、峰值截断或按误差挑样本。

### 若优先大单F1

保留候选中F1最高点：`recons/decoder_isolation_recovery_20261027_112135/original_warm`，step=128，F1=0.00%，挂量RMSE=2.17981 BTC，价格RMSE=78.0532 USDT。它可能属于不同测试口径，不自动替代上面的最小挂量误差点。完整候选与口径见JSON。

## 同预算共同终点参考

该分支保留的共同终点评估中，最低挂量误差点为`recons/phase_encoder_20261026_231213/joint_independent`，step=2048，挂量RMSE=2.17953 BTC、价格RMSE=62.5015 USDT。它与回顾性最低误差检查点是两个选型口径；不同来源或测试协议不能据此直接作因果比较。

## 所有保留候选与比较边界

![误差可视化](metrics.png)

图中最多展示10个低挂量误差点，只能在相同口径内比较；所有候选均保留在[candidates.json](candidates.json)。0步初始化不参与已训练分支排名；显式稀疏地址诊断单独标明。历史参考/未压缩模型不冒充压缩方案结果；训练拟合、合成移动峰和已见窗口不混入测试排名。历史2025Q1已重复查看，不称独立确认。

<!-- PARETO -->
## 价格与挂量的取舍

最小挂量误差可能以明显价格退化为代价；下面按每个评估口径分别列出价格/挂量二维非支配点（最多10个）。这是多目标取舍，不是因果或显著性结论。完整点仍在候选JSON中。

口径：`[256, ["2025-01-01", "2025-04-01"], 16, 16777216, 2.2170201]`

| 组别 | step / epoch | 价格RMSE | 挂量RMSE |
|---|---|---:|---:|
| joint_independent | 2048 / None | 62.5015 | 2.17953 |
| joint_off | 2048 / None | 46.7229 | 2.17982 |
