# 保价logit/配对响应读出

更新：2026-10-29T13:59:27.603099+08:00。输入范围：完整256分钟。

[返回总表](../README.md) · [模型谱系](../../MODEL_LINEAGE.md)

## 模型结构

以下图对应主要展示点的结构；分支其他可选结构见谱系说明，实际开关和精度以检查点配置为准。

![结构](architecture.svg)

```mermaid
flowchart TD
  N0["Observed full256<br/>frame1 encoder; only Encoder frozen here"]
  N1["Single Z<br/>frame1; B x 64 x 128"]
  N0 --> N1
  N2["Original quantity decoder<br/>32-channel full-depth features + frame256"]
  N1 --> N2
  N3["Shape/mass neural readout<br/>Conv1d32->1 + Linear256->2"]
  N2 --> N3
  N4["Quantity shape redistribution + mass scaling<br/>use_shape=True, use_mass=True; detached price geometry"]
  N3 --> N4
```

- 瓶颈：唯一[B,64,128]不变。
- 架构变化：小参数加性logit vs 乘性数量响应；shape/mass、frame/seed读出诊断。
- 结论边界：冻结缓存最小二乘、原生模型训练、固定权重观察须分开，不把缓存拟合成绩移给神经模型。
- 原模型代码（相对OBreconstuction）：`model/autoencoder/paired_logit_vae.py`, `model/autoencoder/logit_readout_vae.py`。

## 各评估口径的最佳已记录检查点

“最佳”定义为本分支、相同窗口长度/测试时段/窗口数/有效计数/目标RMS下所有已保存正步数评估中挂量RMSE最低；这是回顾性选点，包含中间检查点，不冒充独立测试确认；价格、≥100BTC指标和动态相关同时报告，不声称所有指标同时最优。不同口径分别保留最佳点，不做跨口径总冠军。下表第一项只是较大测试样本口径的展示代表。

| 组别 | 步数/epoch | 口径 | 价格RMSE USDT | 挂量RMSE BTC | ≥100召回 | ≥100精确率 | F1 | 动态相关 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| both | 128 / None | 256分钟×16窗，['2025-01-01', '2025-04-01'] | 66.03862 | 2.00867 | 58.32% | 45.09% | 50.86% | 0.09790 |

### 最佳点1：both

- 实验：`recons/recall80_goal_20261027_235344/decoder_readout_pilot`。
- 原始指标：`recons/recall80_goal_20261027_235344/decoder_readout_pilot/both/step128_development.json`，字段/行：`result`。
- [固定指标证据](best_0_evidence.json) · [权重元数据与哈希](best_0_checkpoint.json) · [原实验计划](best_0_plan.json)。
- 权重：`recons/recall80_goal_20261027_235344/decoder_readout_pilot/both/step128.pt`；SHA256 `0cd894724cc4faf14c5bad770def6f2fc973bc21534996647058ee0fa2b11f35`。权重留在原项目，本次不向Git批量上传大检查点。

<details><summary>实际检查点结构配置</summary>

```json
{
  "unified_vae_config": {
    "levels": 2048,
    "quantity_scale": 7.517722085208762,
    "activation_checkpointing": true
  },
  "local_hint_config": {
    "layout": "frame1",
    "frame_ffn": 1536
  },
  "step": 128,
  "lane": "both",
  "hint_loss_config": {
    "state": "complete",
    "channel_mean": [
      0.9250172972679138,
      1.8531343936920166,
      1.5194728374481201,
      1.7943356037139893,
      -1.4503252506256104,
      2.4705286026000977,
      0.3860763907432556,
      1.8132270574569702,
      -0.06249728426337242,
      -0.31646180152893066,
      -0.07554971426725388,
      -0.5030801892280579,
      -0.9876208305358887,
      0.712155818939209,
      -0.4036191403865814,
      0.38522931933403015
    ],
    "channel_scale": [
      0.9728396534919739,
      1.4675811529159546,
      1.2437142133712769,
      2.314589500427246,
      1.4124771356582642,
      7.2413859367370605,
      1.2508740425109863,
      3.029496192932129,
      1.713751196861267,
      3.252234697341919,
      1.3342822790145874,
      1.5868194103240967,
      1.2489420175552368,
      2.3378970623016357,
      1.362396240234375,
      1.2974568605422974
    ],
    "amplitude_bounds": [
      106.1989974975586,
      492.50299072265625
    ],
    "hint_weight": 0.00014210892251719886,
    "teacher_sha256": "542758346c8ed763dc4c04738eb7e9a0080c5959a164acefd4e7dff981dd19c6",
    "training_indices": [
      0,
      872,
      1755,
      2639,
      3515,
      4394,
      5266,
      6152
    ],
    "base_mu_gradient_rms": 0.7122394079445133,
    "raw_hint_mu_gradient_rms": 1252.9815076500806,
    "ratio": 0.25,
    "teacher_hash_unchanged": true,
    "scope": "8 preselected trainingwindows only, nooptimizerupdates/nodevelopmentcalibration; teacherownnormalizer, FP32teacher/BF16student.",
    "mid_weight": 0.5898305515425094,
    "mid_gradient_norms": {
      "pre_guard_loss": 2.614252145220998,
      "mid_protection": 4.432208773154042
    },
    "mid_scope": "Fixed8trainingwindows,PriceDecodermidheadgradientRMSratio1.0,nooptimizerupdates"
  },
  "shape_mass_config": {
    "layout": "frame1",
    "frame_ffn": 1536,
    "use_shape": true,
    "use_mass": true,
    "detach_price_geometry": true
  },
  "state_tensor_count": 245,
  "state_scalars": 11736648
}
```

</details>

原始预测：`recons/recall80_goal_20261027_235344/decoder_readout_pilot/both/step128_profiles.npz`，SHA256 `6b9cd3d29ac46e32e7c338972bccee1a48d04fc6e21bd0f56232e77d2a88d5c9`。固定第一个保存案例，分钟[0, 128, 255]，完整2048档。它是案例可视化，不替代全测试统计；没有新增模型推理。

![真实黑色，预测绿色；完整档位](best_0_btc.png)

![真实黑色，预测绿色；完整档位](best_0_log.png)

![真实黑色，预测绿色；完整档位](best_0_price.png)

[图中精确数据](best_0_display_points.npz)；未做深度平均、峰值截断或按误差挑样本。

### 若优先大单F1

保留候选中F1最高点：`recons/recall80_goal_20261027_235344/paired_logit_readout/shared`，step=512，F1=54.55%，挂量RMSE=2.17470 BTC，价格RMSE=59.2638 USDT。它可能属于不同测试口径，不自动替代上面的最小挂量误差点。完整候选与口径见JSON。

## 同预算共同终点参考

该分支保留的共同终点评估中，最低挂量误差点为`recons/recall80_goal_20261027_235344/decoder_readout_pilot/quantity`，step=256，挂量RMSE=2.02899 BTC、价格RMSE=81.3413 USDT。它与回顾性最低误差检查点是两个选型口径；不同来源或测试协议不能据此直接作因果比较。

## 所有保留候选与比较边界

![误差可视化](metrics.png)

图中最多展示10个低挂量误差点，只能在相同口径内比较；所有候选均保留在[candidates.json](candidates.json)。0步初始化不参与已训练分支排名；显式稀疏地址诊断单独标明。历史参考/未压缩模型不冒充压缩方案结果；训练拟合、合成移动峰和已见窗口不混入测试排名。历史2025Q1已重复查看，不称独立确认。

<!-- PARETO -->
## 价格与挂量的取舍

最小挂量误差可能以明显价格退化为代价；下面按每个评估口径分别列出价格/挂量二维非支配点（最多10个）。这是多目标取舍，不是因果或显著性结论。完整点仍在候选JSON中。

口径：`[256, ["2025-01-01", "2025-04-01"], 16, 16777216, 2.2170201]`

| 组别 | step / epoch | 价格RMSE | 挂量RMSE |
|---|---|---:|---:|
| both | 128 / None | 66.0386 | 2.00867 |
| both | 2048 / None | 59.2638 | 2.07239 |
