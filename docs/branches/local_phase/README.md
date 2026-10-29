# 共享/分区域形状与总量头

更新：2026-10-29T13:59:27.603099+08:00。输入范围：完整256分钟。

[返回总表](../README.md) · [模型谱系](../../MODEL_LINEAGE.md)

## 模型结构

以下图对应主要展示点的结构；分支其他可选结构见谱系说明，实际开关和精度以检查点配置为准。

![结构](architecture.svg)

```mermaid
flowchart TD
  N0["Full256 frame1 encoder<br/>single Gaussian 64 x 128"]
  N1["Shared Z -> quantity seed<br/>256 x 2 x 128 x 16"]
  N0 --> N1
  N2["Selected local phase switches<br/>use_shape=False; use_mass=False"]
  N1 --> N2
  N3["Original quantity path<br/>shape/mass correction bypassed"]
  N2 --> N3
  N4["Quantity + original price decoder<br/>best selected switches listed below"]
  N3 --> N4
```

- 瓶颈：唯一[B,64,128]不变。
- 架构变化：LocalPhase共享形状/总量头；RegionalPhase按区域独立参数；各自开关2×2。
- 结论边界：两组都从Z生成的种子读出；要区分参数化容量变化与损失分布KL。
- 原模型代码（相对OBreconstuction）：`model/autoencoder/local_phase_vae.py`, `model/autoencoder/regional_phase_vae.py`, `model/autoencoder/shape_mass_vae.py`。

## 各评估口径的最佳已记录检查点

“最佳”定义为本分支、相同窗口长度/测试时段/窗口数/有效计数/目标RMS下所有已保存正步数评估中挂量RMSE最低；这是回顾性选点，包含中间检查点，不冒充独立测试确认；价格、≥100BTC指标和动态相关同时报告，不声称所有指标同时最优。不同口径分别保留最佳点，不做跨口径总冠军。下表第一项只是较大测试样本口径的展示代表。

| 组别 | 步数/epoch | 口径 | 价格RMSE USDT | 挂量RMSE BTC | ≥100召回 | ≥100精确率 | F1 | 动态相关 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| quarter | 512 / None | 256分钟×16窗，['2025-01-01', '2025-04-01'] | 52.06147 | 1.95783 | 63.18% | 47.41% | 54.17% | 0.13942 |

### 最佳点1：quarter

**该点的形状/总量头均关闭，结果来自后续训练/损失配置，不能归因于新增结构头。**

- 实验：`recons/recall80_goal_20261027_235344/round_008_auxiliary`。
- 原始指标：`recons/recall80_goal_20261027_235344/round_008_auxiliary/quarter/step512_development.json`，字段/行：`result`。
- [固定指标证据](best_0_evidence.json) · [权重元数据与哈希](best_0_checkpoint.json) · [原实验计划](best_0_plan.json)。
- 权重：`recons/recall80_goal_20261027_235344/round_008_auxiliary/quarter/step512.pt`；SHA256 `c489631d9fbe641bbfaf5c0c54c579da27349247a47b28c7e368294a43076d0c`。权重留在原项目，本次不向Git批量上传大检查点。

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
  "step": 512,
  "lane": "quarter",
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
    "tail_weight": 0.04034386474777693,
    "mid_weight": 0.8240454165997028,
    "new_weights_state": "locked",
    "new_gradient_rms": {
      "pre_guard_loss": 3.519685725838934,
      "tail_positive_huber": 87.2421556001989,
      "tail_false_positive_huber": 110.53622288473561,
      "tail_balanced": 152.3454995859868,
      "mid_protection": 4.271227841254646
    },
    "new_weight_scope": "8preselectedtrainwindows only; nooptimizerupdate; equalinitialmuheadgradientRMS1.0; notdevelopment-selected.",
    "positive_fp_gradient_cosine": 0.17617753477665446,
    "scope_new_weights": "8fixedtrainingwindows,positive-onlymu-gradientRMS1xbaseandmidRMS1xbase;commonall4,nooptimizerupdate/noQ2."
  },
  "regional_phase_config": {
    "layout": "frame1",
    "frame_ffn": 1536,
    "use_shape": false,
    "use_mass": false,
    "readout_kernel": "linear"
  },
  "stratified_quantity_loss_config": {
    "lower_BTC": [
      0.0,
      0.01,
      0.1,
      1.0,
      10.0,
      100.0
    ],
    "scales_BTC": [
      0.0044779047307967985,
      0.0472840202098522,
      0.4412717188339758,
      3.522815580004404,
      24.89547481913624,
      215.94657900798126
    ],
    "weight": 0.0,
    "multiplier": 0.0
  },
  "mid_lowpass_loss_config": {
    "weight": 5.167638744317654,
    "multiplier": 0.25,
    "kind": "mid_lowpass",
    "chunks": [
      16,
      64,
      256
    ]
  },
  "state_tensor_count": 246,
  "state_scalars": 11773094
}
```

</details>

原始预测：`recons/recall80_goal_20261027_235344/round_008_auxiliary/quarter/step512_profiles.npz`，SHA256 `3398f79f202f50793cf51344fb31158d381d515d689bd35d64a16d2036ab1ab4`。固定第一个保存案例，分钟[0, 128, 255]，完整2048档。它是案例可视化，不替代全测试统计；没有新增模型推理。

![真实黑色，预测绿色；完整档位](best_0_btc.png)

![真实黑色，预测绿色；完整档位](best_0_log.png)

![真实黑色，预测绿色；完整档位](best_0_price.png)

[图中精确数据](best_0_display_points.npz)；未做深度平均、峰值截断或按误差挑样本。

### 若优先大单F1

保留候选中F1最高点：`recons/recall80_goal_20261027_235344/relative_aggregation_early_pilot_20261028/control`，step=256，F1=58.67%，挂量RMSE=2.20143 BTC，价格RMSE=30.3350 USDT。它可能属于不同测试口径，不自动替代上面的最小挂量误差点。完整候选与口径见JSON。

## 同预算共同终点参考

该分支保留的共同终点评估中，最低挂量误差点为`recons/recall80_goal_20261027_235344/round_008_auxiliary/unit`，step=2048，挂量RMSE=2.01712 BTC、价格RMSE=54.2211 USDT。它与回顾性最低误差检查点是两个选型口径；不同来源或测试协议不能据此直接作因果比较。

## 所有保留候选与比较边界

![误差可视化](metrics.png)

图中最多展示10个低挂量误差点，只能在相同口径内比较；所有候选均保留在[candidates.json](candidates.json)。0步初始化不参与已训练分支排名；历史参考/未压缩模型不冒充压缩方案结果；训练拟合、合成移动峰和已见窗口不混入测试排名。历史2025Q1已重复查看，不称独立确认。

<!-- PARETO -->
## 价格与挂量的取舍

最小挂量误差可能以明显价格退化为代价；下面按每个评估口径分别列出价格/挂量二维非支配点（最多10个）。这是多目标取舍，不是因果或显著性结论。完整点仍在候选JSON中。

口径：`[256, ["2025-01-01", "2025-04-01"], 16, 16777216, 2.2170201]`

| 组别 | step / epoch | 价格RMSE | 挂量RMSE |
|---|---|---:|---:|
| quarter | 512 / None | 52.0615 | 1.95783 |
| control | 256 / None | 35.4673 | 2.05240 |
| mass | 1024 / None | 34.6865 | 2.08329 |
| control | 2048 / None | 33.9138 | 2.13972 |
| control | 512 / None | 30.3350 | 2.14647 |
