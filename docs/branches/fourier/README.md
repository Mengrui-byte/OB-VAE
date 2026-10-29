# 独立Fourier/FNO全模型

更新：2026-10-29T13:59:27.603099+08:00。输入范围：完整256分钟。

[返回总表](../README.md) · [模型谱系](../../MODEL_LINEAGE.md)

## 模型结构

以下图对应主要展示点的结构；分支其他可选结构见谱系说明，实际开关和精度以检查点配置为准。

![结构](architecture.svg)

```mermaid
flowchart TD
  N0["Observed full256<br/>standalone Fourier package"]
  N1["Log-quantity input + frame1 encoder<br/>single Gaussian 64 x 128"]
  N0 --> N1
  N2["Dual decoders + original GELU path<br/>same shared Z; no external runtime imports"]
  N1 --> N2
  N3["Selected spectral mode: off<br/>width=16, layers=2; off means control"]
  N2 --> N3
  N4["Spectral correction bypassed<br/>delta=0; original GELU path only"]
  N3 --> N4
  N5["Ordered price / nonnegative quantity<br/>ongoing common-checkpoint comparison"]
  N4 --> N5
```

- 瓶颈：唯一[B,64,128]不变。
- 架构变化：完整独立Fourier包；off/local/低64箱/全1153箱谱读出；保留双Decoder。
- 结论边界：不是只加一个FNO层包装旧路径；运行时不依赖model/train_dual/util。中间128步不等于2048终点，也不能据此宣布改进。
- 原模型代码（相对OBreconstuction）：`Fourier/model.py`, `Fourier/layers/spectral_quantity_readout.py`。

## 各评估口径的最佳已记录检查点

“最佳”定义为本分支、相同窗口长度/测试时段/窗口数/有效计数/目标RMS下所有已保存正步数评估中挂量RMSE最低；这是回顾性选点，包含中间检查点，不冒充独立测试确认；价格、≥100BTC指标和动态相关同时报告，不声称所有指标同时最优。不同口径分别保留最佳点，不做跨口径总冠军。下表第一项只是较大测试样本口径的展示代表。

| 组别 | 步数/epoch | 口径 | 价格RMSE USDT | 挂量RMSE BTC | ≥100召回 | ≥100精确率 | F1 | 动态相关 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| off | 128 / None | 256分钟×16窗，['2025-01-01', '2025-04-01'] | 49.05422 | 2.11274 | 75.70% | 46.44% | 57.57% | 0.13873 |

### 最佳点1：off

- 实验：`recons/recall80_goal_20261027_235344/round_014_spectral_quantity_decoder`。
- 原始指标：`recons/recall80_goal_20261027_235344/round_014_spectral_quantity_decoder/off/step128_development.json`，字段/行：`result`。
- [固定指标证据](best_0_evidence.json) · [权重元数据与哈希](best_0_checkpoint.json) · [原实验计划](best_0_plan.json)。
- **该点是控制/参考路径，不代表新增模块改善。**
- FNO开启组中最低挂量RMSE为`full`，2.16363 BTC；价格RMSE 86.5038 USDT。若off更好，不能宣布FNO获益。
- 权重：`recons/recall80_goal_20261027_235344/round_014_spectral_quantity_decoder/off/step128.pt`；SHA256 `05a6482142e30f144fb080ddbb22f06111c4d355429fc4d8b4ab4833e01d24ea`。权重留在原项目，本次不向Git批量上传大检查点。

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
  "lane": "off",
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
    "weight": 0.0,
    "multiplier": 0.0,
    "kind": "mid_lowpass",
    "chunks": [
      16,
      64,
      256
    ]
  },
  "branch_precision_config": {
    "encoder_fp32": true,
    "decoders_fp32": true
  },
  "relative_detail_loss_config": {
    "relative_enabled": true,
    "detail_enabled": false,
    "relative_weight": 0.13141036760364427,
    "detail_weight": 0.0,
    "epsilon_BTC": 0.0001,
    "radii": [
      1,
      2,
      4,
      8
    ],
    "detail_RMS": [
      [
        3.3398533978083975,
        1.9911423466658287,
        2.4028968024686224,
        3.6778582566065494,
        5.403164714852515,
        7.252417799458921
      ],
      [
        3.386690419848797,
        2.0108467766612392,
        2.423846607693328,
        3.731398678944289,
        5.566561014275974,
        7.979593412640151
      ],
      [
        3.378195972597894,
        2.015644717078535,
        2.4331015387808184,
        3.706766817511218,
        5.498297567598686,
        7.828494133020023
      ],
      [
        3.360361959414919,
        2.0117003801308426,
        2.418240756585485,
        3.700888371724588,
        5.468387524643184,
        7.382823110720124
      ]
    ],
    "relative_count_power": 1.0,
    "relative_point_loss": "huber"
  },
  "direct_readout_config": {
    "readout_activation": "gelu",
    "readout_hidden": 2048,
    "readout_seed": 42
  },
  "log_quantity_input_config": {
    "input_enabled": true,
    "input_transform": "physical",
    "input_epsilon": 0.0001
  },
  "spectral_quantity_config": {
    "spectral_mode": "off",
    "spectral_width": 16,
    "spectral_layers": 2,
    "spectral_low_modes": 64,
    "spectral_padding": 256,
    "spectral_chunk_size": 32,
    "spectral_seed": 42
  },
  "state_tensor_count": 261,
  "state_scalars": 21874311
}
```

</details>

原始预测：`recons/recall80_goal_20261027_235344/round_014_spectral_quantity_decoder/off/step128_profiles.npz`，SHA256 `20f6062934280b078e11a96276248aff07b256d093126862d22b44074b9158a5`。固定第一个保存案例，分钟[0, 128, 255]，完整2048档。它是案例可视化，不替代全测试统计；没有新增模型推理。

![真实黑色，预测绿色；完整档位](best_0_btc.png)

![真实黑色，预测绿色；完整档位](best_0_log.png)

![真实黑色，预测绿色；完整档位](best_0_price.png)

[图中精确数据](best_0_display_points.npz)；未做深度平均、峰值截断或按误差挑样本。

### 若优先大单F1

保留候选中F1最高点：`recons/recall80_goal_20261027_235344/round_014_spectral_quantity_decoder/off`，step=128，F1=57.57%，挂量RMSE=2.11274 BTC，价格RMSE=49.0542 USDT。它可能属于不同测试口径，不自动替代上面的最小挂量误差点。完整候选与口径见JSON。

## 同预算共同终点参考

该分支保留的共同终点评估中，最低挂量误差点为`recons/recall80_goal_20261027_235344/round_014_spectral_quantity_decoder/off`，step=128，挂量RMSE=2.11274 BTC、价格RMSE=49.0542 USDT。它与回顾性最低误差检查点是两个选型口径；不同来源或测试协议不能据此直接作因果比较。

## 所有保留候选与比较边界

![误差可视化](metrics.png)

图中最多展示10个低挂量误差点，只能在相同口径内比较；所有候选均保留在[candidates.json](candidates.json)。0步初始化不参与已训练分支排名；显式稀疏地址诊断单独标明。历史参考/未压缩模型不冒充压缩方案结果；训练拟合、合成移动峰和已见窗口不混入测试排名。历史2025Q1已重复查看，不称独立确认。

<!-- PARETO -->
## 价格与挂量的取舍

最小挂量误差可能以明显价格退化为代价；下面按每个评估口径分别列出价格/挂量二维非支配点（最多10个）。这是多目标取舍，不是因果或显著性结论。完整点仍在候选JSON中。

口径：`[256, ["2025-01-01", "2025-04-01"], 16, 16777216, 2.2170201]`

| 组别 | step / epoch | 价格RMSE | 挂量RMSE |
|---|---|---:|---:|
| off | 128 / None | 49.0542 | 2.11274 |
