# 订单簿重建模型谱系

整理时间：2026-10-29T13:34:49.962923+08:00。本次只整理研究模型分支，**不建立新的Git分支，不改变模型或历史实验**。

每个分支的最佳已记录实验、结构和图表见[分支结果图鉴](branches/README.md)。

## 先区分两个基准

**不限制latent容量的完整256分钟参考**：未压缩Cross模型G/L共1,064,960个标量，价格RMSE8.8736 USDT、挂量RMSE0.19861 BTC、大单召回93.83%、精确率97.86%，挂量分钟变化相关0.97814。

**当前唯一Gaussian8192约束下的综合基线**：GELU512，价格RMSE30.3350 USDT、挂量RMSE2.11268 BTC、大单召回77.38%、精确率47.21%，动态相关0.14587。这是当前`main`迁移的模型，不是所有容量模型里的绝对最优。

两者历史训练与容量不同，不能将差异直接归因于某一个模块，也不能称为等预算消融。旧16分钟Cross是第三种任务尺度，不与完整256分钟混用。

## 三条主要研究脉络

```text
A. 早期低维AE
   MLP/卷积AE → 局部查询V4 → patch挂量V5 → 峰V7/价格几何V8
   典型计划输入256分钟、latent16×64；不等于当前8192架构

B. Global/Local表示与压缩
   历史16分钟Dual → Cross/双塔融合
                    ├→ 窗口连续状态/PCA/学习压缩
                    │   └→ VQ/RQ/配对矩阵/小码本/R4×K8
                    ├→ 每帧16×1024矩阵与端到端残差MLP
                    ├→ 联合连续4096PCA → Decoder/Dice/形态损失对照
                    └→ 恢复完整256分钟：未压缩Cross参考 / 整窗16×1024

C. 完整256分钟、唯一Gaussian8192
   从头Unified patch4 [64×128]
      ├→ 区域内16档细节 / 峰辅助 / Query / RegionGrid
      └→ frame1时间组织 → Local hint教师蒸馏
            ├→ LocalPhase/RegionalPhase形状与总量头
            │     └→ 全分辨率Identity/GELU读出（GELU512：当前main）
            │           └→ log数量输入适配
            │                 ├→ 价格对齐时间输入（暂缓）
            │                 └→ 独立Fourier/FNO（训练中）
            ├→ FinePhase Encoder输入细节
            ├→ 小参数保价logit/乘性响应诊断
            └→ 数量Decoder局部/全窗Z读取（预检）
```

此图表示设计演进，不把每条箭头都当作检查点初始化或Python继承。Unified8192是从头初始化；hint的旧模型teacher只用于训练，推理不需要teacher。实际类继承与实验来源在JSON附录中分别保留。

## 模型分支总表

| 分支 | 输入范围 | latent/瓶颈 | 状态 |
|---|---|---|---|
| [早期MLP/卷积AE与1024瓶颈](#legacy) | 早期计划为256分钟；各次实际形状需单独核对 | 典型[B,16,64]，1024标量；非当前8192配置 | 历史实验/实现 |
| [局部查询V4与数量短路V5](#local_queries) | 计划256分钟 | [B,16,64]，1024标量 | 历史训练与拟合诊断 |
| [峰特征V7/价格几何V8](#peak_geometry) | 计划256分钟 | [B,16,64]，1024标量 | 历史实验，含纠正后的数据划分版本 |
| [Global/Local双Decoder VAE](#dual) | 历史16分钟 | G[B,16,64]＋L[B,16,2,128,16]；合计66560标量 | 历史训练 |
| [Cross融合与价量双塔](#cross) | 历史16分钟 | 融合后的Global/Local有序槽；不自动压缩成单向量 | 历史训练 |
| [连续窗口状态、PCA与学习逆映射](#continuous) | 历史16分钟 | 1024/512/256等候选维数；按具体试验读取 | 历史压缩实验 |
| [VQ/RQ、小码本与配对矩阵](#vq) | 历史16分钟 | 量化对象包括连续状态和Global/Local矩阵，不同试验不共用同一形状 | 历史实验，含冻结编解码器控制 |
| [每帧矩阵压缩与端到端残差深度](#matrix16) | 历史16分钟 | [B,16,1024]，16384标量 | 历史训练 |
| [Global/Local统一连续4096 PCA](#joint4096) | 历史16分钟 | 唯一连续[B,4096]；先还原Global/Local再解码 | 历史训练 |
| [恢复完整256分钟：Cross参考/矩阵压缩](#full256) | 实测完整256分钟 | 未压缩G/L合计1064960标量；压缩组[B,16,1024] | 完整256分钟历史训练 |
| [Global并入Local的敏感性/融合](#merge) | 完整256分钟 | 沿用Local网格，去掉独立Global生成输入；按各组实际配置 | 训练/只读敏感性控制并存 |
| [从头训练的唯一Gaussian8192](#gaussian8192) | 实测完整256分钟 | 唯一μ/logvar/Z均[B,64,128] | 完整窗主线起点 |
| [区域内16档直接细节/空间上采样路径](#phase) | 完整256分钟 | [B,64,128]不变 | 完整模型消融及表达能力诊断 |
| [峰辅助、坐标查询与区域Gaussian组织](#peak_region) | 完整256分钟 | 同8192；区域组按64时间×2侧×8区域×8通道组织后固定展平 | 训练拟合/测试角色需按实验区分 |
| [patch4→frame1时间组织](#frame1) | 完整256分钟 | 同[B,64,128]；frame1中间为256×32再重排 | 完整训练；frame_huber成为后续主线来源 |
| [Local教师蒸馏与挂量增强](#hint) | 完整256分钟 | 唯一[B,64,128]不变 | 完整训练并引出recall80连续研究 |
| [共享/分区域形状与总量头](#local_phase) | 完整256分钟 | 唯一[B,64,128]不变 | round004/005已完成；旧设计文档的待训练状态已过时 |
| [细粒度Encoder旁路特征适配](#fine_encoder) | 完整256分钟 | 唯一[B,64,128]不变 | round006已完成 |
| [保价logit/配对响应读出](#paired_readout) | 完整256分钟 | 唯一[B,64,128]不变 | 短试验/缓存诊断，按各自计划评价 |
| [数量Decoder读取全窗Z上下文](#context) | 完整256分钟 | 唯一[B,64,128]不变 | 实现与预检；尚无正式训练收益 |
| [GELU/Identity全分辨率逐档MLP](#direct) | 完整256分钟 | 唯一[B,64,128]不变；新增参数8919040 | 512步匹配对照已完成；GELU512为当前紧凑主线推荐 |
| [数量log1p/physical-log输入适配](#log_input) | 完整256分钟 | 唯一[B,64,128]不变 | round012完整2048步控制已完成 |
| [按档位/真实价格对齐的时间数量输入](#temporal_input) | 完整256分钟 | 唯一[B,64,128]不变 | 候选/预检已完成，正式round014_temporal因FNO优先而暂缓 |
| [独立Fourier/FNO全模型](#fourier) | 完整256分钟 | 唯一[B,64,128]不变 | 四组训练中；共同检查点见阶段结果表 |

## 各分支的目的、实现与证据

原项目路径均相对于`OBreconstuction/`；只有已迁入的新基线代码才位于本仓库。不要把文档列出的历史分支误认为本仓库已包含其全部运行资产。

<a id="legacy"></a>

### 早期MLP/卷积AE与1024瓶颈

- 上游设计：独立研究起点。
- 改动：线性RMS/首帧mid归一化；MLP/卷积重建；后续引出局部查询。
- 范围：早期计划为256分钟；各次实际形状需单独核对；典型[B,16,64]，1024标量；非当前8192配置。
- 记录状态：历史实验/实现。
- 结论边界：不能用这些旧模型的latent默认值覆盖当前主线。
- 源码：`model/factory.py`, `model/autoencoder/encoder.py`, `model/autoencoder/decoder.py`。
- 代表实验：`checkpoints/btc_norm_ab_spatial16x64_20261018_173307`。
- 原说明：`docs/models/README.md`。

<a id="local_queries"></a>

### 局部查询V4与数量短路V5

- 上游设计：早期MLP/卷积AE与1024瓶颈。
- 改动：8个局部空间查询、直接数量/patch残差/mass-share、pool norm消融。
- 范围：计划256分钟；[B,16,64]，1024标量。
- 记录状态：历史训练与拟合诊断。
- 结论边界：小样本overfit与完整测试分开；不能把量预测分支差异当作latent扩容。
- 源码：`model/autoencoder/geometry.py`。
- 代表实验：`checkpoints/btc_local_queries_v4_20261018_193330`, `checkpoints/btc_patch_quantity_v5_20261018_205710`, `checkpoints/btc_quantity_overfit_20261018_mass_share`。

<a id="peak_geometry"></a>

### 峰特征V7/价格几何V8

- 上游设计：局部查询V4与数量短路V5。
- 改动：Top-K/Soft-NMS峰特征、价格几何与历史价格对齐。
- 范围：计划256分钟；[B,16,64]，1024标量。
- 记录状态：历史实验，含纠正后的数据划分版本。
- 结论边界：当前价格对齐上下文不是首次提出价格对齐；旧V8也不是当前单8192架构的匹配控制。
- 源码：`model/autoencoder/geometry.py`, `model/encoders/geometry.py`。
- 代表实验：`checkpoints/btc_peak_v7_20261018_235000`, `checkpoints/btc_v8_corrected_2022_2024_val2025q1_20261021_142435`。

<a id="dual"></a>

### Global/Local双Decoder VAE

- 上游设计：独立研究起点。
- 改动：共享Encoder输出Global/Local；价格/数量独立Decoder；空间/数量/时间容量对照。
- 范围：历史16分钟；G[B,16,64]＋L[B,16,2,128,16]；合计66560标量。
- 记录状态：历史训练。
- 结论边界：dual首先指双Decoder，不等于双塔Encoder；16分钟结果不代表256分钟能力。
- 源码：`model/dual/orderbook_vae.py`, `model/experimental/dual_long.py`。
- 代表实验：`checkpoints/dual_btc_20261023_174910`, `checkpoints/dual_12h_20261024_004600`。
- 原说明：`docs/models/DUAL_TRAINING.md`, `docs/models/DUAL_12H.md`。

<a id="cross"></a>

### Cross融合与价量双塔

- 上游设计：Global/Local双Decoder VAE。
- 改动：control/cross/tower_mean/tower_cross；单Encoder与双塔来源分别比较。
- 范围：历史16分钟；融合后的Global/Local有序槽；不自动压缩成单向量。
- 记录状态：历史训练。
- 结论边界：Cross保留有序局部网格；统一token序列不等于统一低维瓶颈。
- 源码：`model/experimental/dual_fusion.py`, `model/encoders/dual_tower.py`, `model/layers/cross_latent.py`。
- 代表实验：`checkpoints/dual_fusion_20261024_113046_retry1`。
- 原说明：`docs/models/DUAL_FUSION.md`。

<a id="continuous"></a>

### 连续窗口状态、PCA与学习逆映射

- 上游设计：Cross融合与价量双塔。
- 改动：WindowState、分组PCA、残差逆映射、学习窗口压缩器。
- 范围：历史16分钟；1024/512/256等候选维数；按具体试验读取。
- 记录状态：历史压缩实验。
- 结论边界：PCA训练能量保留率不是重建保真率，也不能证明连续latent的无损下界。
- 源码：`model/experimental/window_state.py`, `model/experimental/pca_state.py`, `model/experimental/pca_residual_state.py`, `model/experimental/learned_window_state.py`。
- 代表实验：`checkpoints/window_state_20261024_121006`, `checkpoints/pca_state_20261024_222656`, `checkpoints/pca_inverse_20261024_230006`, `checkpoints/learned_state_20261025_112037`。
- 原说明：`docs/models/WINDOW_STATE.md`, `docs/models/GROUPED_PCA_STATE.md`, `docs/models/LEARNED_WINDOW_STATE.md`。

<a id="vq"></a>

### VQ/RQ、小码本与配对矩阵

- 上游设计：连续窗口状态、PCA与学习逆映射。
- 改动：原RQ256码×8级；Global/Local共同索引；Local冻结量化；小K；R4×K8。
- 范围：历史16分钟；量化对象包括连续状态和Global/Local矩阵，不同试验不共用同一形状。
- 记录状态：历史实验，含冻结编解码器控制。
- 结论边界：残差量化重建是各级码字相加，不是拼接；级数不会自动成为经济学语义。索引、码字和量化后latent必须区分。
- 源码：`model/autoencoder/latent_rqvae.py`, `model/experimental/rq_window_state.py`, `model/experimental/paired_matrix_state.py`, `model/experimental/local_paired_state.py`, `model/experimental/tensor_cluster_state.py`。
- 代表实验：`checkpoints/rq_latent_20261025_124230`, `checkpoints/paired_matrix_20261025_135139`, `checkpoints/local_quant_recon_20261025_150636`, `checkpoints/tensor_cluster_20261025_152112`, `recons/local8_k4_vs_k16_20261026`, `recons/local_r4_k8_20261026`。
- 原说明：`docs/models/RQ_LATENT.md`, `docs/models/PAIRED_MATRIX_RQ.md`, `docs/models/SMALL_CODEBOOK_RESULTS.md`。

<a id="matrix16"></a>

### 每帧矩阵压缩与端到端残差深度

- 上游设计：Cross融合与价量双塔。
- 改动：Global/Local每帧4160→1024；线性/非线性；Decoder适配与端到端；残差深度。
- 范围：历史16分钟；[B,16,1024]，16384标量。
- 记录状态：历史训练。
- 结论边界：该16×1024是每帧压缩后的历史16分钟结果，不能与后续整窗256分钟的同形状状态混淆。
- 源码：`model/experimental/frame_matrix_state.py`, `model/autoencoder/frame_matrix_codec.py`。
- 代表实验：`checkpoints/frame_matrix_20261025_154358`, `checkpoints/frame_matrix_e2e_20261025_160416`, `checkpoints/frame_residual_20261026_010323`。
- 原说明：`docs/models/FRAME_MATRIX_COMPRESSION.md`, `docs/models/FRAME_RESIDUAL_COMPRESSION.md`, `docs/models/WINDOW_LENGTH_CORRECTION.md`。

<a id="joint4096"></a>

### Global/Local统一连续4096 PCA

- 上游设计：Cross融合与价量双塔。
- 改动：固定联合PCA，Decoder重训；Dice/无符号形态/有符号差分损失控制。
- 范围：历史16分钟；唯一连续[B,4096]；先还原Global/Local再解码。
- 记录状态：历史训练。
- 结论边界：原始Local测试失真约70.7%；固定预算形态监督未恢复大单。不能据此推断学习型4096编码器必然失败。
- 源码：`model/autoencoder/joint_pca_codec.py`, `model/experimental/joint_pca_state.py`。
- 代表实验：`recons/joint4096_decoder_20261026`, `recons/joint4096_dice_20261026`, `recons/joint4096_morphology_20261026`, `recons/joint4096_signed_difference_20261026`。

<a id="full256"></a>

### 恢复完整256分钟：Cross参考/矩阵压缩

- 上游设计：Cross融合与价量双塔。
- 改动：共享/双塔完整窗编码；通道压缩＋时间压缩；浅层/残差MLP。
- 范围：实测完整256分钟；未压缩G/L合计1064960标量；压缩组[B,16,1024]。
- 记录状态：完整256分钟历史训练。
- 结论边界：未压缩参考重建质量强，但容量远大于8192；旧16分钟权重仅初始化，实际完整窗训练单独进行。
- 源码：`model/experimental/full_window_matrix.py`, `model/autoencoder/window_matrix_codec.py`。
- 代表实验：`checkpoints/dual256_20261026_090533`。
- 原说明：`docs/models/FULL_WINDOW_256.md`。

<a id="merge"></a>

### Global并入Local的敏感性/融合

- 上游设计：恢复完整256分钟：Cross参考/矩阵压缩。
- 改动：Global/Local依赖、逐帧Local、Cross消融、GlobalIntoLocal。
- 范围：完整256分钟；沿用Local网格，去掉独立Global生成输入；按各组实际配置。
- 记录状态：训练/只读敏感性控制并存。
- 结论边界：依赖/遮蔽审计用于确认生成通路，不直接代表新模型收益。
- 源码：`model/experimental/global_into_local.py`。
- 代表实验：`checkpoints/global_local_256_20261026_100413`, `checkpoints/local_frames_256_20261026_101054`, `checkpoints/cross_ablation_256_20261026_103308`, `checkpoints/merge_global_local_256_20261026_105211`。

<a id="gaussian8192"></a>

### 从头训练的唯一Gaussian8192

- 上游设计：恢复完整256分钟：Cross参考/矩阵压缩。
- 改动：共享Encoder、一次Z采样、双Decoder；确定性AE与3档KL；最初patch4组织。
- 范围：实测完整256分钟；唯一μ/logvar/Z均[B,64,128]。
- 记录状态：完整窗主线起点。
- 结论边界：这是重新初始化的新架构，不是把旧G/L直接改名；8192为标量容量，不是已证明有效信息维数。
- 源码：`model/autoencoder/unified_book_vae.py`, `model/encoders/unified_book.py`, `model/decoders/unified_book.py`。
- 代表实验：`checkpoints/unified8192_20261026_112402`。
- 原说明：`docs/models/UNIFIED_8192.md`。

<a id="phase"></a>

### 区域内16档直接细节/空间上采样路径

- 上游设计：从头训练的唯一Gaussian8192。
- 改动：区域共享修正 vs 16个独立offset；Encoder冻结/训练交叉对照。
- 范围：完整256分钟；[B,64,128]不变。
- 记录状态：完整模型消融及表达能力诊断。
- 结论边界：增加的是post-Z读出路径，不是Encoder跳连；Jacobian集中不等于网络没有表达能力。
- 源码：`model/autoencoder/phase_book_vae.py`, `model/decoders/phase_volume.py`。
- 代表实验：`recons/phase_detail_20261026_210157`, `recons/phase_encoder_20261026_231213`, `recons/decoder_isolation_recovery_20261027_112135`。
- 原说明：`docs/models/PHASE_DETAIL_DECODER.md`, `docs/models/DECODER_ISOLATION.md`。

<a id="peak_region"></a>

### 峰辅助、坐标查询与区域Gaussian组织

- 上游设计：从头训练的唯一Gaussian8192。
- 改动：显式峰监督、QueryDecoder、RegionGrid与量价/仅数量任务控制。
- 范围：完整256分钟；同8192；区域组按64时间×2侧×8区域×8通道组织后固定展平。
- 记录状态：训练拟合/测试角色需按实验区分。
- 结论边界：合成移动峰与4训练窗拟合不能作测试集泛化结果；区域组织仍只有一个Gaussian。
- 源码：`model/autoencoder/peak_book_vae.py`, `model/autoencoder/query_book_vae.py`, `model/autoencoder/region_grid_vae.py`。
- 代表实验：`recons/peak_aux_20261027_002331`, `recons/decoder_isolation_20261027_105856`, `recons/region_task_20261027_145807`。
- 原说明：`docs/models/PEAK_AUX_SUPERVISION.md`, `docs/models/REGION_TASK.md`。

<a id="frame1"></a>

### patch4→frame1时间组织

- 上游设计：从头训练的唯一Gaussian8192。
- 改动：patch/frame时间组织×Huber/激活前logit监督2×2。
- 范围：完整256分钟；同[B,64,128]；frame1中间为256×32再重排。
- 记录状态：完整训练；frame_huber成为后续主线来源。
- 结论边界：时间特征投影、注意力与Decoder展开均改变，不能把收益只归因于reshape。
- 源码：`model/autoencoder/time_layout_vae.py`, `model/encoders/frame_aligned.py`, `model/decoders/frame_aligned.py`。
- 代表实验：`recons/time_logit_20261027_124432`。
- 原说明：`docs/models/TIME_LAYOUT_LOGIT.md`。

<a id="hint"></a>

### Local教师蒸馏与挂量增强

- 上游设计：patch4→frame1时间组织。
- 改动：增加post-Z hint头；训练期蒸馏旧大latent模型；在线数量增强2×2。
- 范围：完整256分钟；唯一[B,64,128]不变。
- 记录状态：完整训练并引出recall80连续研究。
- 结论边界：teacher只在训练同一原始窗口上提供Local目标，推理无需teacher；增强是训练因素，不是新增latent。
- 源码：`model/autoencoder/local_hint_vae.py`, `model/decoders/local_hint.py`。
- 代表实验：`recons/local_hint_20261027_202957`。
- 原说明：`docs/models/LOCAL_HINT_TRAINING.md`。

<a id="local_phase"></a>

### 共享/分区域形状与总量头

- 上游设计：Local教师蒸馏与挂量增强。
- 改动：LocalPhase共享形状/总量头；RegionalPhase按区域独立参数；各自开关2×2。
- 范围：完整256分钟；唯一[B,64,128]不变。
- 记录状态：round004/005已完成；旧设计文档的待训练状态已过时。
- 结论边界：两组都从Z生成的种子读出；要区分参数化容量变化与损失分布KL。
- 源码：`model/autoencoder/local_phase_vae.py`, `model/autoencoder/regional_phase_vae.py`, `model/autoencoder/shape_mass_vae.py`。
- 代表实验：`recons/recall80_goal_20261027_235344/round_004_local_phase`, `recons/recall80_goal_20261027_235344/round_005_regional_phase`。
- 原说明：`docs/models/LOCAL_PHASE_READOUT.md`, `docs/models/REGIONAL_PHASE_VAE.md`。

<a id="fine_encoder"></a>

### 细粒度Encoder旁路特征适配

- 上游设计：Local教师蒸馏与挂量增强。
- 改动：在空间下采样前补充数量细节特征；linear/log1p开关与关闭路径控制。
- 范围：完整256分钟；唯一[B,64,128]不变。
- 记录状态：round006已完成。
- 结论边界：这里的特征旁路仍位于Encoder内部并通过唯一Z；不是Encoder到Decoder的越过瓶颈旁路。
- 源码：`model/autoencoder/fine_phase_vae.py`。
- 代表实验：`recons/recall80_goal_20261027_235344/round_006_fine_phase`。
- 原说明：`docs/models/FINE_PHASE_ENCODER.md`。

<a id="paired_readout"></a>

### 保价logit/配对响应读出

- 上游设计：Local教师蒸馏与挂量增强。
- 改动：小参数加性logit vs 乘性数量响应；shape/mass、frame/seed读出诊断。
- 范围：完整256分钟；唯一[B,64,128]不变。
- 记录状态：短试验/缓存诊断，按各自计划评价。
- 结论边界：冻结缓存最小二乘、原生模型训练、固定权重观察须分开，不把缓存拟合成绩移给神经模型。
- 源码：`model/autoencoder/paired_logit_vae.py`, `model/autoencoder/logit_readout_vae.py`。
- 代表实验：`recons/recall80_goal_20261027_235344/paired_response_space_v2`, `recons/recall80_goal_20261027_235344/decoder_readout_pilot`。
- 原说明：`docs/models/PAIRED_LOGIT_READOUT.md`, `docs/models/PAIRED_RESPONSE_SPACE.md`。

<a id="context"></a>

### 数量Decoder读取全窗Z上下文

- 上游设计：Local教师蒸馏与挂量增强。
- 改动：post-Z数量帧特征query对local5/local17/global256的Z进行Cross-Attention。
- 范围：完整256分钟；唯一[B,64,128]不变。
- 记录状态：实现与预检；尚无正式训练收益。
- 结论边界：Encoder本身已看全窗；扩大Decoder直接读Z的范围是可检验假设，不是瓶颈失败的证据。
- 源码：`model/autoencoder/quantity_context_vae.py`。
- 代表实验：`recons/recall80_goal_20261027_235344/quantity_context_preflight`。
- 原说明：`docs/models/QUANTITY_WINDOW_CONTEXT.md`。

<a id="direct"></a>

### GELU/Identity全分辨率逐档MLP

- 上游设计：共享/分区域形状与总量头。
- 改动：数量帧256维→2048→GELU或Identity→4096，修正每帧双侧逐档logit。
- 范围：完整256分钟；唯一[B,64,128]不变；新增参数8919040。
- 记录状态：512步匹配对照已完成；GELU512为当前紧凑主线推荐。
- 结论边界：仅把它称为8192约束下的综合基线。冻结Encoder/价格，价格相同是控制设计；并非全容量绝对最优。
- 源码：`model/autoencoder/direct_readout_vae.py`, `model/layers/direct_quantity_readout.py`。
- 代表实验：`recons/recall80_goal_20261027_235344/direct_readout_activation_pilot_20261029`。
- 原说明：`docs/models/DIRECT_QUANTITY_READOUT_CONTROL.md`。

<a id="log_input"></a>

### 数量log1p/physical-log输入适配

- 上游设计：GELU/Identity全分辨率逐档MLP。
- 改动：Encoder下采样前加入数量对比/差分/LoG输入通道；关闭/开启×两种变换。
- 范围：完整256分钟；唯一[B,64,128]不变。
- 记录状态：round012完整2048步控制已完成。
- 结论边界：这是输入表示结构改变；round013损失曲率/强度是单独训练消融，不是又一个模型。
- 源码：`model/autoencoder/log_quantity_input_vae.py`。
- 代表实验：`recons/recall80_goal_20261027_235344/round_012_log_quantity_input`, `recons/recall80_goal_20261027_235344/round_013_log_curvature_strength`。
- 原说明：`docs/models/LOG_QUANTITY_INPUT_CONTROL.md`。

<a id="temporal_input"></a>

### 按档位/真实价格对齐的时间数量输入

- 上游设计：数量log1p/physical-log输入适配。
- 改动：lag1/4、rank/price对齐上下文，在Encoder内加6个输入通道。
- 范围：完整256分钟；唯一[B,64,128]不变。
- 记录状态：候选/预检已完成，正式round014_temporal因FNO优先而暂缓。
- 结论边界：只读匹配审计或CPU预检不证明重建收益；整窗网络仍不能当作在线因果因子。
- 源码：`model/autoencoder/temporal_quantity_input_vae.py`。
- 代表实验：`recons/recall80_goal_20261027_235344/price_rank_alignment_20261029`, `recons/recall80_goal_20261027_235344/round_014_temporal_quantity_input`。
- 原说明：`docs/models/TEMPORAL_PRICE_QUANTITY_CONTEXT.md`。

<a id="fourier"></a>

### 独立Fourier/FNO全模型

- 上游设计：数量log1p/physical-log输入适配。
- 改动：完整独立Fourier包；off/local/低64箱/全1153箱谱读出；保留双Decoder。
- 范围：完整256分钟；唯一[B,64,128]不变。
- 记录状态：四组训练中；共同检查点见阶段结果表。
- 结论边界：不是只加一个FNO层包装旧路径；运行时不依赖model/train_dual/util。中间128步不等于2048终点，也不能据此宣布改进。
- 源码：`Fourier/model.py`, `Fourier/layers/spectral_quantity_readout.py`。
- 代表实验：`recons/recall80_goal_20261027_235344/round_014_spectral_quantity_decoder`。
- 原说明：`docs/models/FOURIER_STANDALONE.md`, `docs/models/SPECTRAL_QUANTITY_DECODER.md`。


## 训练控制单列，不伪装成新模型

以下多轮试验的主要差别是损失、精度或数据：mid保护/时间差分权重，大单误报归一化，分数量桶损失，相对log误差，局部有符号细节，Huber/半平方曲率与系数，BF16/FP32分支，增强开关与样本量，Dice和形态监督。这些应附着在对应架构节点，而不是每换一次损失就声称有一个全新网络。

| 正式轮次 | 类型 | 主要对照 | 原计划更新数 | 记录状态 |
|---|---|---|---:|---|
| round_001_mid_tail | 损失/精度/数据训练控制 | Explicitmidprotection x balancedlargequantity/falsepositiveloss2x2 | 4096 | complete |
| round_002_fp_rule | 损失/精度/数据训练控制 | Falsepositiveobjectivecontrolledchain | 2048 | complete |
| round_003_mid_delta | 损失/精度/数据训练控制 | Endtoend singlefactor one-minute mid-difference weight | 2048 | complete |
| round_004_local_phase | 模型/输入/读出结构 | Local16phase shape/mass architecture factorial | 2048 | complete |
| round_005_regional_phase | 模型/输入/读出结构 | Regional local16phase shape/mass architecture factorial | 2048 | complete |
| round_006_fine_phase | 模型/输入/读出结构 | Single Encoder ordered quantity mean residual: enabled x linear/log1p | 2048 | complete |
| round_007_quantity_factorial | 损失/精度/数据训练控制 | 原挂量增强开关 × 分数量级重建损失：控制变量2×2 | 2048 | complete |
| round_008_auxiliary | 损失/精度/数据训练控制 | 固定来源新增单一辅助重建目标系数对照 | 2048 | complete |
| round_009_relative_detail | 损失/精度/数据训练控制 | 相对log挂量误差×有符号log局部细节2x2 | 2048 | complete |
| round_010_branch_precision | 损失/精度/数据训练控制 | 完整Encoder×完整post-Z路径精度2x2 | 2048 | complete |
| round_011_relative_aggregation | 损失/精度/数据训练控制 | 相对log损失聚合：原目标/等桶/√计数/位置平均 | 2048 | complete |
| round_012_log_quantity_input | 模型/输入/读出结构 | Encoder输入适配器：开关×数量变换 | 2048 | complete |
| round_013_log_curvature_strength | 损失/精度/数据训练控制 | 物理log误差：Huber/半平方×源mu梯度低/高强度 | 2048 | stopped_after_saved_common1024_for_requested_FNO |
| round_014_spectral_quantity_decoder | 模型/输入/读出结构 | 独立Fourier主干：原模型/局部/低模态/全模态FNO | 2048 | running |
| round_014_temporal_quantity_input | 模型/输入/读出结构 | Encoder时间上下文：开关×档位/精确价格 | 2048 | deferred_user_FNO_priority |

第13轮计划2048但实际共同1024后停止，不能写成2048完成。第14轮FNO与第14轮时间输入是两个不同目录：前者实际训练，后者按FNO优先暂缓。预检目录及已排队控制器不计为正式训练成绩。

## 不同容量与当前FNO阶段结果

| 模型 | 口径 | 价格RMSE USDT | 挂量RMSE BTC | ≥100 BTC召回 | ≥100 BTC精确率 | 挂量分钟变化相关 |
|---|---|---:|---:|---:|---:|---:|
| 完整256分钟未压缩Cross参考 | 大latent质量标尺；不是8192约束 | 8.8736 | 0.19861 | 93.83% | 97.86% | 0.97814 |
| GELU512紧凑主线基线 | 唯一Gaussian8192；固定512终点 | 30.3350 | 2.11268 | 77.38% | 47.21% | 0.14587 |
| FNO/off | 单Gaussian8192；共同128步中间点，仍在训练 | 49.0542 | 2.11274 | 75.70% | 46.44% | 0.13873 |
| FNO/local | 单Gaussian8192；共同128步中间点，仍在训练 | 99.8163 | 2.18795 | 80.00% | 43.41% | 0.14072 |
| FNO/low | 单Gaussian8192；共同128步中间点，仍在训练 | 84.3551 | 2.19698 | 81.12% | 43.14% | 0.14150 |
| FNO/full | 单Gaussian8192；共同128步中间点，仍在训练 | 86.5038 | 2.16363 | 79.63% | 44.70% | 0.13960 |

FNO表使用所有四组已有的最大共同检查点128步，属于训练中阶段结果；不能冒充2048步终点或独立确认。所有表中测试为相同16个2025Q1完整窗口，历史命名development保留在原文件中，本次文档统一称测试集。测试时段已被反复查看，单seed42不报告跨种子方差。

## 如何继续读

- 高容量重建质量标尺：`full256`。
- 低维/离散表示：`continuous`、`vq`、`matrix16`、`joint4096`；注意它们有历史16分钟限制。
- 当前可复现8192基线：`direct`，详见[BASELINE.md](BASELINE.md)。
- 当前探索：`log_input`→`fourier`；时间价格对齐输入是独立候选，尚未训练。
- 单独诊断：缓存拟合、合成移动峰、固定Z/Decoder拟合；不能据此宣布原生模型完成目标。

本次索引了117个规范实验计划、24个模型研究分支和43条模型类定义。索引排除public/viewer、源码快照、预检复制目录和此次迁移包，避免重复计算。部分旧计划缺少真实形状审计，JSON中明确留空，不通过目录名补造结果。

[交互HTML谱系](model_lineage.html) · [完整证据和类继承JSON](model_lineage.json)
