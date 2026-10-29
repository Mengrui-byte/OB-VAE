# GELU512基线说明

## 选择依据

从完整256分钟、相同16个Q1窗口的原生主线共同终点中，先要求价格RMSE≤82.1746 USDT、挂量RMSE≤2.3974 BTC、分钟mid变化RMSE≤22.0406 USDT，再按严格≥100 BTC F1排序。该规则为迁移前事后选型规则，不是原实验新预注册主指标，也不意味着完整质量门槛通过。GELU512是该候选池的综合推荐。

## 结构

原始价量经过单Encoder，输出唯一Gaussian μ/logσ² `[B,64,128]`。所选实验sample=False、beta=0，因此确定性μ用于重建。价格与数量Decoder读取同一Z。数量路径生成每帧256维特征，新增`Linear256→2048→GELU→Linear2048→4096`读出，输出双侧2048档的有符号logit修正；最终挂量`q=q0·Softplus(original_logit+delta)`。新增8,919,040个共享参数，不增加每窗latent。整模型20,691,748参数。

`model/autoencoder/direct_readout_vae.py`为组装入口，`model/layers/direct_quantity_readout.py`为新增读出；复制自原实验冻结源码，未改模型实现。

## 受控实验

GELU/Identity两组共用相同25088步来源、初始化、三组历史Adam加相同第四组新头、1024训练窗及抽样、seed42、无增强、FP32、lr1e-4、clip1、batch1/累积4和512更新。仅新增隐藏非线性不同。Encoder与价格分支冻结，价格相同来自设计。训练teacher仅用于相同输入的Local特征蒸馏，不是推理输入。

原目标、校准和源步骤保存在`research/plan.json`、`calibration.json`、`quantity_calibration.json`、`detail_calibration.json`。目标包括原价格/挂量/形态分布项、Local hint、固定大单/误报和mid保护，以及锁定系数0.13141036760364427的逐位置物理log相对误差。便携续训复用原`inherited_book_loss`构造，不以新近似MSE替换它。

## 推理

```python
import torch
from obvae.api import load_checkpoint, reconstruct

model, normalizer, metadata = load_checkpoint(device='cuda:0')
# raw['data']: float32[B,256,2,2048,2]，真实USDT/BTC
# raw['level_counts']: int64[B,256,2]
# raw['times']可选；默认连续分钟坐标
prediction = reconstruct(model, normalizer, raw, device='cuda:0')
# prediction: price, quantity, z, mask
```

Normalizer tick_size=0.01，price_scale=104.2454906668734；模型quantity_scale=7.517722085208762。旧checkpoint.config.arguments.precision的bf16是继承元数据，实际`branch_precision_config`两个分支均FP32，已由历史运行审计核对。

## 数据与评价

训练2022–2024，测试2025Q1，窗口256、stride256。`research/*_indices.npy`记录实际窗口索引。2025Q1已在历史研究中反复查看，不是新的未见确认；2025Q2未作为此次迁移评估读取。指标按同分钟、侧、档位对齐；按档位的时序挂量变化不等于追踪固定物理价格的增撤单。

推荐GELU512：价格RMSE30.3350、挂量RMSE2.1127、召回77.38%、精确率47.21%、F1 58.64%，TP/FP/FN=414/463/121。分钟log挂量变化相关0.1459；小单[0,0.01)真实均值0.003577、预测均值0.319829 BTC，背景明显偏高。没有无损压缩或先验生成质量结论。

## 版本化与原始状态

`weights/gelu512.pt`仅移除原237 MiB检查点中的优化器、RNG和抽样进度，保留全部模型张量及模型配置；源SHA256 `56ce8bbcb791367ea7ed07fe2bda969d8ec121e5c6510f881bc2571520ecc7df`。导出SHA256见`weights/manifest.json`。导出不是重训练，逐张量和参数顺序检查见`research/migration_validation.json`。

完整训练检查点、Identity对照、共同来源和teacher在迁移包本地`artifacts/`中，未上传Git。Identity推理权重也在本地`weights/identity512.pt`；Git仅跟踪推荐权重。原始数据仍由`--data`显式指定，仓库不依赖原项目路径。

新续训CLI经过导入、损失构造与CPU测试；未执行生产256×2048 GPU优化，不声称新入口已逐位复现历史训练。完整历史Python运行源码另以归档保留。后续需以同预算配对实验验证任何模型/损失修改。
