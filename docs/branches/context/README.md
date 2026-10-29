# 数量Decoder读取全窗Z上下文

更新：2026-10-29T13:59:27.603099+08:00。输入范围：完整256分钟。

[返回总表](../README.md) · [模型谱系](../../MODEL_LINEAGE.md)

## 模型结构

以下图对应主要展示点的结构；分支其他可选结构见谱系说明，实际开关和精度以检查点配置为准。

![结构](architecture.svg)

```mermaid
flowchart TD
  N0["Full256 frame1 encoder<br/>same 8192 Gaussian"]
  N1["Z reshaped to 256 x 32<br/>only content input to both decoders"]
  N0 --> N1
  N2["Quantity-frame queries<br/>256-dimensional original expansion"]
  N1 --> N2
  N3["Residual Cross-Attention<br/>read local5 / local17 / global256 Z"]
  N2 --> N3
  N4["Quantity spatial decoder + price decoder<br/>zero output projection; no quality result"]
  N3 --> N4
```

- 瓶颈：唯一[B,64,128]不变。
- 架构变化：post-Z数量帧特征query对local5/local17/global256的Z进行Cross-Attention。
- 结论边界：Encoder本身已看全窗；扩大Decoder直接读Z的范围是可检验假设，不是瓶颈失败的证据。
- 原模型代码（相对OBreconstuction）：`model/autoencoder/quantity_context_vae.py`。

## 最佳结果：尚不存在

已找到实现/预检或只读诊断，但没有可排名的已训练测试重建结果。不能把源模型的0步重放、单窗CPU预检或数据匹配审计当作本分支的新模型成绩。

现有证据：[audit_0.json](audit_0.json), [audit_1.json](audit_1.json)。


## 所有保留候选与比较边界

![误差可视化](metrics.png)

图中最多展示10个低挂量误差点，只能在相同口径内比较；所有候选均保留在[candidates.json](candidates.json)。0步初始化不参与已训练分支排名；历史参考/未压缩模型不冒充压缩方案结果；训练拟合、合成移动峰和已见窗口不混入测试排名。历史2025Q1已重复查看，不称独立确认。

<!-- PARETO -->
## 价格与挂量的取舍

最小挂量误差可能以明显价格退化为代价；下面按每个评估口径分别列出价格/挂量二维非支配点（最多10个）。这是多目标取舍，不是因果或显著性结论。完整点仍在候选JSON中。
