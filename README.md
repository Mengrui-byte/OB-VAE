# OB-VAE

完整256分钟订单簿重建的可复现研究基线。当前推荐模型为 **GELU直接挂量读出，512步终点**；保留原模型参数名称、顺序和初始化实现。

```text
原始价量 [B,256,2,2048,2]
      ↓ 单Encoder
唯一Gaussian Z [B,64,128]，整窗8192个标量
      ├→ 价格Decoder → [B,256,2,2048]
      └→ 挂量Decoder＋GELU逐档残差读出 → [B,256,2,2048]
```

两个Decoder共享同一个Z，无Encoder跳接。推理用确定性μ、FP32。Gaussian参数化不意味着模型具有良好的先验生成能力：所选实验采用beta=0。

## 已有结果

相同16个2025Q1测试窗口，完整256分钟、双侧2048档；seed42。

| 模型 | 价格RMSE USDT | 挂量RMSE BTC | ≥100 BTC召回 | ≥100 BTC精确率 | F1 |
|---|---:|---:|---:|---:|---:|
| 推荐GELU512 | 30.3350 | 2.1127 | 77.38% | 47.21% | 58.64% |
| 匹配Identity512 | 30.3350 | 2.1169 | 76.82% | 46.60% | 58.01% |

推荐模型414个TP、463个FP、121个FN，分钟log挂量变化相关0.1459。仍有较多误报、小单背景偏高，未达到>80%召回/≥50%精确率的联合门槛。2025Q1已被反复查看，无Q2独立确认；不能据单种子小差异断言结构优势。

[模型和复现文档](docs/BASELINE.md) · [HTML结果](docs/index.html) · [原始测试指标](research/gelu/step512_development.json) · [匹配控制审计](research/common512_analysis.json)

## 安装与检查

Python ≥3.10。使用适合本机CUDA的PyTorch环境。

```bash
python -m pip install -e .
python -m obvae.cli inspect
python -m unittest discover -s tests -p 'test*.py'
```

仓库包含`weights/gelu512.pt`，约79 MiB，是从原237 MiB训练检查点导出的完整推理模型；没有优化器和RNG。张量内容与原检查点逐项一致，哈希见[manifest](weights/manifest.json)。使用`obvae.api.load_checkpoint`加载。

## 原生测试集评估

数据不上传到Git。需要原项目格式的未压缩分钟NPZ，包含时间、价量、档位数；价量为真实USDT/BTC。数据窗口自动避开缺失分钟。严格复现还需要相同版本原始数据及已保存索引。

```bash
python -m obvae.cli evaluate \
  --data /path/to/BTCUSDT.npz \
  --checkpoint weights/gelu512.pt \
  --device cuda:0 \
  --output recons/evaluation/test.json
```

该入口显式使用2025Q1和原16个测试索引。训练已见/训练期未抽中的历史结果分别保存在`research/gelu/step512_seen.json`、`step512_unseen.json`，不可与测试集混算。

## 后续受控续训

迁移时本地`artifacts/`保留原GELU/Identity完整训练状态、共同来源和teacher，Git忽略这些大文件。GitHub克隆仅包含推荐推理权重；在另一台机器续训，需单独转移这些本地文件及原始NPZ。推理权重不能代替Adam/RNG恢复。

```bash
python -m obvae.cli train \
  --data /path/to/BTCUSDT.npz \
  --resume artifacts/gelu512_training.pt \
  --teacher artifacts/teacher.pt \
  --steps 128 --device cuda:0 \
  --output recons/continuation_seed42
```

新的便携续训入口恢复四组Adam、历史抽样RNG、训练窗索引及冻结Encoder/价格分支，复用原目标和训练窗校准系数；保持完整窗口和唯一Z。训练期间不访问测试集。它是新的明确实验入口，尚未运行生产GPU续训，不承诺逐位重放历史CUDA轨迹；只有完整epoch边界状态可继续恢复。默认单种子，不报告跨种子方差。

## 目录

- `model/`：从所选实验冻结源码提取的完整模型依赖，纯Python模型代码。
- `obvae/`：便携加载、评估和明确续训CLI。
- `train_dual/util/`、`util/`：原数据、指标、损失构造及优化器恢复依赖。
- `tests/`：单Z、冻结代码加载、读出控制、优化器和分桶指标检查。
- `research/`：原配置、校准、索引、测试与对照结果、源码哈希及完整原运行源码归档。
- `weights/`：版本化推荐推理权重及清单。
- `artifacts/`：本地完整训练状态；不进Git。
- `recons/`：后续实验日志、配置、快照和结果；不进Git。

模型代码来自所选历史源码快照，未将后续FNO改动混入基线。`research/original_runtime.tar.gz`保留完整历史Python源码以供审计；其旧多轮调度脚本带有原环境路径，不是新的便携入口。后续FNO只有初始评估，不作为已验证重建基线。
