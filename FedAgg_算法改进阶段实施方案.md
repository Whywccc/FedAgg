# FedAgg 算法改进阶段实施方案

本文档把当前 FedAgg 源码的算法升级任务拆成 6 个阶段。整体思路是：先保证原始复现与实验可信，再逐步加入自适应蒸馏、通信压缩、prototype 语义桥接、可信知识筛选、隐私保护与完整实验闭环。

当前项目已经完成“阶段 1”的初版代码改造，后续阶段应在真实实验验证阶段 1 有效后继续推进。

## 总体路线

原始 FedAgg 的核心是 BSBODP：父子节点通过固定 autoencoder 生成 bridge samples，并在 bridge samples 上做双向在线蒸馏。它的主要问题是：

- 固定轻量 AE 不随联邦训练演化，bridge sample 质量受限。
- 蒸馏权重固定，non-IID 或 teacher 不稳定时容易产生负迁移。
- logits 交换没有压缩，平台化扩展时通信压力会变大。
- 全量 latent/noise 缓存与转发会增加存储压力和隐私攻击面。
- 隐私论证主要依赖“AE 难以重建原图”，缺少形式化保护与攻击评估。

因此改进路线不是直接推翻 FedAgg，而是在保留“end-edge-cloud 三层结构”和“父子双向蒸馏”的基础上，把桥接机制升级成更稳定、更可压缩、更可信、更隐私可控的协议。

推荐阶段如下：

| 阶段 | 名称 | 核心目标 | 状态 |
| --- | --- | --- | --- |
| 阶段 1 | 稳定基线与低风险增强 | 修复实验可靠性问题，加入自适应 KD 和 logits 压缩统计 | 已完成初版 |
| 阶段 2 | Prototype Bank | 构建跨模型共享语义原型，替代部分全量 latent 缓存依赖 | 未开始 |
| 阶段 3 | Prototype-guided Bridge Carrier | 用 prototype 引导生成更贴近任务分布的 bridge samples | 未开始 |
| 阶段 4 | Reliability Scorer | 评估 teacher 知识可信度，过滤高熵或不可靠蒸馏信号 | 未开始 |
| 阶段 5 | Privacy Guard | 加入 DP、secure aggregation、denoising 与攻击评估 | 未开始 |
| 阶段 6 | 完整实验与论文消融 | 建立系统对比、消融、通信、隐私与鲁棒性实验 | 未开始 |

## 阶段 1：稳定基线与低风险增强

### 目标

先让当前 FedAgg 代码成为一个可靠实验基线，并加入两个低风险改进：

- adaptive distillation weight：解决固定蒸馏权重敏感问题。
- top-k quantized delta logits：为后续通信压缩实验打基础。

### 为什么先做这个阶段

这个阶段改动最小，但收益最大。当前源码中存在一些会影响实验可信度的问题，例如 `T_agg` 参数类型不稳定、client model 可能复用、`assert(label, label_)` 不是真正比较 label、设备写死为 `.cuda()`。如果不先修这些问题，后续 prototype、generator、DP 的实验结果很难解释。

### 处理过程

1. 修复基础工程问题：
   - 把 `--T_agg` 从可能返回 list 的参数改成稳定 float。
   - 修复 client 节点复用同一批模型对象的问题。
   - 把 `assert(label, label_)` 改成真正的 `torch.equal(...)` 检查。
   - 将 `.cuda()` 写死改成统一 `--device` 控制。
   - autoencoder 加载从源码目录读取 `params.pkl`，避免依赖当前运行目录。

2. 加入自适应 KD 权重：
   - 基础形式仍是 `CE + alpha * KL`。
   - 将 `alpha` 从固定值改成动态值。
   - 动态权重由三部分决定：
     - teacher confidence：teacher 输出熵越低，说明越确定。
     - teacher-student agreement：学生和教师输出越一致，说明知识越稳定。
     - round warmup：训练早期 teacher 不稳定，蒸馏强度逐步升高。

3. 加入 logits 压缩模拟：
   - 原始 dense logits 作为基准通信量。
   - 对 teacher logits 做按样本 mean baseline。
   - 只保留 delta logits 中绝对值最大的 top-k 项。
   - 可选量化到指定 bit 数。
   - 用重构后的 logits 参与 KD，使压缩不仅是统计，也真实影响训练信号。

4. 加入通信统计：
   - 统计 dense logits 通信量。
   - 统计压缩后等效通信量。
   - 输出压缩 ratio。

### 代码落点

- `fedagg.py`
  - `compute_adaptive_kd_weight(...)`
  - `prepare_teacher_logits_for_kd(...)`
  - `maybe_print_comm_stats(...)`
  - `BSBODP_dir(...)`
- `main_fedagg.py`
  - 新增 KD 与通信压缩参数。
- `autoencoder_pretrained.py`
  - 修复 AE 加载路径和设备管理。
- `utils.py`
  - 将 `scipy` 改成可选依赖，避免无关依赖阻塞主流程。

### 推荐运行方式

原始增强版：

```powershell
conda run -n cs224n python main_fedagg.py --comm_round 100
```

开启通信压缩统计：

```powershell
conda run -n cs224n python main_fedagg.py --comm_round 100 --logit_topk 3 --logit_quant_bits 8 --track_comm
```

关闭自适应 KD，回到固定权重对照：

```powershell
conda run -n cs224n python main_fedagg.py --comm_round 100 --kd_weight_mode fixed
```

### 预期收益

- 降低固定 KD 权重带来的训练不稳定。
- 给后续通信效率实验提供统计基础。
- 让实验结果更可信，方便后续做消融。

### 风险

- 在 CIFAR-10 上 logits 只有 10 类，通信压缩收益可能不明显。
- adaptive KD 的规则是启发式，需要通过消融验证是否稳定。
- 如果压缩过强，可能损伤 teacher soft label 信息。

### 验证指标

- cloud top-1 accuracy。
- edge/client accuracy，如果后续补充测试入口。
- 每轮通信量。
- 达到相同精度所需通信量。
- 固定 KD 与 adaptive KD 的对比。

## 阶段 2：Prototype Bank

### 目标

构建一个共享 prototype bank，用类原型表示每个类别的语义中心，为后续替代固定 AE 和构建 prototype-guided bridge 做准备。

### 为什么做 Prototype Bank

原始 FedAgg 依赖全量 latent/noise 缓存和固定 AE decoder 生成 bridge samples。这个设计有三个问题：

- 缓存量随客户端数据增长。
- latent/noise 和 label 长期存储会扩大隐私攻击面。
- 固定 AE 不知道当前任务的类别语义。

prototype 是更轻量、更稳定的语义摘要。它能把“上传大量 latent”转成“上传每类一个或少量 prototype”，更适合平台化扩展。

### 处理过程

1. 统一特征接口：
   - 当前模型 forward 返回 `(logits, feature)`。
   - CNN、ResNet10、ResNet18 的 feature 形状不同。
   - 需要给不同模型加 projection head，将 feature 映射到统一维度，例如 128 或 256。

2. 提取本地类原型：
   - 对每个 client 的本地 batch，提取 projected feature。
   - 按 label 分组求均值，得到 `p_i,c`。
   - 同时记录每个类别样本数 `n_i,c` 和 confidence 统计。

3. 父节点聚合 prototype：
   - edge 聚合子 client 的 prototype。
   - cloud 聚合 edge prototype。
   - 第一版使用样本数加权平均。
   - 后续可加入可靠性权重。

4. 维护 Prototype Bank：
   - 每个非叶节点维护自己的 prototype bank。
   - 可加入 EMA 更新，避免每轮原型剧烈波动。

### 建议新增模块

- `prototype.py`
  - `ProjectionHead`
  - `PrototypeBank`
  - `extract_local_prototypes(...)`
  - `aggregate_prototypes(...)`

### 需要修改的位置

- `model_zoo.py`
  - 暴露更稳定的 feature。
  - 或额外返回 projected feature。
- `fedagg.py`
  - `Node` 增加 `prototype_bank`。
  - `Init` 或每轮训练后更新 prototype。
  - 父节点聚合子节点 prototype。

### 预期收益

- 减少对全量 latent 缓存的依赖。
- 提供更稳定的类别语义锚点。
- 为 prototype-guided generator 提供条件输入。
- 提升 non-IID 下的跨节点语义对齐。

### 风险

- 不同模型的 feature 分布差异大，projection head 设计不好会导致 prototype 不可比。
- 如果某些 client 缺失某些类别，prototype bank 会稀疏。
- prototype 过粗，可能丢失类内多样性。

### 验证指标

- prototype 距离与类别一致性。
- 加 prototype loss 前后的精度变化。
- non-IID 下 cloud accuracy 是否更稳定。
- class-missing 场景下 prototype 是否有效。

## 阶段 3：Prototype-guided Bridge Carrier

### 目标

用 prototype 引导生成更贴近当前任务分布的 bridge samples，逐步替代固定 AE 生成桥样本的单一路径。

### 为什么做这个阶段

固定 AE 的主要问题是“静态、弱、不了解当前任务”。它只是从 latent/noise 解码图像，而不是围绕当前任务的类别语义和决策边界生成 bridge carrier。

Prototype-guided bridge 的目标不是生成看起来最真实的图片，而是生成对蒸馏最有用的桥接载体。

### 处理过程

1. 第一版保守实现：
   - 保留原 AE 路径作为 baseline。
   - 新增 prototype-conditioned bridge generator。
   - generator 输入：类别 prototype、噪声向量、类别 id。
   - generator 输出：bridge sample，形状与 CIFAR 图像一致。

2. 训练 generator：
   - 使用父节点或 broker 侧维护的 prototype bank。
   - 让 generator 生成样本后，teacher ensemble 对其输出低熵预测。
   - 加入多样性约束，避免生成样本模式坍塌。
   - 加入 prototype consistency，使生成样本的 feature 接近对应类 prototype。

3. 蒸馏时替换 bridge source：
   - 原始路径：`noise -> AE decoder -> fake_data`。
   - 新路径：`prototype + z -> generator -> bridge_data`。
   - 通过参数开关选择 `ae`、`prototype_generator` 或混合模式。

4. 后续增强：
   - 加入 EMA generator，缓解跨轮漂移。
   - 引入 active bridge sampling，只选择高价值 bridge samples。

### 建议新增模块

- `bridge_generator.py`
  - `PrototypeBridgeGenerator`
  - `BridgeSampler`
  - `GeneratorLoss`

### 需要修改的位置

- `fedagg.py`
  - `Node` 增加 generator 或 broker 引用。
  - `BSBODP_dir` 中 bridge sample 来源可插拔。
  - 增加 `--bridge_mode ae/prototype/mixed`。
- `main_fedagg.py`
  - 增加 generator 相关参数。

### 预期收益

- bridge samples 更贴近任务类别语义。
- non-IID 下蒸馏信号更稳定。
- 上层模型不再完全受固定 AE 表达能力限制。

### 风险

- generator 训练不稳定。
- 生成样本可能模式坍塌。
- 如果 prototype 质量差，generator 会被错误语义引导。
- 工程复杂度明显高于阶段 1 和阶段 2。

### 验证指标

- 固定 AE vs prototype generator 的 accuracy 对比。
- bridge sample 上 teacher entropy。
- bridge sample 的 prototype distance。
- 生成样本多样性。
- 是否减少达到目标精度的轮数。

## 阶段 4：Reliability Scorer

### 目标

对 teacher 知识进行可信度评估，过滤或降权不可靠的 logits，避免 non-IID 下错误知识污染学生模型。

### 为什么做这个阶段

在 non-IID 下，并不是每个 client 或 edge 都懂每个 bridge sample。如果直接平均或直接蒸馏所有 teacher logits，高熵、不确定、跨域偏移大的知识可能造成负迁移。

Reliability Scorer 的作用是回答一个问题：这个 teacher 对这个 bridge sample 的知识值不值得学？

### 处理过程

1. 定义可信度指标：
   - entropy：teacher 输出越高熵，越不可信。
   - confidence：最大 softmax 概率越高，通常越可信。
   - agreement：teacher 与 ensemble 或 student 越一致，越可信。
   - prototype distance：样本 feature 距离对应 prototype 越近，越可信。

2. 样本级权重：
   - 为每个 bridge sample 计算 reliability。
   - 将 KD loss 从 batch-level scalar 扩展成 sample-level weighted KD。

3. 客户端级权重：
   - 统计某个节点在一批 bridge samples 上的平均可靠性。
   - 对低质量 teacher 的贡献降权。

4. 过滤策略：
   - soft filtering：用权重连续缩放 KD loss。
   - hard filtering：entropy 高于阈值的样本不参与 KD。
   - 第一版建议使用 soft filtering，更稳定。

### 建议新增模块

- `reliability.py`
  - `compute_entropy_score(...)`
  - `compute_agreement_score(...)`
  - `compute_prototype_score(...)`
  - `combine_reliability(...)`

### 需要修改的位置

- `fedagg.py`
  - `compute_adaptive_kd_weight` 扩展到 sample-level。
  - `Loss_Non_Leaf` 和 `Loss_Leaf` 支持 per-sample KD weight。
- `prototype.py`
  - 提供 prototype distance 计算。

### 预期收益

- 减少错误 teacher 的负迁移。
- 提升强 non-IID、class-missing、domain-skew 下稳定性。
- 让自适应 KD 不只是调全局权重，而是能按样本选择可信知识。

### 风险

- 阈值或权重组合不合理可能过滤掉有价值的暗知识。
- 可靠性分数过于依赖当前模型输出，训练早期可能不准确。
- 需要更多消融来证明每个分数项有效。

### 验证指标

- teacher entropy 分布。
- 被降权样本比例。
- 强 non-IID 下 accuracy 改善。
- 去掉 reliability scorer 后的消融结果。

## 阶段 5：Privacy Guard

### 目标

将隐私保护从“经验性难以重建”升级为“可量化、可攻击评估”的隐私机制。

### 为什么做这个阶段

原始 FedAgg 的隐私论证依赖轻量 AE 不容易恢复原图，但系统仍然上传和缓存 latent/noise、label、logits。这些中间量仍可能被用于标签推断、成员推断或重建攻击。

Privacy Guard 需要解决两个问题：

- 上传什么变量更安全。
- 即使攻击者看到中间变量，泄露风险是否可量化降低。

### 处理过程

1. 先缩小通信变量：
   - 尽量从全量 latent/noise 转向 prototype、top-k delta logits。
   - 变量规模越小，DP 噪声对效用的破坏越可控。

2. 加入 clipping：
   - 对 prototype 或 delta logits 做范数裁剪。
   - 限制单个 client 对聚合结果的最大影响。

3. 加入 DP noise：
   - 对裁剪后的变量加 Gaussian noise。
   - 记录隐私预算，例如 epsilon、delta、noise multiplier。

4. 加入 secure aggregation：
   - 第一版可以先做模拟：只允许父节点看到聚合结果，不暴露单 client 上传值。
   - 后续再实现真实 SecAgg 协议。

5. 加入 denoising：
   - 对含噪聚合结果做简单去噪或 EMA 平滑。
   - 第一版优先做 EMA 平滑，不要直接上复杂神经去噪器。

6. 做攻击评估：
   - 标签推断。
   - 成员推断。
   - 重建攻击。

### 建议新增模块

- `privacy.py`
  - `clip_tensor(...)`
  - `add_gaussian_noise(...)`
  - `PrivacyAccountant`
  - `secure_aggregate_simulation(...)`
- `attacks.py`
  - `label_inference_attack(...)`
  - `membership_inference_attack(...)`
  - `reconstruction_attack(...)`

### 需要修改的位置

- `fedagg.py`
  - 上传 prototype 或 delta logits 前接入 Privacy Guard。
  - 聚合时只使用保护后的变量。
- `main_fedagg.py`
  - 增加隐私参数。

### 预期收益

- 论文隐私部分更有说服力。
- 能报告 privacy-utility tradeoff。
- 攻击评估可以证明改进不是只停留在理论描述。

### 风险

- DP 噪声会降低精度。
- Secure aggregation 真实实现复杂。
- 攻击评估工程量较大。
- 如果主算法还没稳定，过早加入隐私模块会让实验很难解释。

### 验证指标

- 主任务 accuracy。
- epsilon / delta。
- 标签推断准确率。
- 成员推断 AUC。
- 重建攻击 PSNR / SSIM / LPIPS。
- privacy-utility Pareto 曲线。

## 阶段 6：完整实验与论文消融

### 目标

把前面每个模块的收益用系统实验验证出来，形成论文或报告中完整、可信的证据链。

### 为什么最后做这个阶段

算法模块没有稳定前，过早做大规模实验会浪费大量时间。完整实验应该在阶段 1 到阶段 5 的功能都能稳定运行后进行。

### 实验设计

1. 基线方法：
   - 原始 FedAgg。
   - FedAgg + adaptive KD。
   - FedAgg + logits compression。
   - FedAgg + Prototype Bank。
   - FedAgg + Prototype-guided Bridge。
   - Full method。

2. 数据集：
   - CIFAR-10：对齐当前源码。
   - CIFAR-100：放大类别数，体现 logits 压缩价值。
   - 可选跨域数据集：验证平台化泛化能力。

3. non-IID 设置：
   - Dirichlet alpha = 3.0、1.0、0.3、0.1。
   - class-missing。
   - domain-skew，如果加入跨域数据集。

4. 模型异构：
   - client：CNN / MobileNet。
   - edge：ResNet10 / ResNet18。
   - cloud：ResNet18 / ViT-tiny，可选。

5. 消融实验：
   - 去掉 adaptive KD。
   - 去掉 logits compression。
   - 去掉 Prototype Bank。
   - 去掉 generator。
   - 去掉 Reliability Scorer。
   - 去掉 DP / denoising。

6. 通信实验：
   - 每轮通信量。
   - 总通信量。
   - 达到目标精度所需通信量。
   - accuracy-communication 曲线。

7. 隐私实验：
   - 无隐私保护 vs DP vs DP + denoising。
   - 攻击成功率对比。
   - privacy-utility 曲线。

### 主要指标

- cloud top-1 accuracy。
- edge/client accuracy。
- macro-F1。
- 收敛轮数。
- 总通信量。
- bytes-to-accuracy。
- teacher entropy。
- seed 方差。
- 标签推断准确率。
- 成员推断 AUC。
- 重建攻击质量。

### 推荐图表

- Accuracy vs Round。
- Accuracy vs Communication。
- Privacy-Utility Pareto。
- Ablation Bar Chart。
- Teacher Entropy Histogram。
- Prototype t-SNE 或 UMAP。
- Bridge sample 可视化。

### 预期论文故事

论文主线建议写成：

> FedAgg 通过 BSBODP 支持 end-edge-cloud 异构模型协同，但原始桥接机制依赖固定 AE 和全量 latent 缓存，存在桥样本质量、通信效率、知识可信度和隐私论证不足的问题。我们将 BSBODP 升级为 prototype-guided、可信、可压缩、隐私可控的桥接蒸馏协议，在保留 FedAgg 三层协同范式的同时，提高 non-IID 鲁棒性、通信效率和隐私安全性。

不要把创新点写成“调了几个超参”，而要写成“把 FedAgg 的桥接蒸馏机制从固定单载体升级为可演化、可信、压缩、隐私可控的桥接协议”。

## 阶段推进建议

最稳妥的推进顺序如下：

1. 先跑阶段 1 的真实实验，确认 adaptive KD 和 logits compression 是否带来收益。
2. 如果阶段 1 稳定，再做阶段 2 Prototype Bank。
3. Prototype Bank 有效后，再做阶段 3 generator，否则 generator 没有可靠语义条件。
4. generator 初步有效后，再做阶段 4 Reliability Scorer。
5. 主算法稳定后再加阶段 5 Privacy Guard。
6. 最后集中跑阶段 6 完整实验。

每个阶段都应保留开关，保证可以做消融：

```text
--kd_weight_mode fixed/adaptive
--logit_topk
--logit_quant_bits
--bridge_mode ae/prototype/mixed
--use_prototype_bank
--use_reliability
--use_dp
--use_denoising
```

## 当前项目状态

已完成：

- 自适应 KD 权重。
- 可选 top-k / 量化 delta logits。
- 通信统计。
- `T_agg` 参数修复。
- client model 复用问题修复。
- label assert 修复。
- device 管理修复。
- autoencoder 加载路径修复。

下一步建议：

1. 使用真实 CIFAR-10 跑阶段 1 的 baseline 对比。
2. 至少跑以下四组：
   - 原始固定 KD：`--kd_weight_mode fixed`
   - 自适应 KD：默认参数
   - 自适应 KD + top-k：`--logit_topk 3 --track_comm`
   - 自适应 KD + top-k + quant：`--logit_topk 3 --logit_quant_bits 8 --track_comm`
3. 如果阶段 1 的结果稳定，再进入阶段 2 Prototype Bank。
