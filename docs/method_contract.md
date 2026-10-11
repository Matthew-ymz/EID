# 方法与迁移说明

最近方法核对日期：2026-10-11；首次抽取记录保留在下方。

## 论文依据

Zotero 父条目 `P6UJCVG8` 的标题已验证为 *Emergent hierarchical organization of causal interactions in complex systems*。

本次重新检索附件并读取全文中的相关部分：

| 附件 | 内容 | 核对位置 |
|---|---|---|
| `DXGC7JEA` | main text，19 页 | Methods，式（5）–（12），PDF 页 15–17 |
| `MWIWKSVG` | supplementary information，28 页 | S1.2，式（S7）–（S15）；S5.1–S5.2，式（S49）–（S51） |

2026-10-11 再次通过 Zotero 本地 API 验证父条目标题、重新列出附件并读取两份索引全文。
这次扩展还核对了 S1.1、S6–S8、S11、S12.1–S12.3 的实际估计协议。
当前仍仅有这组主文与补充材料；Zotero 的条目 `version` 是元数据修订号，不是稿件版本。

主文附件加入时间为 2026-10-02，附件元数据修改时间为 2026-10-04；补充附件加入时间为 2026-10-04。附件和全文未给出明确稿件版本或稿件日期，因此不能仅由这些时间确认“最新稿”。本次依据当前唯一可用的主文/补充附件，不作全仓库一致性已验证的声明。

## 0.2.0 方法覆盖

| 文稿位置／计算协议 | 工具箱实现 | 覆盖与复现边界 |
|---|---|---|
| Methods Eq.5；S1.1 S1–S3：有限状态 EI | `discrete`，`analyze_tpm` | 一般矩形 TPM 的均匀输入 EI；完整二进制系统的共同干预子集分解 |
| S1.2 S7–S8：有界均匀或独立矩约束高斯干预 | `UniformBox`、`IndependentGaussian` | 两者是不同干预，不能按相同方差宣称等价 |
| S1.2 S11、S13：仿射 TM 密度比 | `tm_affine` | 相同正则化协方差协议拟合联合和边缘密度；raw 平均 log-density ratio |
| S1.2 S12–S15：标量／双标量特征与偏差校正 | `tm_affine_lifted` | 原定特征维数 3／5，S14 correction；不擅自推广到任意高维特征 |
| S7：模块诊断，degree-3 TM、相位 Fourier 源块 | `tm_polynomial` + `source_groups/source_transforms` | 固定目标、样本和源块；Fourier 字典与学习动力学留在实验侧 |
| S8 S74：Oracle degree-1 三角 TM、周期源与未来状态 | `tm_polynomial`，`degree=1` | 完整过程响应可经 `analyze_samples`；周期状态的密度近似和 min_scale 明确记录 |
| S7 Fig.1d：共享条件神经 TM，分量内同一通道积分所有子集 | `tm_neural` | 可选 nflows；上下文 gate、共同 Sobol 点、无重采样校正；`components` 显式声明分量近似 |
| S12.1.2：UniCM first-order affine TM／历史输入分组 | `tm_affine_readout` | 原生坐标的已知均匀干预协方差；校准的标准化／单位源协方差；共享全源投影与残余 ridge；12 个月块由调用者登记 |
| S12.2.2：Gaussian TM、21 维历史、7 个网络块 | `tm_affine`、`analyze_linear_gaussian` | 密度拟合或声明的 Gaussian 线性通道均可；输入矩约束和残余协方差由调用者明确指定 |
| S12.3：独立经验窗口、PCA、degree-2 TM、70/30 留出评估 | `IndependentEmpirical`，`tm_polynomial`，`degree=2, train_fraction=0.7` | 保留独立块；PCA 与零膨胀前处理在训练数据上固定后由实验侧提供 |
| 本地 MGSTN 历史实现中的局部 Jacobian Gaussian 通道 | `local_gaussian`、`linear_gaussian` | 单独标注局部线性近似；不作为 S12.3 有限幅度分析的替代 |

额外按用户要求提供 `kde`、KSG-1（`ksg/knn`）和 `nis`。NIS 依据其原文 Appendix B
Eq.38 实现 square injective channel 的 Jacobian 熵近似，属于补充后端，不能冒充本文所有
子集 EI 的通用公式。KDE 与 KSG 为显式的非 TM 选择，带宽／邻域和数值限制见
[估计器说明](estimators.md)。观测比较中的 WMS、SURD、PCMCI CMIknn、SHAP 等不另称 EI 后端。

方法覆盖是计算协议覆盖，不是全文全部领域实验的数值复现。神经分量发现、0.25-SD
permutation-effect 阈值、领域 Fourier 字典、过程模拟、训练模型、PCA、Shapley 归因与图表
仍在调用方。S1.2 仿射特征近似与 S7 神经／多项式方案是不同协议，必须明确选择。

UniCM S12.1.2 明确使用已知原生干预协方差 `(16/3)I`；不能用普通样本 Gaussian
logdet 拟合直接宣称该协议已复现。`tm_affine_readout` 单独实现这一路径，校准可选择
`standardize=True` 的单位源矩。补充文字称残余 ridge 为 `1e-6`，当前本地
`compute_unicm_order_syn.py` 原生路径实际使用 `1e-6 × max(trace(R)/d, 1)`；新接口通过
`ridge_scaling="absolute"/"trace"` 显式区分，默认遵循手稿的固定值，诊断记录实际 ridge。
这处文字与实现的差异保留在记录中，不宣称全部领域运行细节一致。

## 公共分析协议

1. EI 是共同干预诱导分布下的源—目标互信息。所有源子集固定同一目标、时间尺度、干预分布、响应机制和估计协议。
2. Ξ 是联合 EI 减去单源 EI 总和。二源 Ξ 对应二源 Syn；更多源的 Ξ 不称为单个高阶 PID 原子。
3. SPT 按原始残余对应的标准目标选择分裂，递归到单变量，保留原始节点残余并计算闭合误差。
4. 公共分析默认 nats；允许显式选择 bits，容差与信息量同步使用该单位。
5. Syn 容差必须显式给出。容差内负值保留并计数，显著违反非负性时失败并报告值、阈值和数量。容差不是负 Syn 的理论证据。
6. 已训练模型的 EI 表示模型干预；不从观测数据自动宣称真实系统因果结构已识别。

## 抽取与必要适配

| 来源 | 目标 | 适配 |
|---|---|---|
| `scripts/spt.py` | `eid.spt` | 保留构建和候选算法；增加非有限输入与空/重复源检查 |
| `scripts/phi_hierarchy.py` | `eid.compat` | 修改导入指向新包 |
| `exp/TM/transport_map_density.py` | `eid.estimators.transport_map` | 明确 bits；修正 pairwise 的估计器标签；默认保留负 EI 估计；增加样本有限性检查 |
| `utils.py` | `eid.discrete` | 抽取离散函数与依赖闭包；去掉领域、绘图和训练依赖；补充合法对数底与有限值检查 |
| 两个核心测试文件 | 新项目 tests | 修改导入；移除只检查原实验 notebook 路径的测试 |

`analysis.py` 与 `interventions.py` 是小型适配层，调用复制的算法，共用干预样本和延迟计算的 EI 缓存。没有迁移原项目环境、数据、权重或绝对路径依赖。

公共谱候选接口只在图权重中将容差范围内的负二源 Ξ 设为数值零，以保持谱拉普拉斯的非负权重前提；`affinity_tolerance_zero_count` 记录受影响的无序源对数量，原始估计仍存储在 EI/Ξ 缓存中。显著负值在形成图权重前失败。

## 保留的近似与限制（0.2.0 更新）

- S1.2 的仿射 TM、特征提升和有限样本校正由 `tm_affine_lifted` 显式选择；默认多项式 TM 仍不使用该校正，两者不能作为同一个估计器。
- 同样本拟合／评估仍是默认；新接口支持多项式 TM、未校正仿射 TM 与 KDE 的固定留出密度评估。偏差和负残余仍可能触发失败，不能仅靠扩大容差解决估计器失配。
- 多项式条件回归保留 `ridge=1e-6` 与 `min_scale=1e-8` 的原有数值设置。公共响应通道另要求显式正噪声；这些数值设置不是物理噪声。历史 `jitter` 参数在多项式 MI 路径中未参与计算，未把它宣传为有效配置。
- 连续公共入口现支持独立均匀、高斯矩约束和经验块干预，以及显式相关 Gaussian 响应协方差。领域过程噪声与积分仍由模拟器生成；`analyze_samples` 可接收该完整通道。
- 已知动力学函数需要由调用方定义时间间隔、积分方案与实际目标。不能把导数、增量和未来状态默认为同一变量。
- 对独立、无约束二进制全状态的均匀干预，离散入口与方法一致。具有状态约束的系统需要重新定义干预，不能直接沿用独立源的理论保证。
- 谱候选只在候选集合中优化；全树构建仍是贪心选择。保留的归一化目标不是标准 SPT。
- 历史 signed 策略只作为底层诊断保留；公共入口总是采用非负性检查。显著负结果不能作为 PEID Syn 报告。
- 新增共享条件神经 TM 与变量分组；没有迁移全部领域模型、Shapley 或论文全部实验，也没有完成 EISyn 全仓库一致性审计。
- S1.2 的 Gaussian 模型和 Wishart correction 是有限样本近似；正则化与非 Gaussian 特征分布会限制校正的精确性。原始 EI 与 correction 分别保留。
- 神经 TM 对声明的拟合通道积分，而非直接给出真实系统 EI；有限积分和模型选择仍有误差。经验块或周期表征的支持可能奇异，密度家族的 min_scale／正则化是公开数值近似。

## 授权与发布状态

首次抽取时为本地预发布，源仓库未发现许可证。2026-10-08，用户明确要求发布到 GitHub 和 PyPI，并选择 MIT 许可证；本项目据此添加 MIT 许可和公开发布元数据。同日用户指定工具箱名称为 EID。PyPI 随后拒绝 `eid` 发行名，原因是与已有项目名称过于相似；发行名微调为 `eid-toolbox`，GitHub 仓库与 Python 导入名为 EID / `eid`。PEID 作为方法名称保留。实际发布结果见发布记录或远端页面。
