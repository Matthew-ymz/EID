# EI 估计器与切换

本地开发版 0.2.0。基础依赖仍为 NumPy、SciPy；神经 TM 使用可选安装：

```bash
python -m pip install -e '.[neural]'
```

所有新的公共接口默认 nats，可指定 `unit="bits"`。旧的
`eid.estimators.transport_map.estimate_mutual_information_transport_map` 仍返回 bits。
`InformationEstimate.value`、`AnalysisResult.ei_table` 均保留原始估计，没有非负裁剪。

## 选择入口

| `estimator` | 输入与入口 | 协议和主要参数 |
|---|---|---|
| `discrete` | `estimate_ei(tpm, ...)`；`analyze_tpm` | 有限输入状态上的均匀干预；一般矩形 TPM 的 EI 可直接计算，完整二进制系统可分解 |
| `tm_polynomial`（别名 `tm`） | 样本 EI、`analyze_samples/dynamics/model` | 多项式条件均值、高斯残差；`degree`、`joint_order`、`ridge`、`min_scale` |
| `tm_affine` | 同上 | 仿射 Gaussian TM 的样本平均密度比；`jitter`、`bias_correction`（默认 false） |
| `tm_affine_lifted` | 同上，限一或两维标量源 | 手稿式（S12）的特征提升；默认使用式（S14）的维数相关偏差校正 |
| `tm_affine_readout` | 样本 EI、`analyze_samples/dynamics/model` | UniCM 的共享全源投影、已知输入矩与残余 ridge；`standardize`、`residual_ridge`、`ridge_scaling` |
| `tm_neural` | 同上；需要声明干预 | 单个共享条件神经通道，各子集通过共同干预积分边缘化；可显式指定独立分量 |
| `gaussian` | 样本 EI、`analyze_samples/dynamics/model` | 协方差 logdet MI；`jitter`、`bias_correction`（默认 false） |
| `kde` | 同上 | Gaussian KDE 的联合／边缘密度比；`bandwidth` 默认 `"scott"` |
| `ksg`（别名 `knn`） | 同上 | KSG-1、Chebyshev 距离、严格半径边缘计数；`k` 默认 5 |
| `linear_gaussian` | `estimate_ei(coupling, ...)`；`analyze_linear_gaussian` | 独立高斯输入、线性通道、显式正定噪声的解析 EI；支持矩形通道和遗漏源边缘化 |
| `local_gaussian` | `estimate_ei(dynamics, ...)`；`analyze_dynamics/model` | 在 `center` 的 Jacobian 上应用高斯通道公式；需要 `IndependentGaussian` |
| `nis` | `estimate_ei(dynamics, ...)` | 方形、全局单射映射下的 Jacobian 熵近似；不用于一般子集 Ξ／SPT |

`available_estimators()` 返回可选名称与简述。用字符串切换，用
`EstimatorConfig(method, options)` 或 `estimator_options` 指定参数；二者不能重复传入。
不适用的参数会报错，不自动改用其他后端。连续分析默认优先 TM；非 TM 是调用者显式选择。

## 同一批干预样本上的切换

```python
import numpy as np
from eid import UniformBox, analyze_samples

intervention = UniformBox([[-1, 1], [-1, 1]])
rng = np.random.default_rng(7)
x = intervention.sample(2000, rng)
y = x[:, [0]] * x[:, [1]] + 0.3 * rng.normal(size=(len(x), 1))

result = analyze_samples(
    x, y, intervention=intervention,
    estimator="tm_affine_lifted",  # 可改成 tm、tm_affine、gaussian、kde、ksg
    syn_tolerance=0.01, unit="nats",
)
print(result.ei, result.xi, result.metadata["estimator_method"])
```

更换后端的同时固定完整输入、目标、响应噪声、时间尺度和干预。每一次分解的所有子集
必须使用同一个后端和配置。Syn 的容差单位与结果一致，容差内负数保留并计数，低于
`-syn_tolerance` 明确失败。不能通过自动扩大容差或投影结果解决模型失配。

`estimate_mi(x, y, ...)` 可分析普通观测样本；`estimate_ei(x, y, intervention=...)` 和
`analyze_samples` 要求调用者声明实际施加的干预。传入一个分布对象不能证明观测数据来自干预。
样本权重只保留在 `estimate_mi` 的多项式 TM 诊断路径，避免未经验证的重加权改变 PEID 干预。

## 干预、分组、表征与噪声

- `UniformBox(bounds)`：独立有界均匀干预。
- `IndependentGaussian(mean, std)`：独立均值／方差约束下的最大熵干预。
- `IndependentEmpirical(blocks)`：各注册块独立均匀抽取经验记录，保留块内依赖。
  这是注册离散记录集上的干预；不称为连续区间上的最大熵密度。
- `source_groups` 把所有原始输入列分成不重叠的玩家。`sources` 与 EI 表的键是组编号，
  `targets` 始终是响应列编号。未选中的组在共同输入集合下边缘化。经验块不能再拆成独立玩家。
- `source_transforms` 为每个组指定独立的表征函数，如一个相位的 `(cos θ, sin θ)` 或四个
  Fourier 特征。函数不跨组读取输入，拟合统计量由调用者在训练数据上预先固定。
- `response_noise_std` 和 `response_noise_cov` 二选一，后者允许相关的正定高斯残余协方差。
  无自动物理噪声下限。已有过程噪声或测量噪声的模拟器可生成完整 `y` 后用 `analyze_samples`。
- 时间间隔、ODE 积分、周期目标处理、PCA、历史窗口和学习模型训练留在调用方。

## 留出密度评估

`analyze_samples/dynamics/model(..., train_fraction=0.7)` 固定前 70% 干预样本拟合密度，
后 30% 评估全部子集。多项式 TM、仿射 TM 和 KDE 支持这一模式。源表征在分割前只应用
预先固定的函数；不要在该函数内部用完整样本拟合统计量。

单次 MI 可使用 `estimate_mi(..., evaluation_x=..., evaluation_y=...)`。Gaussian logdet
返回拟合模型的 MI，KSG 用共同样本上的邻域统计，两者不接受留出密度评估。
Wishart 校正来自同样本、无权重的 Gaussian 协方差模型；不能直接套用于留出密度比或多项式 TM。

## 手稿仿射特征协议

`tm_affine_lifted` 严格使用 S12：单标量源为 `(x, x², x³)`，双标量源为
`(x₁, x₂, x₁x₂, x₁², x₂²)`。联合和边缘模型独立拟合相同样本的正则化协方差，按 S13
计算平均 log-density ratio，减去 S14 校正；样本数必须大于提升后的源维数与目标维数之和。

这是特征空间中的 Gaussian 近似。它不能自动扩展到任意高维源或完整 SPT；超过两维时明确
拒绝，调用者应选择多项式／神经 TM 或另行定义并验证分组表征。`tm_affine` 不做特征提升。

## UniCM 的全源仿射读出

`tm_affine_readout` 对所有完整输入一次性拟合中心化的线性投影，计算残余协方差，
然后使用声明干预的已知输入矩重建目标协方差。每个源组的贡献为
$\mathbf{B}_g^{\mathsf T}\mathbf{C}_g\mathbf{B}_g$，所有子集都从该同一通道边缘化遗漏组。
均匀输入的方差为区间宽度平方除以 12；经验干预保留注册块内部的协方差。
这是一阶 Gaussian 读出近似，不能称为均匀输入或非线性响应的精确 MI。

```python
result = analyze_samples(
    x, y, intervention=intervention, estimator="tm_affine_readout",
    estimator_options={"standardize": False, "residual_ridge": 1e-6},
    syn_tolerance=1e-8 * 0.6931471805599453, unit="nats",
)
```

S12.1.2 的原生坐标协议使用 `standardize=False`。校准协议使用 `standardize=True`，
按共同样本的 `ddof=1` 标准差缩放源和目标，并显式用单位源协方差替代样本相关矩。
单位协方差协议限独立原始坐标，不能用于有块内依赖的经验干预。
残余 ridge 只加一次，不加到输入协方差。默认 `ridge_scaling="absolute"` 使用手稿标明的
固定 ridge；`"trace"` 使用本地 UniCM 原生实现的 `ridge × max(trace(R)/d, 1)`。
这一原生实现与手稿表述的缩放差异已记录，不自动混用。

读出要求完整源矩阵满秩，不接受留出密度评估或任意源表征；拟合和归一化统计量、
已知输入协方差、实际应用的 ridge、投影矩阵和残余协方差全部保存在诊断中。
它与 `tm_affine` 的每个子集独立密度拟合是不同协议。

## 共享条件神经 TM

```python
result = analyze_samples(
    x, y, intervention=intervention, estimator="tm_neural",
    estimator_options={
        "epochs": 100, "train_fraction": 0.8,
        "hidden_features": 48, "layers": 2,
        "evaluation_count": 512, "marginal_samples": 256,
        "context_transform": None,  # 可传入固定的领域特征字典
    },
    syn_tolerance=0.069315, unit="nats",
)
```

后端拟合共享的 conditional masked affine autoregressive flow，依据留出似然选择最佳 epoch。
目标按训练部分的逐坐标均值／标准差标准化。默认使用手稿中的两倍 paired SEM 规则选择是否
保留源上下文，相对的无条件基线是 degree-3 目标密度；也可显式 `context_selection=False`。
未保留上下文意味着选择一个源无关的通道，不是把负 Syn 裁成零。

随后从同一个拟合通道采样目标，使用固定的 scrambled Sobol 干预点计算
`log p_TM(y|x_A) - log p_TM(y)`。分母、积分点和估计预算在全部源子集间共享，没有重采样
偏差校正或非负投影。结果描述该拟合通道；`diagnostics` 的 pointwise SEM 仅描述评估散布，
不包含训练模型或有限边缘积分的全部不确定性。

`components=[{"sources": [0, 1], "targets": [0, 1], "context_transform": fn}, ...]` 可声明
互相独立的学习通道分量。source IDs 为组编号，target IDs 为已选目标集合内的局部列编号。
分量必须分别完整分割所有源组与目标。每个分量拟合一次，整体 EI 按独立分量求和。
分量发现、排列作用阈值、领域 Fourier 字典与噪声是否支持分量独立假设由实验端负责；该声明
是一种显式学习通道近似，不可自动当成物理系统的真实独立性。

## 高斯解析通道与局部近似

对于 `coupling` 指定的矩阵 **A**，列对应输入、行对应目标。声明
**x** 的独立高斯方差和正定残余协方差 **R** 后，解析计算
`½[log det Cov(y) − log det Cov(y|x_A)]`。条件协方差包含所有遗漏源的信号协方差。
均匀输入与高斯输入即使方差相同也不能作为同一个精确通道。

```python
from eid import IndependentGaussian, analyze_linear_gaussian

result = analyze_linear_gaussian(
    [[1, 1]], intervention=IndependentGaussian([0, 0], [1, 1]),
    response_noise_cov=[[0.25]], syn_tolerance=1e-12,
)
```

`local_gaussian` 在指定 `center`（默认输入均值）处以批量中心差分计算 Jacobian，或接受
`jacobian(points)` 返回 `[samples, targets, sources]`。结果明确标为局部线性近似。

## NIS 公式的边界

本库采用 [NIS Appendix B，Eq.38](https://arxiv.org/html/2201.10154v2) 的熵形式：
均匀输入熵加平均 `log|det Jacobian|`，减去显式高斯噪声熵。
这是小噪声／边界近似，要求全局单射的方形映射；采样点处非奇异并不能证明全局单射。

```python
from eid import estimate_ei

estimate = estimate_ei(
    lambda x: 2 * x, estimator="nis", intervention=UniformBox([[-2, 2]]),
    response_noise_cov=[[0.01]],
    estimator_options={"assume_injective": True, "dimension_average": False},
)
```

`dimension_average=True` 返回每输入维度的信息量，并记录归一化维数。总 EI 与每维 EI 不能
混入同一个 PEID 分解。矩形／奇异映射明确失败；不会以伪行列式或零值静默代替。
大噪声下公式可能返回负近似，原值保留，不能解释为真实 EI 或 Syn 为负。
因此 NIS 可由统一 `estimate_ei` 选择，但不强行接入任意子集的 Ξ／SPT。

## KDE 与 KSG

这些是用户要求的额外非参数后端，不是手稿所有实验的默认协议。KDE 的带宽、边界和高维
稀疏性影响估计；默认同样本评估也有自包含偏差，支持留出评估但不声称无偏。
KSG 实现 [原论文 Eq.8 的 algorithm 1](https://arxiv.org/html/cond-mat/0305641v1)。它默认
逐坐标标准化，不静默裁剪负估计。重复行／零距离明确报错；可显式指定
`tie_noise`（标准化坐标下的 Gaussian 噪声标准差，若不标准化则为原坐标单位）和 `seed`。
这改变了采样问题，诊断记录受影响的原始重复数；不声称适用于混合离散连续分布。

完整可运行示例：`python examples/estimator_switching.py`；加 `--neural` 检查可选神经路径。
文稿覆盖与尚未复现的领域细节见 [方法核对记录](method_contract.md)。
