# EID

从 [EISyn](https://github.com/Matthew-ymz/EISyn) 抽取的独立 Python 工具库，用于有效信息（EI）、整合有效信息（Ξ）和 Synergy Partition Tree（SPT）分析。

当前版本为 `0.1.1`，处于 Alpha 阶段。发行包名为 `eid-toolbox`，导入名为 `eid`。源码仓库：[Matthew-ymz/EID](https://github.com/Matthew-ymz/EID)。

## 安装

需要 Python 3.10 或以上。基础依赖只有 NumPy 和 SciPy。

```bash
python -m pip install eid-toolbox
```

从源码安装时：

```bash
git clone https://github.com/Matthew-ymz/EID.git
cd EID
python -m pip install .
```

开发时使用：

```bash
python -m pip install -e '.[dev]'
python -m pytest
python -m build
```

安装后无需 EISyn 的源码、数据目录、模型权重或运行环境。

[`eid-toolbox 0.1.1`](https://pypi.org/project/eid-toolbox/0.1.1/) 已发布到 PyPI，代码使用 `import eid`。需要固定当前版本时，使用 `python -m pip install eid-toolbox==0.1.1`。`eid` 作为 PyPI 发行名被平台的相似名称规则拒绝，工具箱的项目名和 Python 导入名仍为 EID / `eid`。

## 当前可用入口

| 入口 | 输入 | 计算方式 |
|---|---|---|
| `analyze_tpm` | 完整二进制状态转移矩阵 | 独立均匀干预，精确离散 EI，可指定预测步数 |
| `analyze_dynamics` | 批量响应函数 | 明确的独立均匀区间干预、响应噪声、多项式三角 TM |
| `analyze_model` | 已训练且具有 `predict(X)` 的模型 | 对学习模型施加同一干预，调用连续分析入口 |

三个入口返回 `AnalysisResult`，包含整体 EI、Ξ、单源 EI、已查询的子集 EI、SPT、数值诊断和配置。`result.to_dict()` 可以导出为 JSON。

### 离散 XOR

```python
from eid import analyze_tpm, build_deterministic_boolean_tpm

tpm = build_deterministic_boolean_tpm(
    2, lambda state: (state[0] ^ state[1], state[1])
)
result = analyze_tpm(
    tpm, n_nodes=2, targets=[0], unit="bits", syn_tolerance=1e-12
)
print(result.ei, result.xi)  # 1.0, 1.0
```

行、列按照二进制状态的字典序排列。默认所有当前变量为源、所有未来变量为目标。未选中的当前变量仍在共同的全状态独立均匀干预下边缘化。`horizon` 先形成多步转移核，再计算各子集，不能先边缘化再反复迭代。

### 已知连续响应

```python
from eid import UniformBox, analyze_dynamics

result = analyze_dynamics(
    lambda x: x[:, 0] * x[:, 1],
    intervention=UniformBox([[-1, 1], [-1, 1]]),
    response_noise_std=0.3,
    syn_tolerance=0.03,
    degree=2,
    seed=7,
)
print(result.xi, result.unit)
```

响应函数接收 `[样本数, 输入维度]` 数组，返回同样样本数的目标。函数应提供所需时间间隔的完整响应；当前入口不自动积分 ODE，也不自动迭代连续映射。调用方可在函数内完成这些操作。

所有 EI 项使用同一批完整输入与响应样本。`response_noise_std` 是用户显式指定的独立高斯响应噪声，可以为标量或逐目标标准差；这是分析通道的一部分，必须有物理、测量或模型误差依据。当前连续入口要求正噪声，不自动用噪声下限替代确定性通道。真实过程噪声可由模拟器自行产生，但适配器的附加响应噪声仍需明确指定。

### 数据驱动模型

```python
from eid import UniformBox, analyze_model

# model 已完成拟合、验证，具有 predict(X) 方法。
result = analyze_model(
    model,
    intervention=UniformBox([[-1, 1], [-1, 1]]),
    response_noise_std=validated_noise_std,
    syn_tolerance=0.03,
    degree=1,
)
```

信息量描述对拟合模型的干预。模型的预测误差、干预支持集内的可靠性、历史窗口、时间尺度和遗漏变量需由调用方核查。只有观测相关性或较低预测误差，不能自动建立真实系统中的因果解释。

`examples/learned_model.py` 给出了不依赖外部数据的拟合、留出验证与分析流程。训练流程目前保留在示例或用户项目中，工具箱尚未提供通用时间序列自动训练器。

## 结果的含义

- `ei`：所选源集合到固定目标的 EI。
- `xi`：整体 EI 减去同一目标下各单源 EI；两源时等于二源 Syn。
- `tree.root`：SPT 的根节点。节点的 `xi_value`、`syn_value` 采用 `result.unit`。
- `tree.closure_error`：节点贡献之和相对根 Ξ 的闭合误差。
- `ei_table`：实际查询过的子集，不保证枚举全部非空子集；是否完整在诊断中记录。
- `diagnostics`：容差、容差内负值数量、最小值、搜索方式和查询数量。
- `metadata`：干预、单位以外的分析配置及估计器假设。

默认输出 nats，可指定 bits。`syn_tolerance` 必须显式给出，单位与结果一致。落在 `[-tolerance, 0)` 的原始估计保留并记录；低于阈值时抛出异常，不裁剪 Syn，不把数值误差解释成负协同。

谱候选所需的图权重把容差范围内的负二源 Ξ 视为数值零，并记录 `affinity_tolerance_zero_count`；原始 Ξ、EI 和树节点值保持保留。

标准 SPT 固定目标、干预与时间尺度，以最大化两个子节点保留的 Ξ 选择分裂，并递归到单变量。源集合不超过 `exact_max_size` 时逐节点穷举，否则使用现有谱候选策略。逐节点最优并不意味着整棵树全局最优；谱搜索只在生成的候选中比较。

高阶节点的 Syn 是所选树路径上的层级贡献，不是唯一的纯阶 PID 原子。

## 底层模块

- `eid.discrete`：离散 TPM、源/目标边缘化、布尔 TPM 构建、粗粒化辅助函数。
- `eid.estimators.transport_map`：仿射、多项式三角密度模型及 MI 估计。
- `eid.spt`：统一 SPT 构建、非负性检查、精确/谱/分层随机候选。
- `eid.compat`：抽取时保留的历史 Phi 命名适配层。

底层 TM 的 `mi_hat` 和 specific-MI 返回 bits，并带有单位字段；连续公共入口负责转换。仿射密度模型存在，但公共连续入口目前使用多项式三角 TM，不等同于补充材料 S1.2 的仿射加特征提升估计器。

复制的 SPT 还保留历史 `SIGNED` 诊断策略和非标准归一化目标，仅用于检查历史结果，不作为本项目公共 PEID 分析协议。历史 `xi_bits` / `syn_bits` 别名不能用于判断单位；使用 `xi_value` / `syn_value` 和外层声明的单位。

## 示例与验证

```bash
python examples/discrete_xor.py
python examples/known_dynamics.py
python examples/learned_model.py
python -m pytest
```

测试包括解析 XOR、单源 COPY、单位转换、多步转移、非连续源索引、模型适配一致性、SPT 闭合和非负性失败，以及从 EISyn 迁移的 TM/SPT 回归测试。

本次 macOS / Python 3.11 的 39 项测试、三个示例、安装与构建已通过。从 PyPI 安装到独立环境后的三个示例也已通过，记录见 [验证记录](https://github.com/Matthew-ymz/EID/blob/main/docs/validation.md)。

## 维护与来源

来源文件、内容哈希和抽取时的 EISyn 提交见 [迁移清单](https://github.com/Matthew-ymz/EID/blob/main/docs/source_manifest.json)。方法核对范围、估计器差异和当前未覆盖功能见 [方法与迁移说明](https://github.com/Matthew-ymz/EID/blob/main/docs/method_contract.md)。后续范围见 [路线图](https://github.com/Matthew-ymz/EID/blob/main/docs/roadmap.md)。

大型数据、模型权重、领域实验、论文图表与研究日志继续由 EISyn 维护。核心代码后续以本工具箱为维护入口，论文复现项目在迁移后固定调用具体版本。

本项目采用 [MIT License](https://github.com/Matthew-ymz/EID/blob/main/LICENSE)。发布流程见 [发布说明](https://github.com/Matthew-ymz/EID/blob/main/docs/publishing.md)。
