# 固定权重 EI / MI 读取器边界

本实验校准已知真值下的信息读取器。它不训练 reweight 权重，也不宣称已从 HCP 观测中识别独立干预响应。

历史扫描和独立验证：[结果报告](../../docs/reports/ei_mi_reader_boundary_results_2026-10-29.md)。已有结果只需运行 `analyze.py` 重新生成图与统计；复跑从新目录保存，避免混用原始缓存的代码指纹。

执行前冻结的完整设计见 [protocol](../../docs/reports/ei_mi_boundary_benchmark_protocol_2026-10-28.md)。所有信息量为 **nats**，所有估计负值保留。

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg python experiments/reader_boundary/run.py
python experiments/reader_boundary/analyze.py
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/reader_boundary/validate.py
```

先安装本仓库及实验绘图所需的 Matplotlib；基础工具包依赖未变。`run.py` 默认逐格点保存 JSON，重跑时只复用科学代码与配置指纹一致的缓存。`analyze.py` 从缓存重建汇总及 `fig/reader_boundary_20261029/` 中的图。`validate.py` 在主扫描完成并汇总后，按预先声明规则选择边界候选及下一个更大 N，使用 100 个独立新种子验证，不重调方法。

`manifest.elapsed_seconds` 是该次启动的用时；重启后会重新计时。每格点缓存的 `elapsed_seconds` 保留该格实际计算用时。报告记录的是本次从头完成扫描与验证的耗时。

- `methods.py`：生成干预／自然 Gaussian 输入、解析真值、优化后的全协方差和结构化 Gaussian/TM。
- `run.py`：63 点 Gaussian 全扫描及 9 点非线性补充，默认每点 20 次。
- `analyze.py`：零与非零目标配对通过率、95% 精确二项区间、持续通过的候选边界、结果图。
- `validate.py`：独立种子的边界复核；100 次通过率的置信下界按单点解释。
- `../../tests/test_reader_boundary.py`：与公共接口等价性、解析真值、正交不变性及边界选择逻辑的核对。

Gaussian 精确解析去偏只用于无权重、iid、联合 Gaussian、无 ridge 且 N>2d 的读数。结构方法使用预先给定的独立块；稠密旋转和乘积通道明确检验结构／分布假设失配。KSG 作为普通非参数 MI 对照，只扫描 d≤20，其余区域未知。三次多项式 TM 补充也依赖已知独立输入配对块，其边缘密度是近似。

“可靠”同时要求每个随机种子的零信息总误差≤0.05 nats、信号误差≤max(0.05 nats, 真值的10%)。另有明确放宽零信息误差的每维精度分析。两种标准不能混报，20 次经验通过不能当作普遍可靠性定理。
