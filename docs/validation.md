# 本地验证

日期：2026-10-08。EID 0.1.0 改名后的验证环境：macOS ARM64，Python 3.11.8，NumPy 2.4.6，SciPy 1.17.1。

- 独立虚拟环境完成本地安装和开发安装。
- `pytest`：39 项通过。
- `ruff`：源码、测试与示例检查通过。
- wheel 和源码分发包构建成功；源码分发包包含使用说明、来源记录、示例和测试。
- wheel 普通安装后的依赖检查通过。
- 在项目目录外以隔离模式导入安装于 site-packages 的 `eid`，XOR 的 Ξ 为 1 bit；没有导入 EISyn 的 `yrd`、`utils`、`scripts` 或 `exp`。

## 小示例结果

| 示例 | EI | Ξ | 单位 | 闭合误差 | 容差内负 Ξ / 节点 Syn 数量 |
|---|---:|---:|---|---:|---|
| 离散 XOR | 1.000000 | 1.000000 | bits | 0 | 0 / 0 |
| 已知非线性响应 | 0.394690 | 0.394028 | nats | 0 | 0 / 0 |
| 学习模型 | 1.081994 | 0.507397 | nats | 0 | 0 / 0 |

离散示例的容差为 `1e-12 bits`，两个连续示例为 `0.03 nats`。连续结果只是这些固定示例及其干预、噪声、随机种子下的估计，不是对其他系统的精度保证。

GitHub 持续集成已在 Ubuntu / Python 3.10、3.11、3.12 上通过全部测试、构建和三个示例，见 [首次 CI 记录](https://github.com/Matthew-ymz/EID/actions/runs/37727627821)。未运行大型领域实验，也未验证 Windows。

## PyPI 0.1.1 独立安装验证

日期：2026-10-08。macOS ARM64 / Python 3.11.8。新建不继承系统 site-packages 的虚拟环境，在仓库以外的临时目录中执行安装和示例。

- 从 `https://pypi.org/simple` 安装 `eid-toolbox==0.1.1`，关闭安装缓存，没有使用本地 wheel 或开发安装。
- 导入路径指向该新环境的 `site-packages/eid`；发行版本与 `eid.__version__` 均为 `0.1.1`，EID / EISyn 源码目录未进入模块搜索路径。
- 安装的 wheel 来自 `files.pythonhosted.org`，SHA-256 与 PyPI 0.1.1 发布记录一致：`9eae2739ebc642060eff281c255bce1d419f1ab165f434c425564999e8714f3e`。
- 基础安装只增加工具箱、NumPy 2.4.6 和 SciPy 1.17.1；依赖检查通过，`py.typed` 已包含。
- 三个示例在隔离模式下运行，结果与上表一致；XOR 的解析 EI / Ξ 均为 1 bit，三个 SPT 的闭合误差均为 0，声明容差内的负值计数均为 0。
- [GitHub 0.1.1 发布流程](https://github.com/Matthew-ymz/EID/actions/runs/37729548561) 的构建和 PyPI 上传成功；[对应 CI](https://github.com/Matthew-ymz/EID/actions/runs/37729548393) 在 Python 3.10、3.11、3.12 上通过 39 项测试、三个示例和构建。

这是安装和接口集成验证；连续示例使用固定参数，不代表其他系统上的估计精度。
