# 发布流程

发行包：`eid-toolbox`。导入名：`eid`。许可证：MIT。

GitHub 仓库：<https://github.com/Matthew-ymz/EID>。

2026-10-08，[EID 0.1.1](https://github.com/Matthew-ymz/EID/releases/tag/v0.1.1) 和 [PyPI eid-toolbox 0.1.1](https://pypi.org/project/eid-toolbox/0.1.1/) 已发布。首次 Trusted Publishing 上传成功，仓库变量 `PYPI_PUBLISHING_ENABLED` 已启用；独立安装结果见 [验证记录](validation.md)。

## PyPI Trusted Publisher

在 PyPI 的 Account → Publishing 中登记 GitHub pending publisher：

| 字段 | 值 |
|---|---|
| PyPI Project Name | `eid-toolbox` |
| Owner | `Matthew-ymz` |
| Repository name | `EID` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

登记完成后，将 GitHub 仓库变量 `PYPI_PUBLISHING_ENABLED` 设为 `true`，然后运行 GitHub Actions 的 **Release** workflow，或发布 GitHub release。工作流检查版本与 tag（存在时）一致，运行测试和三个示例，构建 wheel/sdist，检查元数据，然后使用 OIDC 上传到 PyPI。没有保存长期 PyPI 令牌。

首次完成 PyPI 账号配置前，工作流只构建并检查分发包；上传步骤保持关闭。

项目名称只有在首次成功上传后才在 PyPI 创建，pending publisher 本身不预留包名。

PyPI 拒绝了最初的发行名 `eid`，原因是与已有项目名称过于相似。`eid-toolbox` 已通过登记并成功发布。已发布的 GitHub `v0.1.0` 保留原状，名称修订采用 `0.1.1`。

## 每次发布

1. 同步 `pyproject.toml` 和包内 `__version__`，更新变更记录。
2. 本地执行测试和示例，检查迁移来源或方法变更说明。
3. 推送到远端默认分支，等待持续集成通过。
4. 创建对应的 `vX.Y.Z` tag 和 GitHub release，或手动运行 Release workflow。
5. 查看上传结果，再从 PyPI 安装该明确版本并检查导入与解析 XOR。

## 首次发布边界

首发只公开目前的离散 TPM、已知批量响应和已训练模型适配功能。连续估计器、干预、噪声和 SPT 搜索限制以 README 与方法说明为准。
