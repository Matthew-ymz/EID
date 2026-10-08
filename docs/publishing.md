# 发布流程

发行包：`eid`。导入名：`eid`。许可证：MIT。

GitHub 仓库：<https://github.com/Matthew-ymz/EID>。

## PyPI Trusted Publisher

在 PyPI 的 Account → Publishing 中登记 GitHub pending publisher：

| 字段 | 值 |
|---|---|
| PyPI Project Name | `eid` |
| Owner | `Matthew-ymz` |
| Repository name | `eid` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

登记完成后，将 GitHub 仓库变量 `PYPI_PUBLISHING_ENABLED` 设为 `true`，然后运行 GitHub Actions 的 **Release** workflow，或发布 GitHub release。工作流检查版本与 tag（存在时）一致，运行测试和三个示例，构建 wheel/sdist，检查元数据，然后使用 OIDC 上传到 PyPI。没有保存长期 PyPI 令牌。

首次完成 PyPI 账号配置前，工作流只构建并检查分发包；上传步骤保持关闭。

项目名称只有在首次成功上传后才在 PyPI 创建，pending publisher 本身不预留包名。

## 每次发布

1. 同步 `pyproject.toml` 和包内 `__version__`，更新变更记录。
2. 本地执行测试和示例，检查迁移来源或方法变更说明。
3. 推送到远端默认分支，等待持续集成通过。
4. 创建对应的 `vX.Y.Z` tag 和 GitHub release，或手动运行 Release workflow。
5. 查看上传结果，再从 PyPI 安装该明确版本并检查导入与解析 XOR。

## 首次发布边界

首发只公开目前的离散 TPM、已知批量响应和已训练模型适配功能。连续估计器、干预、噪声和 SPT 搜索限制以 README 与方法说明为准。
