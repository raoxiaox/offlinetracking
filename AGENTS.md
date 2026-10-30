# AGENTS.md

本文件给 AI 编码助手（Claude Code、Codex、Cursor 等）阅读，规定在本仓库中的工作方式。人类读者请看 [README.md](README.md)。

## 1. 项目概述

- 任务：离线多目标跟踪（MOT）。输入完整视频的逐帧检测结果，输出全局一致的轨迹。
- 语言：Python 3.10+；主要依赖 numpy、scipy、PyTorch、OpenCV、TrackEval。
- 核心流程：检测加载 → 外观/运动特征 → 数据关联（匈牙利 / 图优化）→ 轨迹插值与平滑 → 评估。

## 2. 常用命令

```bash
pip install -e .                                     # 安装
pytest -q                                            # 全部测试，提交前必须通过
ruff check . && ruff format .                        # 代码检查与格式化
python scripts/track.py --config configs/<exp>.yaml  # 跟踪
python scripts/eval.py  --config configs/<exp>.yaml  # 评估
```

## 3. 编码准则

> 改编自 [forrestchang/andrej-karpathy-skills](https://github.com/forrestchang/andrej-karpathy-skills) 的 CLAUDE.md（MIT）。

1. **先想再写**：写代码前说明假设；需求有多种理解时列出来，不要默默选一种；不清楚就先问。
2. **简单优先**：只实现被要求的内容；不为一次性代码加抽象，不加“以后可能用到”的参数。
3. **外科手术式修改**：只改与任务相关的代码，沿用现有风格；不顺手重构无关代码，看到的问题在回复中指出即可。
4. **目标驱动**：把任务转成可验证的目标（如先写能复现 bug 的测试，再修复）；多步骤任务先给出简短计划和每步的验证方式。

## 4. 算法工程规范

### 4.1 可复现性
- 所有超参数放在 `configs/*.yaml`，代码中不写魔法数字；一次实验对应一个配置文件。
- 固定随机种子（`random`、`numpy`、`torch`），种子写在配置里。
- 每次运行把以下内容保存到 `outputs/<exp_name>/<timestamp>/`：生效的完整配置、git commit hash、日志、指标结果。

### 4.2 实验流程
- 先在小子集上试跑（如 1–2 个序列、前 100 帧）确认流程无误，再跑全量。
- 不得声称“效果提升”除非给出对比数据：基线与新方法在**同一数据划分、同一评估脚本**下的 HOTA / IDF1 / MOTA。
- 指标出现异常（突增、突降、NaN）时先排查 bug（数据泄漏、帧号偏移、ID 重复），不要直接报告为结论。
- 严禁用测试集调参；调参只用验证集。

### 4.3 跟踪任务特有约定
- 帧号从 1 开始（MOTChallenge 约定）；坐标格式统一为 `x, y, w, h`（左上角 + 宽高），内部转换要在函数名或注释中写明。
- 轨迹 ID 在单个序列内唯一且为正整数。
- 关联代价矩阵、门控阈值、轨迹最短长度等关键阈值必须来自配置。
- 修改关联或后处理逻辑时，补充或更新 `tests/` 中的小规模合成数据测试（几帧、几个目标即可）。

### 4.4 代码与性能
- 张量/数组操作优先向量化，避免逐元素 Python 循环；在函数文档中注明关键数组的 shape。
- 新增依赖前先说明理由。
- 大文件（数据、权重、输出）不入库，路径通过配置或环境变量传入。

## 5. 边界

- **可以直接做**：读代码、跑测试、跑小规模实验、修改 `src/`、`configs/`、`scripts/`、`tests/`。
- **先问再做**：删除文件、新增依赖、修改评估脚本或指标计算方式、跑耗时超过 30 分钟的全量实验。
- **禁止**：提交数据集、模型权重或密钥；修改测试集标注；为让测试通过而删改测试断言。

## 6. 提交规范

- 提交前运行 `ruff check .` 与 `pytest -q`。
- Commit 信息格式：`<type>: <说明>`，type 取 `feat` / `fix` / `exp` / `refactor` / `docs` / `test`。
- 涉及算法改动的提交，在信息中附上关键指标变化。

## 参考

- [AGENTS.md 规范](https://agents.md)
- [forrestchang/andrej-karpathy-skills](https://github.com/forrestchang/andrej-karpathy-skills)
- [numerai/example-scripts AGENTS.md](https://github.com/numerai/example-scripts/blob/master/AGENTS.md)（配置驱动实验、先小数据试跑）
