# offlinetracking

离线多目标跟踪（Offline Multi-Object Tracking）算法仓库。

“离线”指输入完整视频或检测序列后，可以利用全局信息（前后帧）做关联、轨迹补全与平滑，追求精度而非实时性。

## 目录结构

```
configs/              # 实验配置（YAML），一次实验 = 一个配置文件
src/offlinetracking/  # 核心代码：检测读取、关联、轨迹后处理、评估
scripts/              # 训练 / 推理 / 评估入口脚本
tests/                # 单元测试（pytest）
data/                 # 数据集（不入库）
outputs/              # 实验输出（不入库）
```

## 快速开始

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # 待添加 pyproject.toml
pytest -q
python scripts/track.py --config configs/<exp>.yaml
python scripts/eval.py  --config configs/<exp>.yaml   # HOTA / MOTA / IDF1
```

## 约定

- 评估指标：HOTA（主）、IDF1、MOTA，基于 TrackEval。
- 结果格式：MOTChallenge `frame,id,x,y,w,h,score,-1,-1,-1`。
- AI 编码助手请先阅读 [AGENTS.md](AGENTS.md)。
