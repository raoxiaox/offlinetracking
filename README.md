# offlinetracking

2D 离线轨迹重连（STWO，Spatio-Temporal Without Overlap）基线：把上游跟踪器输出的**生命周期不重叠的短轨迹**重新连成完整轨迹。
本仓库不训练跟踪器，只复用已有的检测、短轨迹和 ReID 特征。每条候选连接都会输出分数和证据，并给出三档决策：**自动连接 / 待复核 / 不连接**。最终产出一份给人工审查的 HTML 报告（含轨迹俯视图）。

设计依据：《2D 离线轨迹重连（STWO）工程基线方案》，代码中的章节号（§）与它对应。

## 流水线

| 模块 | 文件 | 作用 |
|---|---|---|
| 0 输入构建 | `cmc.py`、`quality.py` | 帧间相机运动补偿（GFTT + LK + RANSAC 仿射）及可靠度 ρ；观测质量 q；贴边标记 |
| 1 候选门控 | `gating.py` | 时间、类别、CMC 补偿后的归一化位移；只去掉物理上不可能的对 |
| 2 双端运动 | `motion.py` | 多帧加权拟合端点与速度，前向 + 反向逐帧传播；阶段 4 增加 NLL 代价、地平面运动 |
| 3 轨迹级外观 | `appearance.py` | 两端按质量选代表帧，用集合距离比较；做轨迹纯度自检，并支持外观否决 |
| 4 融合 | `fusion.py` | 尺度趋势、缺口惩罚、质量自适应加权与证据充分度 E；阶段 4 增加逻辑回归融合 |
| 5 竞争感知 | `competition.py` | 把“不连”作为虚拟候选，计算互选（MNN）与歧义度 A（只作标记） |
| 6 全局关联 | `association.py` | 带哑节点的 2N×2N 指派矩阵 + 匈牙利（按连通分量求解）；贪心 MNN 基线 |
| 7 决策输出 | `decision.py` | 三档决策、链合并、缺口插值、JSONL 边记录 |
| 评测 / 标定 | `eval/`、`calibrate.py` | 轨迹归属与边标签、分桶指标、IDF1、ECE；所有阈值都从标定集得到 |
| 可视化 | `viz/` | 俯视图、时间–x 图、HTML 审查报告 |

## 快速开始

```bash
pip install -e ".[dev]"
pytest -q

python scripts/make_synth.py --out data/synth --n-seq 24          # 合成数据（含帧图像，约 210 MB）
python scripts/calibrate.py --data data/synth --out configs/calibrated.yaml
python scripts/run_stwo.py --config configs/calibrated.yaml --split test   # 输出边记录、指标与 report.html
python scripts/ablate.py --tau-sweep                              # 消融 A0–A12 + τ_max 扫描
```

每次运行的输出在 `outputs/<实验名>/<时间戳>/`，包括生效配置、git hash、日志、`metrics.json`、`report.html`，以及每段序列的 `edges.jsonl` 和 `merged.txt`。

## 输入格式（每段序列一个目录）

| 文件 | 必需 | 内容 |
|---|---|---|
| `seqinfo.yaml` | 是 | `fps`、`img_w`、`img_h`、`n_frames` |
| `tracklets.txt` | 是 | `frame,id,x,y,w,h,score,cls`：上游短轨迹，x、y 为框左上角，帧号从 1 开始 |
| `feats.npy` | 推荐 | `(行数, D)` ReID 特征，行顺序与 `tracklets.txt` 一致 |
| `frames/%06d.jpg` | 推荐 | 视频帧。用于估计 CMC，以及审查报告里的裁剪图 |
| `cmc.npz` | 可选 | 预计算的相机运动：`H (T+2,3,3)`，其中 H[t] 把帧 t 映到帧 t+1；另有 `rho`、`sigma` |
| `camera.yaml` | 可选 | `Hg`：自车系地面 (x 前, y 左, 1) → 像素的单应。用于俯视图和地平面运动 |
| `ego.txt` | 可选 | `frame,x,y,yaw` 自车位姿。有它时俯视图使用世界坐标 |
| `gt.txt` | 评测用 | `frame,gt_id,x,y,w,h,visibility,cls` |

`splits.yaml` 按**视频**划分为 `calib` 和 `test`。阈值只在 `calib` 上标定，`test` 只用来报告结果。

## 输出

- `edges.jsonl`：每条候选边一条记录，包含 src、dst、缺口、代价/概率、运动、外观、尺度、证据充分度、竞争者、标记、求解结果和决策。复核界面、消融分析和回滚都以它为数据源。
- `merged.txt`：只按**自动档**合并后的结果，格式为 `frame,id,x,y,w,h,score,is_interp`。插值框 `is_interp=1`。
- `report.html`：自包含的审查报告，内容如下：
  - 总体指标；
  - 每段序列的俯视图和时间–x 图；
  - 复核队列，按歧义度从高到低排列。每条有局部俯视图、两端代表帧裁剪、证据表和竞争候选。

## 合成数据

`synth.py` 模拟车载前视相机：

- 自车前进并伴有横摆；相机俯仰和偏航有抖动；
- 场景里有同向、对向和路边停放的车辆，并计算遮挡与截断；
- 检测带噪声，会漏检，也会成段失效 0.2–3 s；
- 上游轨迹会在缺口处断开；
- ReID 特征里有“同型车”，被遮挡的观测会混入遮挡者的特征。

帧图像渲染了带纹理的地面和远景建筑，CMC 在这些图像上真实地跑一遍。真值包含被完全遮挡的目标，因此补框（插值）能提高 IDF1。

**合成数据上的数字只用来验证实现是否正确、消融趋势是否合理，不代表真实效果。** 换成真实数据后需要重新标定。换相机、帧率或检测器时也一样。

## 约定

- 框内部统一为 `cx, cy, w, h`（像素），文件中为 `x, y, w, h`（左上角）。
- AI 编码助手请先读 [AGENTS.md](AGENTS.md)。
