"""核心数据结构。框统一为 cxcywh（中心 + 宽高，像素），帧号从 1 开始。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(eq=False)
class Tracklet:
    tid: int
    cls: str
    frames: np.ndarray            # (n,) int，严格递增
    boxes: np.ndarray             # (n, 4) cxcywh
    scores: np.ndarray            # (n,)
    feats: np.ndarray | None = None   # (n, D) L2 归一化
    q: np.ndarray | None = None       # (n,) 观测质量，模块 0 填
    q_trunc: np.ndarray | None = None
    q_occ: np.ndarray | None = None
    exit_border: bool = False
    enter_border: bool = False
    gt_id: int = -1               # 标签构造时填，-1 表示无归属
    purity: float = 1.0
    impure: bool = False          # 外观自检得到的“不纯”标记（模块 3）

    @property
    def t_s(self) -> int:
        return int(self.frames[0])

    @property
    def t_e(self) -> int:
        return int(self.frames[-1])

    def __len__(self) -> int:
        return len(self.frames)


@dataclass(eq=False)
class Edge:
    """一条有向候选边 i → j 及其全部证据。i、j 是 tracklets 列表的下标。"""

    i: int
    j: int
    gap: int                      # Δ_ij（帧）
    tau: float                    # τ_ij（秒）
    cls: str = ""                 # 轨迹 i 的类别
    g: float = 0.0
    rho: float = 1.0
    flags: set = field(default_factory=set)
    # 证据
    e_f: float = np.nan
    e_b: float = np.nan
    C_m: float = 1.0
    Q_m: float = 0.0
    d2: float = np.nan            # NLL 模式的平均马氏距离平方
    trace: float = np.nan
    d_set: float = np.nan
    d_proto: float = np.nan
    C_a: float = 0.5
    Q_a: float = 0.0
    reps_i: list = field(default_factory=list)
    reps_j: list = field(default_factory=list)
    C_s: float = 0.0
    Q_s: float = 0.0
    C_t: float = 0.0
    # 融合与决策
    C: float = np.inf
    P: float = np.nan
    E: float = 0.0
    mnn: bool = False
    best_out: bool = False        # j 是 i 的最优后继（且优于拒绝代价）
    best_in: bool = False         # i 是 j 的最优前驱
    A: float = -np.inf
    competitors: list = field(default_factory=list)
    x: int = 0
    decision: str = "reject"
    label: int = -1               # 1 正 / 0 负 / -1 未知
    # 内部：插值用的逐帧预测（不导出）
    pred: dict | None = field(default=None, repr=False)


@dataclass
class Sequence:
    name: str
    fps: float
    img_w: int
    img_h: int
    tracklets: list[Tracklet]
    gt: np.ndarray | None = None          # (m, 7) frame, gt_id, cx, cy, w, h, cls_idx
    cmc_H: np.ndarray | None = None       # (T+1, 3, 3)；cmc_H[t] 把帧 t 映到帧 t+1
    cmc_rho: np.ndarray | None = None     # (T+1,)
    cmc_sigma: np.ndarray | None = None   # (T+1,) 内点残差（像素）
    frames_dir: str | None = None
    camera: dict | None = None            # {"Hg": 3x3 地面(自车系)→像素, ...}
    ego: np.ndarray | None = None         # (T+1, 3) x, y, yaw；行号=帧号
    n_frames: int = 0
    meta: dict = field(default_factory=dict)
