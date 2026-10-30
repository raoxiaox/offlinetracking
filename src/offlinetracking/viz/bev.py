"""轨迹俯视图（BEV）与“时间–图像 x”视图。

BEV：框底边中点经地面单应投到自车系地面，再用自车位姿转到世界坐标。没有相机标定时只画时间–x 视图。
颜色：每个合并后的全局 ID 一种颜色；缺口连线按决策着色（自动=绿实线，复核=橙虚线），插值段为点线。
"""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..decision import interpolate  # noqa: E402
from ..structs import Edge, Sequence  # noqa: E402

DEC_STYLE = {"auto": dict(color="#1a9850", ls="-", lw=2.0), "review": dict(color="#f28e2b", ls="--", lw=1.8),
             "reject": dict(color="#9e9e9e", ls=":", lw=1.0)}
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                     "font.sans-serif": ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei", "DejaVu Sans"],
                     "axes.unicode_minus": False})


def has_bev(seq: Sequence) -> bool:
    return seq.camera is not None and "Hg" in seq.camera


def to_ground(seq: Sequence, frames: np.ndarray, boxes: np.ndarray, max_range: float = 70.0):
    """框 (n,4 cxcywh) → 世界坐标 (n,2) 与有效标记。无 ego 位姿时为自车系坐标。"""
    Hinv = np.linalg.inv(np.asarray(seq.camera["Hg"], float))
    bc = np.column_stack([boxes[:, 0], boxes[:, 1] + boxes[:, 3] / 2, np.ones(len(boxes))])
    g = bc @ Hinv.T
    valid = g[:, 2] > 1e-6
    xy = g[:, :2] / np.where(valid, g[:, 2], 1)[:, None]
    valid &= (xy[:, 0] > 1.0) & (xy[:, 0] < max_range)
    if seq.ego is None:
        return xy, valid
    ego = seq.ego[np.clip(frames.astype(int), 0, len(seq.ego) - 1)]
    c, s = np.cos(ego[:, 2]), np.sin(ego[:, 2])
    return np.column_stack([ego[:, 0] + c * xy[:, 0] - s * xy[:, 1], ego[:, 1] + s * xy[:, 0] + c * xy[:, 1]]), valid


def _color(gid: int):
    return plt.get_cmap("tab20")(int(gid) % 20)


def _smooth(xy: np.ndarray, k: int = 3) -> np.ndarray:
    if len(xy) < k:
        return xy
    ker = np.ones(k) / k
    pad = np.pad(xy, ((k // 2, k // 2), (0, 0)), mode="edge")
    return np.column_stack([np.convolve(pad[:, d], ker, mode="valid") for d in range(2)])


def _endpoint(seq, tr, end):
    k = -1 if end == "tail" else 0
    xy, ok = to_ground(seq, tr.frames[[k]], tr.boxes[[k]])
    if ok[0]:
        return xy[0]
    xy, ok = to_ground(seq, tr.frames, tr.boxes)
    if not ok.any():
        return None
    idx = np.nonzero(ok)[0]
    return xy[idx[-1] if end == "tail" else idx[0]]


def draw_bev(ax, seq: Sequence, edges: list[Edge], gid: np.ndarray, only: set | None = None,
             decisions=("auto", "review"), highlight: Edge | None = None, cfg: dict | None = None):
    if seq.ego is not None:
        ax.plot(seq.ego[1:, 0], seq.ego[1:, 1], color="k", lw=1.2, alpha=0.6, label="自车")
    for k, tr in enumerate(seq.tracklets):
        if only is not None and k not in only:
            continue
        xy, ok = to_ground(seq, tr.frames, tr.boxes)
        if ok.sum() == 0:
            continue
        col = _color(gid[k])
        lw, alpha = 1.6, 0.9
        if highlight is not None:
            if k == highlight.i:
                col, lw = "#2166ac", 2.6
            elif k == highlight.j:
                col, lw = "#b2182b", 2.6
            else:
                col, alpha = "#7f7f7f", 0.6
        p = _smooth(xy[ok])
        ax.plot(p[:, 0], p[:, 1], color=col, lw=lw, alpha=alpha)
        ax.plot(*p[0], marker="o", ms=3, color=col, alpha=alpha)
        ax.plot(*p[-1], marker="s", ms=3, color=col, alpha=alpha)
        if highlight is None and len(p) > 3:
            ax.annotate(str(tr.tid), p[len(p) // 2], fontsize=6, color=col, alpha=0.9)
    for e in edges:
        if e.decision not in decisions and e is not highlight:
            continue
        if only is not None and (e.i not in only or e.j not in only):
            continue
        a = _endpoint(seq, seq.tracklets[e.i], "tail")
        b = _endpoint(seq, seq.tracklets[e.j], "head")
        if a is None or b is None:
            continue
        st = dict(DEC_STYLE[e.decision])
        if e is highlight:
            st["lw"] = 2.6
        ax.plot([a[0], b[0]], [a[1], b[1]], **st)
        if e.decision == "auto" and cfg is not None:
            ip = interpolate(e, seq, cfg)
            if len(ip):
                ixy, iok = to_ground(seq, ip[:, 0], ip[:, 1:5])
                if iok.any():
                    ax.plot(ixy[iok, 0], ixy[iok, 1], ls=":", lw=1.4, color=_color(gid[e.i]))
    ax.set_xlabel("世界 X（米，沿行驶方向）")
    ax.set_ylabel("世界 Y（米，向左）")


def draw_time_x(ax, seq: Sequence, edges: list[Edge], gid: np.ndarray, only: set | None = None,
                highlight: Edge | None = None):
    for k, tr in enumerate(seq.tracklets):
        if only is not None and k not in only:
            continue
        col = _color(gid[k])
        if highlight is not None:
            col = "#2166ac" if k == highlight.i else "#b2182b" if k == highlight.j else "#9e9e9e"
        ax.plot(tr.frames / seq.fps, tr.boxes[:, 0], color=col, lw=1.4)
    for e in edges:
        if e.decision == "reject" and e is not highlight:
            continue
        if only is not None and (e.i not in only or e.j not in only):
            continue
        a, b = seq.tracklets[e.i], seq.tracklets[e.j]
        ax.plot([a.t_e / seq.fps, b.t_s / seq.fps], [a.boxes[-1, 0], b.boxes[0, 0]], **DEC_STYLE[e.decision])
    ax.set_xlabel("时间（秒）")
    ax.set_ylabel("图像 x（像素）")
    ax.set_ylim(0, seq.img_w)


def overview_figure(seq, edges, gid, cfg=None):
    if has_bev(seq):
        fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), gridspec_kw={"height_ratios": [1.15, 1]})
        draw_bev(axes[0], seq, edges, gid, cfg=cfg)
        axes[0].set_title(f"{seq.name} · 俯视图（颜色=合并后 ID；绿实线=自动连接，橙虚线=待复核，点线=插值）", loc="left")
        draw_time_x(axes[1], seq, edges, gid)
        axes[1].set_title("时间–图像 x 视图", loc="left")
    else:
        fig, ax = plt.subplots(figsize=(13, 4))
        draw_time_x(ax, seq, edges, gid)
        ax.set_title(f"{seq.name} · 无相机标定，只显示时间–图像 x 视图", loc="left")
    fig.tight_layout()
    return fig


def edge_figure(seq, edges, gid, e: Edge, cfg=None):
    """单条边的局部视图：i（蓝）、j（红）、竞争者（灰）。"""
    involved = {e.i, e.j} | {a for a, b, _ in e.competitors} | {b for a, b, _ in e.competitors}
    sub = [x for x in edges if x is e or ((x.i, x.j) in {(a, b) for a, b, _ in e.competitors})]
    ncol = 2 if has_bev(seq) else 1
    fig, axes = plt.subplots(1, ncol, figsize=(4.6 * ncol, 3.2))
    axes = np.atleast_1d(axes)
    if has_bev(seq):
        draw_bev(axes[0], seq, sub, gid, only=involved, decisions=("auto", "review", "reject"), highlight=e)
        pts = [p for p in (_endpoint(seq, seq.tracklets[e.i], "tail"), _endpoint(seq, seq.tracklets[e.j], "head"))
               if p is not None]
        if pts:
            c = np.mean(pts, 0)
            axes[0].set_xlim(c[0] - 30, c[0] + 30)
            axes[0].set_ylim(c[1] - 12, c[1] + 12)
        axes[0].set_title("局部俯视图", loc="left")
    draw_time_x(axes[-1], seq, sub, gid, only=involved, highlight=e)
    t0 = seq.tracklets[e.i].t_e / seq.fps
    axes[-1].set_xlim(t0 - 3, t0 + e.tau + 3)
    axes[-1].set_title("时间–图像 x", loc="left")
    fig.tight_layout()
    return fig


def reliability_figure(curve: list, ece_val: float):
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.plot([0, 1], [0, 1], color="#9e9e9e", ls="--", lw=1)
    if curve:
        c = np.array(curve)
        ax.plot(c[:, 0], c[:, 1], marker="o", color="#2166ac")
    ax.set_xlabel("预测链接概率 P_ij")
    ax.set_ylabel("实际正样本比例")
    ax.set_title(f"可靠性曲线 · ECE={ece_val:.3f}", loc="left")
    fig.tight_layout()
    return fig


def fig_to_png(fig, dpi: int = 110) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    return buf.getvalue()
