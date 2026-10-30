"""模块 7：三档决策、合并与缺口插值（§10）。"""

from __future__ import annotations

import math

import numpy as np

from .structs import Edge, Sequence


def decide(edges: list[Edge], cfg: dict) -> None:
    dc = cfg["decision"]
    use_comp = cfg["competition"].get("enabled", True)
    block = set(dc["blocking_flags"])
    for e in edges:
        auto = (
            e.x == 1
            and e.C <= dc["theta_auto"]
            and (not use_comp or e.A >= dc["delta_auto"])
            and e.E >= dc["e_min"]
            and not (e.flags & block)
        )
        if auto:
            e.decision = "auto"
        elif e.x == 1 or (e.C <= dc["theta_review"] and (e.best_out or e.best_in)):
            e.decision = "review"
        else:
            e.decision = "reject"


def merge_chains(n: int, links: list[tuple[int, int]]) -> np.ndarray:
    """由后继指针串链，每条链一个全局 ID（从 1 开始）。"""
    succ, has_pred = {}, set()
    for i, j in links:
        succ[i] = j
        has_pred.add(j)
    gid = np.zeros(n, int)
    nxt = 1
    for s in range(n):
        if s in has_pred:
            continue
        k = s
        while True:
            gid[k] = nxt
            if k not in succ:
                break
            k = succ[k]
        nxt += 1
    return gid


def interpolate(e: Edge, seq: Sequence, cfg: dict) -> np.ndarray:
    """缺口内逐帧框：前向/反向预测按距离加权融合（§10.3）。返回 (Δ−1, 6) frame,cx,cy,w,h,score。"""
    tri, trj = seq.tracklets[e.i], seq.tracklets[e.j]
    if e.pred is None or e.gap < 2:
        return np.zeros((0, 6))
    fwd, bwd, fi, fj = e.pred["fwd"], e.pred["bwd"], e.pred["fi"], e.pred["fj"]
    ri = tri.boxes[-1, 2] / tri.boxes[-1, 3]
    rj = trj.boxes[0, 2] / trj.boxes[0, 3]
    s0 = 0.5 * (tri.scores[-1] + trj.scores[0]) * math.exp(-e.tau / max(cfg["decision"]["g_fill_sec"], 1e-6))
    rows = []
    for t in range(tri.t_e + 1, trj.t_s):
        a, b = trj.t_s - t, t - tri.t_e
        p = (a * fwd[t - tri.t_e - 1] + b * bwd[trj.t_s - t - 1]) / e.gap
        lh = (a * (fi.alpha + fi.beta * (t - tri.t_e)) + b * (fj.alpha - fj.beta * (trj.t_s - t))) / e.gap
        h = math.exp(lh)
        r = (a * ri + b * rj) / e.gap
        rows.append([t, p[0], p[1], r * h, h, s0])
    return np.array(rows)


def build_output(seq: Sequence, edges: list[Edge], gid: np.ndarray, cfg: dict,
                 accept: tuple[str, ...] = ("auto",)) -> np.ndarray:
    """合并后的逐帧结果 (n, 8)：frame, id, cx, cy, w, h, score, is_interp。"""
    rows = []
    for tr, g in zip(seq.tracklets, gid):
        rows.append(np.column_stack([tr.frames, np.full(len(tr), g), tr.boxes, tr.scores, np.zeros(len(tr))]))
    dc = cfg["decision"]
    if dc.get("interp", True):
        g_fill = dc["g_fill_sec"] * seq.fps
        for e in edges:
            if e.decision in accept and e.gap <= g_fill:
                ip = interpolate(e, seq, cfg)
                if len(ip):
                    rows.append(np.column_stack([ip[:, 0], np.full(len(ip), gid[e.i]), ip[:, 1:6], np.ones(len(ip))]))
    out = np.concatenate(rows) if rows else np.zeros((0, 8))
    return out[np.lexsort((out[:, 1], out[:, 0]))]
