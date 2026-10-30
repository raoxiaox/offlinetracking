"""模块 1：候选门控（§4）。只去掉物理上不可能的对，以召回为准。"""

from __future__ import annotations

import bisect

import numpy as np

from .cmc import CMCTrack, apply
from .config import per_class
from .structs import Edge, Tracklet


def class_compatible(a: str, b: str, groups: list[list[str]]) -> bool:
    return a == b or any(a in g and b in g for g in groups)


def mean_scale(bi: np.ndarray, bj: np.ndarray) -> float:
    """s̄_ij = (w_i h_i w_j h_j)^(1/4)。"""
    return float((bi[2] * bi[3] * bj[2] * bj[3]) ** 0.25)


def gate_metrics(tri: Tracklet, trj: Tracklet, cmc: CMCTrack) -> tuple[float, float]:
    """返回 (g_ij, rho)：CMC 补偿后、按目标尺度归一化的端点位移。"""
    p = apply(cmc.Hab(tri.t_e, trj.t_s), tri.boxes[-1, :2])
    g = float(np.linalg.norm(trj.boxes[0, :2] - p)) / max(mean_scale(tri.boxes[-1], trj.boxes[0]), 1e-6)
    return g, cmc.rho(tri.t_e, trj.t_s)


def build_candidates(trs: list[Tracklet], cmc: CMCTrack, fps: float, cfg: dict,
                     loose: bool = False) -> list[Edge]:
    """按 t^s 排序 + 二分取时间窗，再逐一检查约束 2–5。loose=True 时只做时间约束（用于标定）。"""
    gc = cfg["gating"]
    max_gap = int(round(gc["tau_max"] * fps))
    order = sorted(range(len(trs)), key=lambda k: trs[k].t_s)
    starts = [trs[k].t_s for k in order]
    cannot = {tuple(p) for p in gc.get("cannot_link", [])}
    tid = [tr.tid for tr in trs]
    edges: list[Edge] = []
    for i, tri in enumerate(trs):
        lo = bisect.bisect_right(starts, tri.t_e)
        hi = bisect.bisect_right(starts, tri.t_e + max_gap)
        for j in (order[k] for k in range(lo, hi)):
            trj = trs[j]
            gap = trj.t_s - tri.t_e
            if gap < 1:
                continue
            tau = gap / fps
            g, rho = gate_metrics(tri, trj, cmc)
            e = Edge(i=i, j=j, gap=gap, tau=tau, cls=tri.cls, g=g, rho=rho)
            if not loose:
                if not class_compatible(tri.cls, trj.cls, gc["class_groups"]):
                    continue
                if (tid[i], tid[j]) in cannot:
                    continue
                lim = per_class(gc["v_max"], tri.cls) * tau + per_class(gc["g0"], tri.cls)
                if rho < gc["rho_min"]:
                    lim *= gc["cmc_fail_relax"]
                    e.flags.add("cmc_fail")
                if g > lim:
                    continue
            elif rho < gc["rho_min"]:
                e.flags.add("cmc_fail")
            if tri.exit_border:
                e.flags.add("exit_border")
            if trj.enter_border:
                e.flags.add("enter_border")
            edges.append(e)
    return edges
