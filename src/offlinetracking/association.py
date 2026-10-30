"""模块 6：带拒绝的全局关联（§9）。"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .structs import Edge, Tracklet

BIG = 1e6


def end_start_costs(trs: list[Tracklet], cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    ac = cfg["association"]
    half = ac["theta_link"] / 2
    eta = ac.get("eta_border", 0.0)
    c_end = np.array([half - eta * tr.exit_border for tr in trs], float)
    c_start = np.array([half - eta * tr.enter_border for tr in trs], float)
    return c_end, c_start


def solve_hungarian(n: int, edges: list[Edge], c_end: np.ndarray, c_start: np.ndarray) -> None:
    """按弱连通分量构造 2n×2n 增广指派矩阵并求解，结果写入 e.x。"""
    for e in edges:
        e.x = 0
    if not edges:
        return
    ii = np.array([e.i for e in edges])
    jj = np.array([e.j for e in edges])
    g = coo_matrix((np.ones(len(edges)), (ii, jj)), shape=(n, n))
    _, lab = connected_components(g, directed=True, connection="weak")
    by_comp: dict[int, list[int]] = {}
    for k, e in enumerate(edges):
        by_comp.setdefault(int(lab[e.i]), []).append(k)
    for ks in by_comp.values():
        nodes = sorted({edges[k].i for k in ks} | {edges[k].j for k in ks})
        pos = {v: a for a, v in enumerate(nodes)}
        m = len(nodes)
        M = np.full((2 * m, 2 * m), BIG)
        M[m:, m:] = 0.0
        for a, v in enumerate(nodes):
            M[a, m + a] = c_end[v]
            M[m + a, a] = c_start[v]
        for k in ks:
            e = edges[k]
            if np.isfinite(e.C):
                M[pos[e.i], pos[e.j]] = min(M[pos[e.i], pos[e.j]], e.C)
        r, c = linear_sum_assignment(M)
        chosen = {(nodes[a], nodes[b]) for a, b in zip(r, c) if a < m and b < m and M[a, b] < BIG}
        for k in ks:
            if (edges[k].i, edges[k].j) in chosen:
                edges[k].x = 1


def solve_greedy(edges: list[Edge], c_end: np.ndarray, c_start: np.ndarray,
                 require_mnn: bool, delta_m: float) -> None:
    """可解释基线：按代价从小到大，满足阈值（与 MNN、margin）且两端空闲即接受。"""
    used_out, used_in = set(), set()
    for e in edges:
        e.x = 0
    for e in sorted(edges, key=lambda e: e.C):
        if not e.C < c_end[e.i] + c_start[e.j]:
            continue
        if require_mnn and not (e.mnn and e.A >= delta_m):
            continue
        if e.i in used_out or e.j in used_in:
            continue
        e.x = 1
        used_out.add(e.i)
        used_in.add(e.j)


def associate(trs: list[Tracklet], edges: list[Edge], cfg: dict) -> None:
    c_end, c_start = end_start_costs(trs, cfg)
    if cfg["association"]["solver"] == "greedy":
        solve_greedy(edges, c_end, c_start, cfg["competition"].get("enabled", True), cfg["competition"]["delta_m"])
    else:
        solve_hungarian(len(trs), edges, c_end, c_start)
