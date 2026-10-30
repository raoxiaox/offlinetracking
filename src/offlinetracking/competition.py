"""模块 5：竞争感知（§8）。只产生标记，不改代价。"""

from __future__ import annotations

from collections import defaultdict

from .structs import Edge


def competition(edges: list[Edge], C0: float, delta_m: float) -> None:
    """把“不连”作为代价 C0 的虚拟候选加入每个列表，计算 MNN、歧义度 A 和竞争者。"""
    out_e, in_e = defaultdict(list), defaultdict(list)
    for k, e in enumerate(edges):
        out_e[e.i].append(k)
        in_e[e.j].append(k)

    def best_two(ks):
        vals = sorted([(edges[k].C, k) for k in ks] + [(C0, -1)])
        return vals[0], vals[1]

    best_out = {i: best_two(ks) for i, ks in out_e.items()}
    best_in = {j: best_two(ks) for j, ks in in_e.items()}
    for k, e in enumerate(edges):
        (b1, k1), (b2, _) = best_out[e.i]
        m_out = (b2 - e.C) if k1 == k else (b1 - e.C)
        (c1, l1), (c2, _) = best_in[e.j]
        m_in = (c2 - e.C) if l1 == k else (c1 - e.C)
        e.best_out, e.best_in = k1 == k, l1 == k
        e.mnn = e.best_out and e.best_in
        e.A = float(min(m_out, m_in))
        comp = []
        for kk in set(out_e[e.i]) | set(in_e[e.j]):
            if kk != k and edges[kk].C < e.C + delta_m:
                comp.append((edges[kk].i, edges[kk].j, float(edges[kk].C)))
        e.competitors = sorted(comp, key=lambda t: t[2])
