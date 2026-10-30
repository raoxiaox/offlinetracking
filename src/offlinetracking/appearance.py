"""模块 3：轨迹级外观（§6）。"""

from __future__ import annotations

import numpy as np

from .structs import Edge, Sequence, Tracklet


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def select_reps(tr: Tracklet, end: str, window: int, q_min: float, k: int, min_sep: int) -> np.ndarray:
    """端点附近 window 帧内，按质量选 top-k 代表帧，相邻代表帧至少间隔 min_sep 帧。返回下标。"""
    if end == "tail":
        idx = np.nonzero(tr.frames > tr.t_e - window)[0]
    else:
        idx = np.nonzero(tr.frames < tr.t_s + window)[0]
    span = int(tr.frames[idx[-1]] - tr.frames[idx[0]]) + 1 if len(idx) else 0
    min_sep = min(min_sep, max(1, span // k))   # 短窗口放宽间隔，否则短轨迹永远凑不够代表帧
    idx = idx[tr.q[idx] >= q_min]
    chosen: list[int] = []
    for k_ in idx[np.argsort(-tr.q[idx], kind="stable")]:
        if all(abs(int(tr.frames[k_]) - int(tr.frames[c])) >= min_sep for c in chosen):
            chosen.append(int(k_))
            if len(chosen) == k:
                break
    return np.array(sorted(chosen), int)


def proto(feats: np.ndarray, w: np.ndarray) -> np.ndarray:
    v = (w[:, None] * feats).sum(0)
    return v / max(np.linalg.norm(v), 1e-9)


def set_distances(fa: np.ndarray, fb: np.ndarray, wa: np.ndarray, wb: np.ndarray) -> tuple[float, float]:
    """d_set = 两组特征两两余弦距离的中位数；d_proto = 质量加权原型的余弦距离。"""
    d_set = float(np.median(1.0 - fa @ fb.T))
    d_proto = float(1.0 - proto(fa, wa) @ proto(fb, wb))
    return d_set, d_proto


def tracklet_self_distance(tr: Tracklet, min_len: int = 10) -> float:
    """轨迹前后两半的原型距离（§6.2 轨迹内一致性）。过短返回 nan。"""
    if tr.feats is None or len(tr) < min_len:
        return np.nan
    h = len(tr) // 2
    w = np.maximum(tr.q, 1e-3)
    return float(1.0 - proto(tr.feats[:h], w[:h]) @ proto(tr.feats[h:], w[h:]))


class AppearanceModel:
    def __init__(self, seq: Sequence, cfg: dict):
        ac = cfg["appearance"]
        self.ac = ac
        self.seq = seq
        self.window = int(round(ac["window_sec"] * seq.fps))
        self.reps: dict = {}
        for tr in seq.tracklets:
            d = tracklet_self_distance(tr)
            tr.impure = bool(np.isfinite(d) and d > ac["purity_max"])

    def _reps(self, i: int, end: str) -> np.ndarray:
        key = (i, end)
        if key not in self.reps:
            tr = self.seq.tracklets[i]
            ac = self.ac
            if ac.get("mode", "set") == "lastfirst":
                self.reps[key] = np.array([len(tr) - 1 if end == "tail" else 0])
            else:
                self.reps[key] = select_reps(tr, end, self.window, ac["q_min"], ac["k"], ac["min_sep"])
        return self.reps[key]

    def evidence(self, e: Edge) -> None:
        ac = self.ac
        tri, trj = self.seq.tracklets[e.i], self.seq.tracklets[e.j]
        if tri.impure or trj.impure:
            e.flags.add("impure")
        if not ac.get("enabled", True) or tri.feats is None or trj.feats is None:
            return
        ri, rj = self._reps(e.i, "tail"), self._reps(e.j, "head")
        e.reps_i = [int(tri.frames[k]) for k in ri]
        e.reps_j = [int(trj.frames[k]) for k in rj]
        lastfirst = ac.get("mode", "set") == "lastfirst"
        kmin = 1 if lastfirst else ac["k_min"]
        if len(ri) < kmin or len(rj) < kmin:
            e.Q_a, e.C_a = 0.0, 0.5
            return
        wi, wj = np.maximum(tri.q[ri], 1e-3), np.maximum(trj.q[rj], 1e-3)
        e.d_set, e.d_proto = set_distances(tri.feats[ri], trj.feats[rj], wi, wj)
        if not lastfirst and abs(e.d_set - e.d_proto) > ac["delta_a"]:
            e.flags.add("app_inconsistent")
        e.C_a = float(1.0 - sigmoid(ac["beta0"] + ac["beta1"] * e.d_set))
        q_rep = float(np.mean(np.concatenate([tri.q[ri], trj.q[rj]])))
        k_valid = min(len(ri), len(rj))
        e.Q_a = q_rep * (1.0 if lastfirst else min(1.0, k_valid / ac["k"]))
        if e.Q_a >= ac.get("veto_q", 1.0) and e.C_a >= ac.get("veto_ca", 1.0):
            e.flags.add("app_veto")
