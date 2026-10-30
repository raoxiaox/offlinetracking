"""模块 2：双端运动预测（§5），以及阶段 4 的 NLL 代价（§5.6）与地平面运动。

速度单位为“像素/帧”（ground 模式为“米/帧”），所有误差按目标尺度归一化。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .cmc import CMCTrack, apply
from .gating import mean_scale
from .structs import Edge, Sequence, Tracklet


@dataclass
class EndFit:
    a: np.ndarray          # 端点位置（拟合值）
    v: np.ndarray          # 速度（每帧），沿时间正方向
    alpha: float           # 端点 log h（拟合值）
    beta: float            # log h 斜率（每帧）
    n: int                 # 用于拟合的有效观测数
    var_v: float           # 速度估计方差（每轴，点坐标单位²/帧²）
    low: bool              # 观测不足，v 置 0
    t_ref: int


class Identity:
    """ground 模式下的“变换”：世界坐标系不需要 CMC。"""

    def Hab(self, a, b):
        return np.eye(3)

    def step(self, t):
        return np.eye(3)

    def rho(self, a, b):
        return 1.0

    def sigma(self, a, b):
        return 0.0


def fit_end(frames: np.ndarray, pts: np.ndarray, h: np.ndarray, q: np.ndarray, end: str, K: int,
            mode: str, tf) -> EndFit:
    """取末端（end='tail'）或起始端（'head'）K 个观测，对齐到端点帧后做质量加权最小二乘。"""
    sl = slice(-K, None) if end == "tail" else slice(0, K)
    fr, p, hh = frames[sl], pts[sl], h[sl]
    w = np.maximum(q[sl], 1e-3)
    t_ref = int(frames[-1] if end == "tail" else frames[0])
    pt = np.array([apply(tf.Hab(int(f), t_ref), pk) for f, pk in zip(fr, p)])
    ref_k = -1 if end == "tail" else 0
    lh = np.log(np.maximum(hh, 1e-3))
    n = len(fr)
    if n < 3 or (mode == "twoframe" and n < 2):
        return EndFit(pt[ref_k], np.zeros(2), float(lh[ref_k]), 0.0, n, 0.0, True, t_ref)
    if mode == "twoframe":
        k1, k0 = (-1, -2) if end == "tail" else (1, 0)
        v = (pt[k1] - pt[k0]) / max(fr[k1] - fr[k0], 1)
        dh = (lh[k1] - lh[k0]) / max(fr[k1] - fr[k0], 1)
        return EndFit(pt[ref_k], v, float(lh[ref_k]), float(dh), n, 0.0, False, t_ref)
    t = fr.astype(float)
    tb = np.sum(w * t) / w.sum()
    pb = (w[:, None] * pt).sum(0) / w.sum()
    sxx = np.sum(w * (t - tb) ** 2)
    if sxx < 1e-9:
        return EndFit(pt[ref_k], np.zeros(2), float(lh[ref_k]), 0.0, n, 0.0, True, t_ref)
    v = (w[:, None] * (t - tb)[:, None] * (pt - pb)).sum(0) / sxx
    a = pb + v * (t_ref - tb)
    hb = np.sum(w * lh) / w.sum()
    beta = float(np.sum(w * (t - tb) * (lh - hb)) / sxx)
    alpha = float(hb + beta * (t_ref - tb))
    res = pt - (a + np.outer(t - t_ref, v))
    s2 = float(np.sum(w * np.sum(res**2, 1)) / (2 * w.sum())) * n / max(n - 2, 1)
    var_v = s2 / max(np.sum((t - t.mean()) ** 2), 1e-9)
    return EndFit(a, v, alpha, beta, n, var_v, False, t_ref)


def propagate(a: np.ndarray, v: np.ndarray, t0: int, steps: int, tf, forward: bool) -> np.ndarray:
    """逐帧传播：物体在当前帧坐标下走一步，再随背景变换到下一帧（§5.3）。
    返回 (steps, 2)：前向为帧 t0+1..t0+steps，反向为帧 t0−1..t0−steps。"""
    out = np.zeros((steps, 2))
    p, vv = np.asarray(a, float).copy(), np.asarray(v, float).copy()
    for k in range(steps):
        if forward:
            H = tf.step(t0 + k)
            p = apply(H, p + vv)
            vv = H[:2, :2] @ vv
        else:
            Hinv = np.linalg.inv(tf.step(t0 - k - 1))
            vv = Hinv[:2, :2] @ vv
            p = apply(Hinv, p) - vv
        out[k] = p
    return out


def psi(x: float, kappa: float) -> float:
    return min(x, kappa) / kappa


# ---------- 端点拟合的缓存 ----------

def ground_points(seq: Sequence, tr: Tracklet) -> tuple[np.ndarray, np.ndarray]:
    """框底边中点 → 自车系地面 → 世界坐标。返回 (pts, valid)。"""
    Hinv = np.linalg.inv(seq.camera["Hg"])
    bc = np.column_stack([tr.boxes[:, 0], tr.boxes[:, 1] + tr.boxes[:, 3] / 2, np.ones(len(tr))])
    g = bc @ Hinv.T
    valid = g[:, 2] > 1e-6
    xy = g[:, :2] / np.where(valid, g[:, 2], 1)[:, None]
    valid &= (xy[:, 0] > 1.0) & (xy[:, 0] < 150)
    if seq.ego is None:
        return xy, valid
    ego = seq.ego[np.clip(tr.frames, 0, len(seq.ego) - 1)]
    c, s = np.cos(ego[:, 2]), np.sin(ego[:, 2])
    wx = ego[:, 0] + c * xy[:, 0] - s * xy[:, 1]
    wy = ego[:, 1] + s * xy[:, 0] + c * xy[:, 1]
    return np.column_stack([wx, wy]), valid


def end_fits(seq: Sequence, cmc: CMCTrack, cfg: dict) -> dict:
    """为每条轨迹的两端计算拟合。返回 {(i,'tail'|'head', space): EndFit}。"""
    mc = cfg["motion"]
    K, mode = mc["K"], mc["velocity"]
    out = {}
    use_ground = mc.get("space") == "ground" and seq.camera is not None
    ident = Identity()
    for i, tr in enumerate(seq.tracklets):
        for end in ("tail", "head"):
            out[(i, end, "image")] = fit_end(tr.frames, tr.boxes[:, :2], tr.boxes[:, 3], tr.q, end, K, mode, cmc)
            if use_ground:
                gp, valid = ground_points(seq, tr)
                q = np.where(valid, tr.q, 0.0)
                out[(i, end, "ground")] = fit_end(tr.frames, gp, tr.boxes[:, 3], q, end, K, mode, ident)
    return out


def motion_evidence(e: Edge, seq: Sequence, fits: dict, cmc: CMCTrack, cfg: dict) -> None:
    """填 e_f, e_b, C_m, Q_m, d2, trace、运动类标记和插值用的逐帧预测。"""
    mc = cfg["motion"]
    tri, trj = seq.tracklets[e.i], seq.tracklets[e.j]
    K = mc["K"]
    # 图像空间预测：插值总要用
    fi, fj = fits[(e.i, "tail", "image")], fits[(e.j, "head", "image")]
    fwd = propagate(fi.a, fi.v, tri.t_e, e.gap, cmc, True)
    bwd = propagate(fj.a, fj.v, trj.t_s, e.gap, cmc, False)
    e.pred = {"fwd": fwd, "bwd": bwd, "fi": fi, "fj": fj}

    space = "ground" if (mc.get("space") == "ground" and (e.i, "tail", "ground") in fits) else "image"
    if space == "ground":
        gi, gj = fits[(e.i, "tail", "ground")], fits[(e.j, "head", "ground")]
        tf = Identity()
        pf = propagate(gi.a, gi.v, tri.t_e, e.gap, tf, True)[-1]
        pb = propagate(gj.a, gj.v, trj.t_s, e.gap, tf, False)[-1]
        ef, eb = gj.a - pf, gi.a - pb
        s = mc.get("ground_scale_m", 4.0)
        var_vi, var_vj, rho, sig_c = gi.var_v, gj.var_v, 1.0, 0.0
        low = gi.low or gj.low
    else:
        ef, eb = fj.a - fwd[-1], fi.a - bwd[-1]
        s = mean_scale(tri.boxes[-1], trj.boxes[0])
        var_vi, var_vj, rho, sig_c = fi.var_v, fj.var_v, e.rho, cmc.sigma(tri.t_e, trj.t_s)
        low = fi.low or fj.low
        if not cfg["cmc"].get("enabled", True):
            rho = 1.0
    s = max(s, 1e-6)
    e.e_f = float(np.linalg.norm(ef) / s)
    e.e_b = float(np.linalg.norm(eb) / s)
    two = mc.get("two_sided", True)
    km = mc["kappa_m"]
    if mc.get("cost") == "nll":
        def nll(err, var_v):
            sig2 = 2 * mc["sigma_obs"] ** 2 + (e.gap**2) * var_v / s**2 + e.gap * (sig_c / s) ** 2
            d2 = float(np.sum((err / s) ** 2) / sig2)
            return 0.5 * d2 + math.log(sig2), d2, 2 * sig2   # ½d² + ½·log det(σ²I₂)
        cf, d2f, trf = nll(ef, var_vi)
        cb, d2b, trb = nll(eb, var_vj)
        e.C_m = 0.5 * (cf + cb) if two else cf
        e.d2 = 0.5 * (d2f + d2b) if two else d2f
        e.trace = max(trf, trb) if two else trf
        if e.trace > mc["trace_max"]:
            e.flags.add("motion_uncertain")
    else:
        e.C_m = 0.5 * (psi(e.e_f, km) + psi(e.e_b, km)) if two else psi(e.e_f, km)
    if two and abs(e.e_f - e.e_b) > mc["consistency_max"]:
        e.flags.add("motion_inconsistent")
    if low:
        e.flags.add("low_motion")
    n_i = (fits[(e.i, "tail", space)]).n
    n_j = (fits[(e.j, "head", space)]).n
    e.Q_m = rho * min(1, n_i / K) * min(1, n_j / K) * math.exp(-e.tau / mc["tau0"])
