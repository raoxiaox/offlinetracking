"""模块 4：尺度一致性、缺口代价与质量自适应融合（§7）；阶段 4 的逻辑回归融合（§7.5）。"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .appearance import sigmoid
from .structs import Edge, Sequence

LR_FEATURES = ["C_m", "C_a", "C_s", "QmCm", "QaCa", "QsCs", "Q_m", "Q_a", "C_t"]


def scale_evidence(e: Edge, seq: Sequence, cfg: dict) -> None:
    """C_s：尺度变化是否符合两端尺度趋势；Q_s：任一端截断或遮挡严重时为 0。"""
    sc = cfg["scale"]
    tri, trj = seq.tracklets[e.i], seq.tracklets[e.j]
    if not sc.get("enabled", True) or e.pred is None:
        e.C_s, e.Q_s = 0.0, 0.0
        return
    fi, fj = e.pred["fi"], e.pred["fj"]
    expect = 0.5 * (fi.beta + fj.beta) * e.gap
    resid = abs((fj.alpha - fi.alpha) - expect)
    e.C_s = min(resid, sc["kappa_s"]) / sc["kappa_s"]
    bad = tri.q_trunc[-1] < 1 or trj.q_trunc[0] < 1 or tri.q_occ[-1] < 0.5 or trj.q_occ[0] < 0.5
    e.Q_s = 0.0 if bad else 1.0


def gap_cost(e: Edge, cfg: dict) -> None:
    e.C_t = e.tau / cfg["gating"]["tau_max"]


def enabled_kinds(cfg: dict) -> list[str]:
    kinds = ["m"]
    if cfg["appearance"].get("enabled", True):
        kinds.append("a")
    if cfg["scale"].get("enabled", True):
        kinds.append("s")
    return kinds


def lr_vector(e: Edge, cfg: dict) -> np.ndarray:
    kinds = enabled_kinds(cfg)
    Cm, Qm = e.C_m, e.Q_m
    Ca, Qa = (e.C_a, e.Q_a) if "a" in kinds else (0.0, 0.0)
    Cs, Qs = (e.C_s, e.Q_s) if "s" in kinds else (0.0, 0.0)
    return np.array([Cm, Ca, Cs, Qm * Cm, Qa * Ca, Qs * Cs, Qm, Qa, e.C_t])


def fuse(edges: list[Edge], cfg: dict) -> None:
    fc = cfg["fusion"]
    kinds = enabled_kinds(cfg)
    w = {"m": fc["w_m"], "a": fc["w_a"], "s": fc["w_s"]}
    wt = fc["w_t"]
    wsum = sum(w[k] for k in kinds)
    mode = fc.get("mode", "adaptive")
    lr = fc.get("logreg") if mode == "logreg" else None
    for e in edges:
        if "app_veto" in e.flags:      # 外观强力否决：不参与连接
            e.C, e.E = 30.0, 0.0
            e.P = 0.0 if lr is not None else np.nan
            continue
        C = {"m": e.C_m, "a": e.C_a, "s": e.C_s}
        Q = {"m": e.Q_m, "a": e.Q_a, "s": e.Q_s}
        e.E = float(sum(w[k] * Q[k] for k in kinds) / wsum) if wsum > 0 else 0.0
        if lr is not None:
            z = float(np.dot(lr["coef"], lr_vector(e, cfg)) + lr["intercept"])
            e.P = float(sigmoid(z))
            e.C = float(np.clip(-z, -30, 30))
            continue
        if mode == "fixed":
            ev = sum(w[k] * C[k] for k in kinds) / wsum
        else:
            den = sum(w[k] * Q[k] for k in kinds)
            ev = sum(w[k] * Q[k] * C[k] for k in kinds) / den if den > 1e-9 else 1.0
        e.C = float((1 - wt) * ev + wt * e.C_t)


def fit_logreg(X: np.ndarray, y: np.ndarray, l2: float = 1e-2) -> dict:
    """带 L2 正则的逻辑回归（标准化特征后用 L-BFGS 拟合），返回原始特征空间的系数。"""
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    pos_w = 0.5 / max(y.mean(), 1e-6)
    neg_w = 0.5 / max(1 - y.mean(), 1e-6)
    sw = np.where(y == 1, pos_w, neg_w)

    def loss(th):
        z = Z @ th[1:] + th[0]
        p = sigmoid(z)
        ll = -np.sum(sw * (y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))) / len(y)
        g = Z.T @ (sw * (p - y)) / len(y)
        g0 = np.sum(sw * (p - y)) / len(y)
        return ll + l2 * np.sum(th[1:] ** 2), np.concatenate([[g0], g + 2 * l2 * th[1:]])

    th = minimize(loss, np.zeros(X.shape[1] + 1), jac=True, method="L-BFGS-B").x
    coef = th[1:] / sd
    intercept = th[0] - np.sum(th[1:] * mu / sd)
    # 类别加权训练会把截距拉向 0.5 先验；用真实正样本比例做先验校正，输出校准后的概率
    prior = np.log(max(y.mean(), 1e-6) / max(1 - y.mean(), 1e-6))
    return {"coef": coef.tolist(), "intercept": float(intercept + prior), "features": LR_FEATURES}
