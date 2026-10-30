"""参数标定（§11）：所有阈值和权重都从标定集得到。

步骤：
1. 宽松门控收集候选边，用真实 + 人造缺口正样本统计 v_max、g0（99.5 分位）、κ_m、κ_s（95 分位）、
   双端一致性上限、轨迹纯度上限、外观逻辑回归 β0/β1 与 δ_a；
2. 用标定后的门控重建候选边；按配置标定 NLL 的 σ_obs（卡方校准检查）、融合权重（AUC 网格）或逻辑回归；
3. 只在真实序列上：θ_link（链接 F1 最大）、自动档 θ_auto / δ_auto / E_min（精度 ≥ 目标时覆盖率最大）、θ_review。
"""

from __future__ import annotations

import copy
import itertools
import logging

import numpy as np
from scipy.stats import rankdata

from .appearance import tracklet_self_distance
from .config import per_class
from .eval.labels import artificial_gaps, assign_gt, label_edges, true_links
from .fusion import enabled_kinds, fit_logreg, fuse, lr_vector
from .motion import end_fits, motion_evidence
from .pipeline import build_edges, link, prepare
from .structs import Sequence

log = logging.getLogger("calibrate")


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = labels == 1
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def q(x, p, default):
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    return float(np.percentile(x, p)) if len(x) >= 10 else default


class Data:
    """一组序列及其 CMC、边与真链接。"""

    def __init__(self, seqs: list[Sequence], cfg: dict):
        self.seqs = seqs
        self.cmcs = [prepare(s, cfg) for s in seqs]
        self.edges: list = []
        self.links: list = []

    def build(self, cfg, loose=False):
        self.edges, self.links = [], []
        for s, c in zip(self.seqs, self.cmcs):
            es = build_edges(s, c, cfg, loose=loose)
            lk = true_links(s.tracklets)
            label_edges(es, lk)
            self.edges.append(es)
            self.links.append(lk)

    def all_edges(self):
        return [e for es in self.edges for e in es]


def _set_motion(data: Data, cfg: dict):
    for s, c, es in zip(data.seqs, data.cmcs, data.edges):
        fits = end_fits(s, c, cfg)
        for e in es:
            e.flags.discard("motion_uncertain")
            motion_evidence(e, s, fits, c, cfg)


def calibrate(cfg: dict, seqs: list[Sequence], seed: int = 0, n_cuts: int = 15) -> tuple[dict, dict]:
    cfg = copy.deepcopy(cfg)
    rep: dict = {}
    if not cfg["pipeline"].get("link", True):
        return cfg, rep
    rng = np.random.default_rng(seed)
    ec = cfg["eval"]
    for s in seqs:
        assign_gt(s, ec["iou_thr"], ec["purity_thr"])
    aug = [artificial_gaps(s, rng, n_cuts) for s in seqs]
    real = Data(seqs, cfg)
    both = Data(seqs + aug, cfg)
    fps = seqs[0].fps

    # ---- 1. 证据分布 ----
    both.build(cfg, loose=True)
    E = both.all_edges()
    pos = [e for e in E if e.label == 1]
    gc = cfg["gating"]
    allp = {"default": pos}
    for c in sorted({e.cls for e in pos}):
        allp[c] = [e for e in pos if e.cls == c]
    for c, P in allp.items():
        if len(P) < 30 and c != "default":
            continue
        g0 = q([e.g for e in P if e.gap == 1], 99.5, q([e.g for e in P], 50, 1.0))
        vmax = q([max(e.g - g0, 0) / e.tau for e in P], 99.5, 3.0)
        mg = gc.get("calib_margin", 1.0)
        gc["g0"][c], gc["v_max"][c] = round(g0 * mg, 4), round(max(vmax, 0.1) * mg, 4)
    rep["n_pos_loose"] = len(pos)
    mc = cfg["motion"]
    good = [e for e in pos if "low_motion" not in e.flags]
    mc["kappa_m"] = round(q([e.e_f for e in good] + [e.e_b for e in good], 95, mc["kappa_m"]), 4)
    mc["consistency_max"] = round(q([abs(e.e_f - e.e_b) for e in good], 99, mc["consistency_max"]), 4)
    res_s = []
    for e in pos:
        if e.Q_s > 0 and e.pred is not None:
            fi, fj = e.pred["fi"], e.pred["fj"]
            res_s.append(abs((fj.alpha - fi.alpha) - 0.5 * (fi.beta + fj.beta) * e.gap))
    cfg["scale"]["kappa_s"] = round(max(q(res_s, 95, cfg["scale"]["kappa_s"]), 0.02), 4)

    pure = [tracklet_self_distance(t) for s in both.seqs for t in s.tracklets if t.gt_id >= 0 and t.purity >= 0.99]
    ac = cfg["appearance"]
    ac["purity_max"] = round(q(pure, 99.5, ac["purity_max"]) * 1.2, 4)
    if ac.get("enabled", True):
        def gated(e):
            return e.g <= per_class(gc["v_max"], e.cls) * e.tau + per_class(gc["g0"], e.cls)
        A = [e for e in E if gated(e) and np.isfinite(e.d_set)]
        if sum(e.label for e in A) >= 10:
            X = np.array([[e.d_set] for e in A])
            y = np.array([e.label for e in A], float)
            lr = fit_logreg(X, y, l2=1e-4)
            ac["beta0"], ac["beta1"] = round(lr["intercept"], 4), round(lr["coef"][0], 4)
        ac["delta_a"] = round(q([abs(e.d_set - e.d_proto) for e in pos], 99, ac["delta_a"]), 4)
        from .appearance import sigmoid
        ca_pos = [1 - sigmoid(ac["beta0"] + ac["beta1"] * e.d_set) for e in pos
                  if np.isfinite(e.d_set) and e.Q_a >= ac["veto_q"]]
        ac["veto_ca"] = round(max(q(ca_pos, 99.5, ac["veto_ca"]), 0.9), 6)

    # ---- 2. 标定门控下的候选边 ----
    both.build(cfg)
    E = both.all_edges()
    y = np.array([e.label for e in E])
    rep["gate_recall_calib"] = _gate_recall(both, cfg, fps)
    if mc.get("cost") == "nll":
        best = None
        for so in [0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5]:
            mc["sigma_obs"] = so
            _set_motion(both, cfg)
            d2 = [e.d2 for e in both.all_edges() if e.label == 1 and np.isfinite(e.d2)]
            frac = float(np.mean(np.array(d2) <= 5.99)) if d2 else 0.0
            if frac >= 0.95:
                best = (so, frac)
                break
            best = (so, frac)
        mc["sigma_obs"] = best[0]
        rep["nll_chi2_frac"] = round(best[1], 4)
        if best[1] < mc["chi2_min_frac"]:
            log.warning("NLL 卡方校准未达标（%.3f），回退到截断线性代价", best[1])
            mc["cost"] = "psi"
            rep["nll_fallback"] = True
        _set_motion(both, cfg)
        mc["trace_max"] = round(q([e.trace for e in both.all_edges() if e.label == 1], 99, mc["trace_max"]), 4)
        _set_motion(both, cfg)
        E = both.all_edges()

    fc = cfg["fusion"]
    if fc.get("mode") == "logreg":
        X = np.array([lr_vector(e, cfg) for e in E])
        fc["logreg"] = fit_logreg(X, y.astype(float), l2=1e-3)
        cfg["association"]["theta_link"] = 0.0
        rep["auc"] = auc(-np.array([-(np.dot(fc["logreg"]["coef"], x) + fc["logreg"]["intercept"]) for x in X]), y)
    else:
        kinds = enabled_kinds(cfg)
        grid = {"w_m": [0.3, 0.5, 0.7], "w_a": [0.2, 0.4, 0.6] if "a" in kinds else [fc["w_a"]],
                "w_s": [0.0, 0.1, 0.2] if "s" in kinds else [fc["w_s"]], "w_t": [0.05, 0.1, 0.2]}
        best = (-1, None)
        for vals in itertools.product(*grid.values()):
            fc.update(dict(zip(grid.keys(), vals)))
            fuse(E, cfg)
            a = auc(-np.array([e.C for e in E]), y)
            if a > best[0]:
                best = (a, dict(zip(grid.keys(), vals)))
        fc.update(best[1])
        rep["auc"] = round(best[0], 4)

    # ---- 3. 阈值（只用真实序列）----
    real.build(cfg)
    fuse(real.all_edges(), cfg)
    if fc.get("mode") != "logreg":
        Cs = np.array([e.C for e in real.all_edges()])
        best = (-1, cfg["association"]["theta_link"])
        for th in np.unique(np.round(np.percentile(Cs, np.linspace(5, 95, 19)), 4)):
            cfg["association"]["theta_link"] = float(th)
            f1 = _link_f1(real, cfg, fps)
            if f1 > best[0]:
                best = (f1, float(th))
        cfg["association"]["theta_link"] = best[1]
        rep["link_f1_calib"] = round(best[0], 4)
    else:
        rep["link_f1_calib"] = round(_link_f1(real, cfg, fps), 4)
    _calibrate_auto(real, cfg, rep)
    return cfg, rep


def _gate_recall(data: Data, cfg: dict, fps: float) -> float:
    n = hit = 0
    for es, lk in zip(data.edges, data.links):
        eset = {(e.i, e.j) for e in es}
        for k, g in lk.items():
            if g <= cfg["gating"]["tau_max"] * fps:
                n += 1
                hit += k in eset
    return round(hit / n, 4) if n else float("nan")


def _link_f1(data: Data, cfg: dict, fps: float) -> float:
    tp = pred = n_true = 0
    for s, es, lk in zip(data.seqs, data.edges, data.links):
        link(s, es, cfg)
        pred += sum(e.x for e in es)
        tp += sum(e.x * e.label for e in es)
        n_true += sum(1 for g in lk.values() if g <= cfg["gating"]["tau_max"] * fps)
    p, r = tp / max(pred, 1), tp / max(n_true, 1)
    return 2 * p * r / max(p + r, 1e-9)


def _calibrate_auto(data: Data, cfg: dict, rep: dict) -> None:
    """自动档：在 x=1 的边上网格搜 (δ_auto, E_min)，每组取精度 ≥ 目标的最大 θ_auto，选覆盖率最大的一组。"""
    dc = cfg["decision"]
    target = dc["precision_target"]
    pseudo = dc.get("precision_pseudo_fp", 0)
    for s, es in zip(data.seqs, data.edges):
        link(s, es, cfg)
    X = [e for es in data.edges for e in es if e.x == 1]
    n_true = sum(len(lk) for lk in data.links)
    block = set(dc["blocking_flags"])
    use_comp = cfg["competition"].get("enabled", True)
    tune_e = cfg["fusion"].get("mode") != "fixed"
    As = np.array([e.A for e in X]) if X else np.zeros(1)
    deltas = sorted({0.0, *np.round(np.percentile(As[np.isfinite(As)], [10, 25, 40]), 4)}) if use_comp else [-np.inf]
    e_mins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5] if tune_e else [0.0]
    best = None
    for d, em in itertools.product(deltas, e_mins):
        cand = sorted((e for e in X if e.A >= d and e.E >= em and not (e.flags & block)), key=lambda e: e.C)
        tp, th, cov = 0, None, 0.0
        for k, e in enumerate(cand):
            tp += e.label
            if tp / (k + 1 + pseudo) >= target and (k + 1 == len(cand) or cand[k + 1].C > e.C):
                th, cov = e.C, tp / max(n_true, 1)
        key = (cov, d, em)
        if th is not None and (best is None or key > best[0]):
            best = (key, th, d, em)
    if best is None:
        dc["theta_auto"] = -1e9
        rep["auto_coverage_calib"] = 0.0
    else:
        (cov, _, _), th, d, em = best
        dc["theta_auto"], dc["delta_auto"], dc["e_min"] = float(th), float(d) if np.isfinite(d) else 0.0, float(em)
        rep["auto_coverage_calib"] = round(cov, 4)
    posC = [e.C for es in data.edges for e in es if e.label == 1]
    dc["theta_review"] = float(max(q(posC, 95, dc["theta_review"]), cfg["association"]["theta_link"]))
