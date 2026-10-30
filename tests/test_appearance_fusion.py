import numpy as np
from conftest import make_seq, make_tracklet, unit

from offlinetracking.appearance import AppearanceModel, select_reps, set_distances
from offlinetracking.fusion import fit_logreg, fuse
from offlinetracking.quality import compute_quality
from offlinetracking.structs import Edge


def test_select_reps_quality_and_spacing():
    tr = make_tracklet(1, range(1, 31), 100, 0)
    tr.q = np.linspace(0.1, 1.0, 30)
    idx = select_reps(tr, "tail", 20, 0.05, 5, 3)
    fr = tr.frames[idx]
    assert len(idx) == 5 and np.all(np.diff(fr) >= 3) and fr.max() == 30


def test_set_distances():
    a = np.array([unit([1, 0, 0]), unit([1, 0.1, 0])])
    b = np.array([unit([0, 1, 0])])
    d_set, d_proto = set_distances(a, b, np.ones(2), np.ones(1))
    assert 0.9 < d_set <= 1.0 and 0.9 < d_proto <= 1.0


def test_appearance_veto_and_availability(cfg):
    fa, fb = unit([1, 0, 0, 0]), unit([0, 1, 0, 0])
    i = make_tracklet(1, range(1, 21), 100, 0, w=60, h=60, feat=fa)
    j_same = make_tracklet(2, range(25, 45), 100, 0, w=60, h=60, feat=fa)
    j_diff = make_tracklet(3, range(25, 45), 100, 0, w=60, h=60, feat=fb)
    s = make_seq([i, j_same, j_diff])
    compute_quality(s, cfg)
    app = AppearanceModel(s, cfg)
    e1, e2 = Edge(0, 1, 5, 0.5), Edge(0, 2, 5, 0.5)
    app.evidence(e1)
    app.evidence(e2)
    assert e1.C_a < 0.1 and e1.Q_a > 0 and "app_veto" not in e1.flags
    assert e2.C_a > 0.9 and "app_veto" in e2.flags


def test_adaptive_fusion_and_sufficiency(cfg):
    e = Edge(0, 1, 5, 0.5, C_m=0.2, Q_m=1.0, C_a=0.9, Q_a=0.0, C_s=0.5, Q_s=0.0, C_t=0.1)
    fuse([e], cfg)
    wt = cfg["fusion"]["w_t"]
    assert np.isclose(e.C, (1 - wt) * 0.2 + wt * 0.1)    # 只有运动可信 → 权重全给运动
    f = cfg["fusion"]
    assert np.isclose(e.E, f["w_m"] / (f["w_m"] + f["w_a"] + f["w_s"]))   # 但证据充分度低
    cfg["fusion"]["mode"] = "fixed"
    fuse([e], cfg)
    assert e.C > (1 - wt) * 0.2 + wt * 0.1


def test_fit_logreg_separable():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 0.3, 200), rng.normal(2, 0.3, 200)])
    y = np.concatenate([np.ones(200), np.zeros(200)])
    lr = fit_logreg(x[:, None], y)
    z = lr["coef"][0] * np.array([0.0, 2.0]) + lr["intercept"]
    assert z[0] > 2 and z[1] < -2
