import numpy as np
from conftest import make_seq, make_tracklet

from offlinetracking.cmc import CMCTrack
from offlinetracking.motion import end_fits, fit_end, motion_evidence, propagate
from offlinetracking.quality import compute_quality
from offlinetracking.structs import Edge


def test_fit_end_recovers_velocity_and_scale_trend():
    fr = np.arange(1, 11)
    pts = np.column_stack([10 + 3.0 * fr, 50 - 1.0 * fr])
    h = 20 * np.exp(0.02 * fr)
    cmc = CMCTrack.identity(20)
    f = fit_end(fr, pts, h, np.ones(10), "tail", 8, "lsq", cmc)
    assert np.allclose(f.v, [3, -1]) and np.allclose(f.a, pts[-1]) and np.isclose(f.beta, 0.02)
    g = fit_end(fr, pts, h, np.ones(10), "head", 8, "lsq", cmc)
    assert np.allclose(g.v, [3, -1]) and np.allclose(g.a, pts[0])


def test_fit_end_low_motion():
    cmc = CMCTrack.identity(5)
    f = fit_end(np.array([1, 2]), np.array([[0, 0], [1, 1.0]]), np.array([10, 10.0]), np.ones(2), "tail", 8, "lsq", cmc)
    assert f.low and np.allclose(f.v, 0)


def test_propagate_with_camera_translation():
    n = 10
    H = np.tile(np.eye(3), (n, 1, 1))
    H[1:, 0, 2] = 2.0   # 相机运动让背景每帧右移 2 像素
    cmc = CMCTrack(H, np.ones(n))
    fwd = propagate(np.array([0.0, 0]), np.array([1.0, 0]), 1, 3, cmc, True)
    assert np.allclose(fwd[-1], [9, 0])           # 每帧：物体 +1，背景 +2
    bwd = propagate(fwd[-1], np.array([1.0, 0]), 4, 3, cmc, False)
    assert np.allclose(bwd[-1], [0, 0])


def test_two_sided_motion_consistent_track_has_small_error(cfg):
    i = make_tracklet(1, range(1, 11), 100, 2)
    j = make_tracklet(2, range(15, 25), 128, 2)
    s = make_seq([i, j])
    compute_quality(s, cfg)
    cmc = CMCTrack.identity(100)
    fits = end_fits(s, cmc, cfg)
    e = Edge(0, 1, gap=5, tau=0.5)
    motion_evidence(e, s, fits, cmc, cfg)
    assert e.e_f < 1e-6 and e.e_b < 1e-6 and e.C_m < 1e-6
    assert "motion_inconsistent" not in e.flags and e.Q_m > 0


def test_nll_penalises_uncertainty(cfg):
    cfg["motion"]["cost"] = "nll"
    i = make_tracklet(1, range(1, 11), 100, 2)
    j = make_tracklet(2, range(15, 25), 128, 2)
    s = make_seq([i, j])
    compute_quality(s, cfg)
    cmc = CMCTrack.identity(100)
    fits = end_fits(s, cmc, cfg)
    e1 = Edge(0, 1, gap=5, tau=0.5)
    motion_evidence(e1, s, fits, cmc, cfg)
    cfg["motion"]["sigma_obs"] *= 4
    e2 = Edge(0, 1, gap=5, tau=0.5)
    motion_evidence(e2, s, fits, cmc, cfg)
    assert e2.C_m > e1.C_m          # 零误差时，协方差越大代价越大（log det 项）
