import numpy as np
from conftest import make_seq, make_tracklet

from offlinetracking.cmc import CMCTrack
from offlinetracking.gating import build_candidates
from offlinetracking.quality import compute_quality, ioa


def test_ioa():
    b = np.array([50, 50, 20, 20.0])
    assert np.isclose(ioa(b, np.array([[60, 50, 20, 20.0]]))[0], 0.5)


def test_quality_truncation_and_occlusion(cfg):
    a = make_tracklet(1, [1, 2], 3, 0, y=100, w=20, h=20)          # 贴左边
    b = make_tracklet(2, [1, 2], 300, 0, y=100, w=40, h=40)
    c = make_tracklet(3, [1, 2], 300, 0, y=110, w=40, h=40)        # 底边更低，遮挡 b
    s = make_seq([a, b, c])
    compute_quality(s, cfg)
    assert a.q_trunc[0] == cfg["quality"]["trunc_value"] and a.exit_border
    assert b.q_occ[0] < 1 and c.q_occ[0] == 1


def test_gating_time_class_and_motion(cfg):
    cfg["gating"]["v_max"] = {"default": 2.0}
    cfg["gating"]["g0"] = {"default": 0.5}
    i = make_tracklet(1, range(1, 11), 100, 2)                   # 结束于 x=118
    j_ok = make_tracklet(2, range(13, 20), 122, 2)
    j_far = make_tracklet(3, range(13, 20), 400, 2)               # 太远
    j_cls = make_tracklet(4, range(13, 20), 122, 2, cls="truck")
    j_late = make_tracklet(5, range(60, 70), 122, 2)              # 超过 τ_max
    j_overlap = make_tracklet(6, range(8, 15), 122, 2)
    s = make_seq([i, j_ok, j_far, j_cls, j_late, j_overlap])
    compute_quality(s, cfg)
    E = build_candidates(s.tracklets, CMCTrack.identity(100), 10.0, cfg)
    assert {(e.i, e.j) for e in E} == {(0, 1)}
    assert E[0].gap == 3 and np.isclose(E[0].tau, 0.3)


def test_gating_cmc_failure_relaxes_and_flags(cfg):
    cfg["gating"]["v_max"] = {"default": 1.0}
    cfg["gating"]["g0"] = {"default": 0.5}
    i = make_tracklet(1, range(1, 11), 100, 0)
    j = make_tracklet(2, range(12, 20), 160, 0)                  # g = 60/35 ≈ 1.71 > 0.7，放宽 2 倍后 1.4…仍不够
    j2 = make_tracklet(3, range(12, 20), 140, 0)                 # g ≈ 1.14：放宽后通过
    s = make_seq([i, j, j2])
    compute_quality(s, cfg)
    cmc = CMCTrack.identity(100)
    cmc.rho_t[:] = 0.0
    E = build_candidates(s.tracklets, cmc, 10.0, cfg)
    assert [(e.i, e.j) for e in E] == [(0, 2)] and "cmc_fail" in E[0].flags
