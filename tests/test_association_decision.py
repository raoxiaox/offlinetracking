import numpy as np
from conftest import make_seq, make_tracklet

from offlinetracking.association import solve_greedy, solve_hungarian
from offlinetracking.competition import competition
from offlinetracking.decision import decide, interpolate, merge_chains
from offlinetracking.structs import Edge


def E(i, j, C):
    e = Edge(i, j, gap=1, tau=0.1)
    e.C = C
    return e


def test_competition_margin_and_mnn():
    es = [E(0, 1, 0.20), E(0, 2, 0.22), E(3, 2, 0.6)]
    competition(es, C0=0.5, delta_m=0.1)
    assert es[0].mnn and np.isclose(es[0].A, 0.02)
    assert not es[1].mnn and np.isclose(es[1].A, -0.02)
    assert len(es[0].competitors) == 1
    assert not es[2].mnn          # 比拒绝代价差


def test_hungarian_global_optimum_beats_greedy():
    # 贪心先拿 0→2(0.1)，导致 1 无处可去；全局最优是 0→3、1→2
    es = [E(0, 2, 0.10), E(0, 3, 0.15), E(1, 2, 0.12)]
    n = 4
    ce, cs = np.full(n, 0.5), np.full(n, 0.5)
    solve_hungarian(n, es, ce / 2, cs / 2)
    assert {(e.i, e.j) for e in es if e.x} == {(0, 3), (1, 2)}
    solve_greedy(es, ce / 2, cs / 2, False, 0.0)
    assert {(e.i, e.j) for e in es if e.x} == {(0, 2)}


def test_rejection_option():
    es = [E(0, 1, 0.9)]
    solve_hungarian(2, es, np.full(2, 0.25), np.full(2, 0.25))
    assert es[0].x == 0
    es[0].C = 0.3
    solve_hungarian(2, es, np.full(2, 0.25), np.full(2, 0.25))
    assert es[0].x == 1


def test_decide_three_tiers(cfg):
    d = cfg["decision"]
    a = E(0, 1, d["theta_auto"] - 0.01)
    a.x, a.A, a.E = 1, 1.0, 1.0
    b = E(2, 3, d["theta_auto"] - 0.01)
    b.x, b.A, b.E = 1, 1.0, 1.0
    b.flags.add("cmc_fail")
    c = E(4, 5, d["theta_review"] + 1)
    decide([a, b, c], cfg)
    assert (a.decision, b.decision, c.decision) == ("auto", "review", "reject")


def test_merge_chains():
    gid = merge_chains(5, [(0, 2), (2, 4)])
    assert gid[0] == gid[2] == gid[4] and len(set(gid)) == 3


def test_interpolation_is_continuous(cfg):
    from offlinetracking.cmc import CMCTrack
    from offlinetracking.motion import end_fits, motion_evidence
    from offlinetracking.quality import compute_quality
    i = make_tracklet(1, range(1, 11), 100, 2)
    j = make_tracklet(2, range(16, 26), 130, 2)
    s = make_seq([i, j])
    compute_quality(s, cfg)
    cmc = CMCTrack.identity(100)
    e = Edge(0, 1, gap=6, tau=0.6)
    motion_evidence(e, s, end_fits(s, cmc, cfg), cmc, cfg)
    ip = interpolate(e, s, cfg)
    assert np.array_equal(ip[:, 0], np.arange(11, 16))
    assert np.allclose(ip[:, 1], 100 + 2 * (ip[:, 0] - 1))
    assert np.allclose(ip[:, 4], 30)
