import numpy as np
from conftest import make_seq, make_tracklet

from offlinetracking.eval.labels import assign_gt, true_links
from offlinetracking.eval.metrics import ece, mot_counts


def gt_rows(gid, frames, x0, vx):
    return [[f, gid, x0 + vx * (f - frames[0]), 100, 40, 30, 1.0, 0] for f in frames]


def test_assign_gt_and_true_links():
    a = make_tracklet(1, range(1, 6), 100, 2)
    b = make_tracklet(2, range(8, 12), 114, 2)
    c = make_tracklet(3, range(14, 18), 126, 2)
    d = make_tracklet(4, range(1, 6), 400, 0)
    s = make_seq([a, b, c, d])
    s.gt = np.array(gt_rows(7, list(range(1, 18)), 100, 2) + gt_rows(9, list(range(1, 6)), 400, 0), float)
    assign_gt(s)
    assert [t.gt_id for t in s.tracklets] == [7, 7, 7, 9]
    assert set(true_links(s.tracklets)) == {(0, 1), (1, 2)}     # 0→2 不是直接后继


def test_mot_counts_perfect_and_switch():
    gt = np.array(gt_rows(1, list(range(1, 11)), 100, 2), float)
    perfect = gt.copy()
    m = mot_counts(perfect, gt)
    assert m["idtp"] == 10 and m["idsw"] == 0
    split = gt.copy()
    split[5:, 1] = 2
    m2 = mot_counts(split, gt)
    assert m2["idsw"] == 1 and m2["idtp"] == 5


def test_ece_perfectly_calibrated():
    p = np.array([0.0] * 10 + [1.0] * 10)
    y = np.array([0] * 10 + [1] * 10)
    assert ece(p, y)[0] == 0.0
