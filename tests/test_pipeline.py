import numpy as np

from offlinetracking.calibrate import calibrate
from offlinetracking.eval.metrics import sequence_counts, summarize
from offlinetracking.pipeline import run_stwo
from offlinetracking.synth import SynthParams, generate_sequence


def small_seqs(n, seed=0):
    p = SynthParams(n_frames=80, render=False)
    out = []
    for k in range(n):
        s, _ = generate_sequence(f"s{k}", seed + k, p)
        out.append(s)
    return out


def test_end_to_end_runs_and_links(cfg):
    cfg["cmc"]["enabled"] = False      # 无帧图像
    seqs = small_seqs(3)
    cal, rep = calibrate(cfg, seqs[:2], n_cuts=5)
    assert rep["gate_recall_calib"] > 0.9
    r = run_stwo(seqs[2], cal)
    m = summarize([sequence_counts(r, cal)], cal)
    assert len(r.edges) > 0
    assert r.rows.shape[1] == 8 and r.gid.max() <= len(seqs[2].tracklets)
    assert m["link_precision"] > 0.7
    assert np.isfinite(m["mot_auto"]["IDF1"])
