import numpy as np

from offlinetracking.seqio import load_sequence, save_sequence
from offlinetracking.synth import SynthParams, generate_sequence


def test_sequence_roundtrip(tmp_path):
    s, _ = generate_sequence("rt", 3, SynthParams(n_frames=40, render=False))
    save_sequence(s, tmp_path / "rt")
    r = load_sequence(tmp_path / "rt")
    assert len(r.tracklets) == len(s.tracklets)
    a, b = s.tracklets[0], r.tracklets[0]
    assert np.array_equal(a.frames, b.frames) and np.allclose(a.boxes, b.boxes, atol=0.01)
    assert np.allclose(a.feats, b.feats, atol=1e-6)
    assert np.allclose(r.camera["Hg"], s.camera["Hg"]) and np.allclose(r.ego[1:], s.ego[1:], atol=1e-3)
    assert np.allclose(r.gt[:, 2:6], s.gt[:, 2:6], atol=0.01)
