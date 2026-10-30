import numpy as np
import pytest

from offlinetracking.config import load_config
from offlinetracking.structs import Sequence, Tracklet


@pytest.fixture
def cfg():
    return load_config()


def make_tracklet(tid, frames, x0, vx, y=100.0, w=40.0, h=30.0, cls="car", feat=None, score=0.9):
    frames = np.asarray(frames)
    boxes = np.column_stack([x0 + vx * (frames - frames[0]), np.full(len(frames), y),
                             np.full(len(frames), w), np.full(len(frames), h)])
    feats = None if feat is None else np.tile(feat, (len(frames), 1)).astype(np.float32)
    return Tracklet(tid, cls, frames, boxes, np.full(len(frames), score), feats)


def make_seq(trs, n_frames=100):
    return Sequence(name="t", fps=10.0, img_w=640, img_h=360, tracklets=trs, n_frames=n_frames)


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)
