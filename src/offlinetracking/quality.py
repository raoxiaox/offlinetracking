"""模块 0.2 / 0.3：观测质量 q 与边界标记（§3.2、§3.3）。"""

from __future__ import annotations

import numpy as np

from .structs import Sequence


def ioa(b: np.ndarray, others: np.ndarray) -> np.ndarray:
    """b (4,) 与 others (m,4) 的 IoA = |b∩o| / |b|，cxcywh。"""
    if len(others) == 0:
        return np.zeros(0)
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    ox1, oy1 = others[:, 0] - others[:, 2] / 2, others[:, 1] - others[:, 3] / 2
    ox2, oy2 = others[:, 0] + others[:, 2] / 2, others[:, 1] + others[:, 3] / 2
    iw = np.clip(np.minimum(bx2, ox2) - np.maximum(bx1, ox1), 0, None)
    ih = np.clip(np.minimum(by2, oy2) - np.maximum(by1, oy1), 0, None)
    return iw * ih / max(b[2] * b[3], 1e-9)


def near_border(box: np.ndarray, w: int, h: int, m: float) -> bool:
    cx, cy, bw, bh = box
    return cx - bw / 2 < m or cy - bh / 2 < m or cx + bw / 2 > w - m or cy + bh / 2 > h - m


def compute_quality(seq: Sequence, cfg: dict, blur: dict | None = None) -> None:
    """为每条轨迹的每个观测填 q、q_trunc、q_occ，以及 exit/enter_border。"""
    qc = cfg["quality"]
    s_min, s_max, m = qc["s_min"], qc["s_max"], qc["border_margin"]
    by_frame: dict[int, list] = {}
    for ti, tr in enumerate(seq.tracklets):
        for k, f in enumerate(tr.frames):
            by_frame.setdefault(int(f), []).append((ti, k))
    for tr in seq.tracklets:
        n = len(tr)
        tr.q_trunc = np.ones(n)
        tr.q_occ = np.ones(n)
    for _f, lst in by_frame.items():
        boxes = np.array([seq.tracklets[ti].boxes[k] for ti, k in lst])
        bottoms = boxes[:, 1] + boxes[:, 3] / 2
        for a, (ti, k) in enumerate(lst):
            tr = seq.tracklets[ti]
            if near_border(boxes[a], seq.img_w, seq.img_h, m):
                tr.q_trunc[k] = qc["trunc_value"]
            occl = bottoms > bottoms[a]
            occl[a] = False
            v = ioa(boxes[a], boxes[occl])
            tr.q_occ[k] = 1.0 - (v.max() if len(v) else 0.0)
    for ti, tr in enumerate(seq.tracklets):
        size = np.sqrt(tr.boxes[:, 2] * tr.boxes[:, 3])
        q_size = np.clip((size - s_min) / (s_max - s_min), 0, 1)
        q = q_size * np.clip(tr.scores, 0, 1) * tr.q_trunc * tr.q_occ
        if qc.get("use_blur") and blur is not None and ti in blur:
            q = q * blur[ti]
        tr.q = q
        tr.exit_border = near_border(tr.boxes[-1], seq.img_w, seq.img_h, m)
        tr.enter_border = near_border(tr.boxes[0], seq.img_w, seq.img_h, m)
