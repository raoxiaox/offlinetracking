"""阶段 0：轨迹归属与边标签（§11.1）。"""

from __future__ import annotations

import copy

import numpy as np

from ..structs import Edge, Sequence, Tracklet


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a (n,4)、b (m,4)，cxcywh。"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a1, a2 = a[:, None, :2] - a[:, None, 2:] / 2, a[:, None, :2] + a[:, None, 2:] / 2
    b1, b2 = b[None, :, :2] - b[None, :, 2:] / 2, b[None, :, :2] + b[None, :, 2:] / 2
    wh = np.clip(np.minimum(a2, b2) - np.maximum(a1, b1), 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area = a[:, None, 2] * a[:, None, 3] + b[None, :, 2] * b[None, :, 3] - inter
    return inter / np.maximum(area, 1e-9)


def gt_by_frame(gt: np.ndarray) -> dict[int, np.ndarray]:
    out: dict[int, list] = {}
    for r in gt:
        out.setdefault(int(r[0]), []).append(r)
    return {k: np.array(v) for k, v in out.items()}


def assign_gt(seq: Sequence, iou_thr: float = 0.5, purity_thr: float = 0.8) -> None:
    """逐帧 IoU > thr 匹配 + 多数投票；多数票占比低于 purity_thr 的轨迹记为不纯（gt_id 仍取多数）。"""
    if seq.gt is None:
        return
    gbf = gt_by_frame(seq.gt)
    for tr in seq.tracklets:
        votes: dict[int, int] = {}
        for f, b in zip(tr.frames, tr.boxes):
            g = gbf.get(int(f))
            if g is None:
                continue
            iou = iou_matrix(b[None], g[:, 2:6])[0]
            k = int(np.argmax(iou))
            if iou[k] > iou_thr:
                votes[int(g[k, 1])] = votes.get(int(g[k, 1]), 0) + 1
        if votes:
            gid, cnt = max(votes.items(), key=lambda kv: kv[1])
            tr.gt_id, tr.purity = gid, cnt / len(tr)
        else:
            tr.gt_id, tr.purity = -1, 0.0
    seq.meta["n_impure_gt"] = int(sum(tr.gt_id >= 0 and tr.purity < purity_thr for tr in seq.tracklets))


def true_links(trs: list[Tracklet]) -> dict[tuple[int, int], int]:
    """直接后继：同一真值 ID，j 是 i 结束后最早开始的片段，且 i 是 j 开始前最晚结束的片段。返回 {(i,j): gap}。"""
    by_gt: dict[int, list[int]] = {}
    for k, tr in enumerate(trs):
        if tr.gt_id >= 0:
            by_gt.setdefault(tr.gt_id, []).append(k)
    links = {}
    for ks in by_gt.values():
        for i in ks:
            after = [j for j in ks if trs[j].t_s > trs[i].t_e]
            if not after:
                continue
            j = min(after, key=lambda j: (trs[j].t_s, -len(trs[j])))
            before = [k for k in ks if trs[k].t_e < trs[j].t_s]
            if max(before, key=lambda k: (trs[k].t_e, len(trs[k]))) == i:
                links[(i, j)] = trs[j].t_s - trs[i].t_e
    return links


def label_edges(edges: list[Edge], links: dict) -> None:
    for e in edges:
        e.label = int((e.i, e.j) in links)


def artificial_gaps(seq: Sequence, rng: np.random.Generator, n_cuts: int, gap_sec=(0.2, 3.0),
                    min_side: int = 6) -> Sequence:
    """在长轨迹中间挖去 0.2–3 s，扩充长缺口正样本（只用于标定）。"""
    new = copy.copy(seq)
    trs = [copy.copy(t) for t in seq.tracklets]
    cand = [k for k, t in enumerate(trs) if t.gt_id >= 0]
    rng.shuffle(cand)
    out, cut = [], set()
    for k in cand:
        if len(cut) >= n_cuts:
            break
        tr = trs[k]
        gap = int(rng.uniform(*gap_sec) * seq.fps)
        if len(tr) < 2 * min_side + gap:
            continue
        c = int(rng.integers(min_side, len(tr) - min_side - gap + 1))
        for sl in (slice(0, c), slice(c + gap, None)):
            out.append(Tracklet(0, tr.cls, tr.frames[sl], tr.boxes[sl], tr.scores[sl],
                                None if tr.feats is None else tr.feats[sl], gt_id=tr.gt_id, purity=tr.purity))
        cut.add(k)
    out += [t for k, t in enumerate(trs) if k not in cut]
    out.sort(key=lambda t: (t.t_s, t.t_e))
    for k, t in enumerate(out):
        t.tid = k + 1
    new.tracklets = out
    new.meta = dict(seq.meta, artificial=True)
    return new
