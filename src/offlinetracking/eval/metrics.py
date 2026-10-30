"""评测指标（§11.2）。每段序列先出计数，汇总时再算比例，保证多序列合并正确。"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..decision import build_output, merge_chains
from .labels import assign_gt, gt_by_frame, iou_matrix, label_edges, true_links


def mot_counts(pred: np.ndarray, gt: np.ndarray, iou_thr: float = 0.5) -> dict:
    """IDF1 所需的 IDTP，以及 CLEAR 风格的 ID 切换与碎片数。pred/gt 前 6 列：frame, id, cx, cy, w, h。"""
    pbf, gbf = gt_by_frame(pred), gt_by_frame(gt)
    co: dict[tuple[int, int], int] = defaultdict(int)
    last_match: dict[int, int] = {}
    prev_tracked: dict[int, bool] = {}
    idsw = frag = 0
    for f in sorted(gbf):
        g = gbf[f]
        p = pbf.get(f, np.zeros((0, 8)))
        iou = iou_matrix(g[:, 2:6], p[:, 2:6])
        ok = iou >= iou_thr
        for a, b in zip(*np.nonzero(ok)):
            co[(int(g[a, 1]), int(p[b, 1]))] += 1
        # 逐帧匹配：先延续上一帧的匹配，再对其余做匈牙利
        match: dict[int, int] = {}
        used = set()
        for a in range(len(g)):
            gid = int(g[a, 1])
            if gid in last_match:
                cand = [b for b in range(len(p)) if int(p[b, 1]) == last_match[gid] and ok[a, b]]
                if cand:
                    match[a] = cand[0]
                    used.add(cand[0])
        ra = [a for a in range(len(g)) if a not in match]
        rb = [b for b in range(len(p)) if b not in used]
        if ra and rb:
            c = 1 - iou[np.ix_(ra, rb)]
            r, cc = linear_sum_assignment(c)
            for x, y in zip(r, cc):
                if ok[ra[x], rb[y]]:
                    match[ra[x]] = rb[y]
        for a in range(len(g)):
            gid = int(g[a, 1])
            if a in match:
                pid = int(p[match[a], 1])
                if gid in last_match and last_match[gid] != pid:
                    idsw += 1
                if gid in last_match and not prev_tracked.get(gid, False):
                    frag += 1
                last_match[gid] = pid
                prev_tracked[gid] = True
            else:
                prev_tracked[gid] = False
    gids = sorted({int(x) for x in gt[:, 1]})
    pids = sorted({int(x) for x in pred[:, 1]}) if len(pred) else []
    idtp = 0
    if gids and pids:
        gi, pi = {g: k for k, g in enumerate(gids)}, {p: k for k, p in enumerate(pids)}
        M = np.zeros((len(gids), len(pids)))
        for (g, p), v in co.items():
            M[gi[g], pi[p]] = v
        r, c = linear_sum_assignment(-M)
        idtp = int(M[r, c].sum())
    ids_per_gt = defaultdict(set)
    for (g, p), v in co.items():
        if v > 0:
            ids_per_gt[g].add(p)
    per_gt = float(np.mean([len(v) for v in ids_per_gt.values()])) if ids_per_gt else 0.0
    return {"idtp": idtp, "n_gt": len(gt), "n_pred": len(pred), "idsw": idsw, "frag": frag,
            "n_gt_ids": len(gids), "pred_ids_per_gt": per_gt}


def ece(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> tuple[float, list]:
    probs, labels = np.asarray(probs, float), np.asarray(labels, float)
    ok = np.isfinite(probs)
    probs, labels = probs[ok], labels[ok]
    if len(probs) == 0:
        return float("nan"), []
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(probs, edges) - 1, 0, bins - 1)
    err, curve = 0.0, []
    for b in range(bins):
        m = idx == b
        if m.any():
            conf, acc = probs[m].mean(), labels[m].mean()
            err += m.mean() * abs(conf - acc)
            curve.append((float(conf), float(acc), int(m.sum())))
    return float(err), curve


def sequence_counts(result, cfg: dict) -> dict:
    """单序列计数。会给 tracklets 赋 gt_id、给 edges 打标签。"""
    seq, edges = result.seq, result.edges
    ec = cfg["eval"]
    assign_gt(seq, ec["iou_thr"], ec["purity_thr"])
    links = true_links(seq.tracklets)
    label_edges(edges, links)
    fps = seq.fps
    buckets = ec["gap_buckets"]
    nb = len(buckets) - 1

    def bucket(gap_frames):
        s = gap_frames / fps
        return min(int(np.searchsorted(buckets, s, side="right") - 1), nb - 1) if s <= buckets[-1] else None

    c = defaultdict(float)
    c["minutes"] = seq.n_frames / fps / 60
    c["n_tracklets"] = len(seq.tracklets)
    c["n_edges"] = len(edges)
    c["n_true"] = len(links)
    tau_max_f = cfg["gating"]["tau_max"] * fps
    c["n_true_within"] = sum(1 for g in links.values() if g <= tau_max_f)
    eset = {(e.i, e.j) for e in edges}
    c["n_true_in_E"] = sum(1 for k in links if k in eset)
    for g in links.values():
        b = bucket(g)
        if b is not None:
            c[f"true_b{b}"] += 1
    for e in edges:
        b = bucket(e.gap)
        if e.x:
            c["pred"] += 1
            c["tp"] += e.label
            if b is not None:
                c[f"pred_b{b}"] += 1
                c[f"tp_b{b}"] += e.label
        if e.decision == "auto":
            c["n_auto"] += 1
            c["auto_tp"] += e.label
        elif e.decision == "review":
            c["n_review"] += 1
            c["review_pos"] += e.label
    out = dict(c)
    out["probs"] = [e.P for e in edges if np.isfinite(e.P)]
    out["plabels"] = [e.label for e in edges if np.isfinite(e.P)]
    if seq.gt is not None and len(seq.gt):
        gt = seq.gt[:, :6]
        out["mot_auto"] = mot_counts(result.rows, gt, ec["iou_thr"])
        gid_x = merge_chains(len(seq.tracklets), [(e.i, e.j) for e in edges if e.x])
        rows_x = build_output(seq, [e for e in edges if e.x], gid_x, cfg, accept=("auto", "review"))
        out["mot_link"] = mot_counts(rows_x, gt, ec["iou_thr"])
        out["mot_raw"] = mot_counts(build_output(seq, [], np.arange(1, len(seq.tracklets) + 1), cfg), gt, ec["iou_thr"])
    return out


def summarize(counts: list[dict], cfg: dict) -> dict:
    """把多段序列的计数汇总为最终指标。"""
    S = defaultdict(float)
    for c in counts:
        for k, v in c.items():
            if isinstance(v, (int, float)):
                S[k] += v
    div = lambda a, b: float(a / b) if b else float("nan")
    nb = len(cfg["eval"]["gap_buckets"]) - 1
    bk = cfg["eval"]["gap_buckets"]
    m = {
        "gate_recall": div(S["n_true_in_E"], S["n_true_within"]),
        "cands_per_tracklet": div(S["n_edges"], S["n_tracklets"]),
        "link_precision": div(S["tp"], S["pred"]),
        "link_recall": div(S["tp"], S["n_true"]),
        "auto_precision": div(S["auto_tp"], S["n_auto"]),
        "auto_coverage": div(S["auto_tp"], S["n_true"]),
        "n_auto": int(S["n_auto"]),
        "review_per_min": div(S["n_review"], S["minutes"]),
        "review_pos_ratio": div(S["review_pos"], S["n_review"]),
        "n_true_links": int(S["n_true"]),
        "buckets": {f"{bk[b]}-{bk[b + 1]}s": {"precision": div(S[f"tp_b{b}"], S[f"pred_b{b}"]),
                                              "recall": div(S[f"tp_b{b}"], S[f"true_b{b}"]),
                                              "n_true": int(S[f"true_b{b}"])} for b in range(nb)},
    }
    probs = np.concatenate([np.asarray(c.get("probs", []), float) for c in counts]) if counts else np.zeros(0)
    labs = np.concatenate([np.asarray(c.get("plabels", []), float) for c in counts]) if counts else np.zeros(0)
    m["ece"], m["reliability"] = ece(probs, labs) if len(probs) else (float("nan"), [])
    for key in ("mot_auto", "mot_link", "mot_raw"):
        if all(key in c for c in counts) and counts:
            T = defaultdict(float)
            for c in counts:
                for k, v in c[key].items():
                    T[k] += v
            m[key] = {"IDF1": div(2 * T["idtp"], T["n_gt"] + T["n_pred"]), "IDsw": int(T["idsw"]),
                      "Frag": int(T["frag"]), "pred_ids_per_gt": T["pred_ids_per_gt"] / len(counts)}
    return m
