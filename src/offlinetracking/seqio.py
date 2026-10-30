"""序列目录的读写。目录结构（详见 README）：

    <seq>/seqinfo.yaml    name, fps, img_w, img_h, n_frames
    <seq>/tracklets.txt   frame,id,x,y,w,h,score,cls     上游短轨迹，x,y 为左上角
    <seq>/feats.npy       (行数, D) ReID 特征，与 tracklets.txt 行顺序一致
    <seq>/gt.txt          frame,gt_id,x,y,w,h,visibility,cls   可选
    <seq>/cmc.npz         H (T+1,3,3)、rho、sigma            可选，预计算的相机运动
    <seq>/camera.yaml     Hg：自车系地面 (x 前, y 左) → 像素   可选，BEV 与地平面运动用
    <seq>/ego.txt         frame,x,y,yaw                       可选，自车位姿
    <seq>/frames/%06d.jpg                                     可选，CMC 与复核裁剪图用
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import yaml

from .structs import Edge, Sequence, Tracklet

CLASSES = ["car", "van", "truck", "pedestrian", "cyclist"]


def tlwh_to_cxcywh(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, float).copy()
    b[..., 0] += b[..., 2] / 2
    b[..., 1] += b[..., 3] / 2
    return b


def cxcywh_to_tlwh(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, float).copy()
    b[..., 0] -= b[..., 2] / 2
    b[..., 1] -= b[..., 3] / 2
    return b


def _read_rows(path: Path) -> list[list[str]]:
    with open(path) as f:
        return [r for r in csv.reader(f) if r and not r[0].startswith("#")]


def write_tracklets(path: Path, tracklets: list[Tracklet]) -> np.ndarray | None:
    """写 tracklets.txt，返回按同样行顺序拼好的特征矩阵。"""
    feats = []
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        for tr in tracklets:
            tl = cxcywh_to_tlwh(tr.boxes)
            for k in range(len(tr)):
                w.writerow([int(tr.frames[k]), tr.tid, *[f"{v:.2f}" for v in tl[k]], f"{tr.scores[k]:.3f}", tr.cls])
                if tr.feats is not None:
                    feats.append(tr.feats[k])
    return np.asarray(feats, np.float32) if feats else None


def read_tracklets(path: Path, feats: np.ndarray | None = None) -> list[Tracklet]:
    rows = _read_rows(path)
    by_id: dict[int, list[int]] = {}
    for r_idx, r in enumerate(rows):
        by_id.setdefault(int(float(r[1])), []).append(r_idx)
    out = []
    for tid, idx in sorted(by_id.items()):
        idx = sorted(idx, key=lambda r_i: int(float(rows[r_i][0])))
        arr = np.array([[float(v) for v in rows[r_i][:7]] for r_i in idx])
        tr = Tracklet(
            tid=tid,
            cls=rows[idx[0]][7] if len(rows[idx[0]]) > 7 else "car",
            frames=arr[:, 0].astype(int),
            boxes=tlwh_to_cxcywh(arr[:, 2:6]),
            scores=arr[:, 6],
            feats=None if feats is None else feats[idx],
        )
        out.append(tr)
    return out


def read_gt(path: Path) -> np.ndarray:
    """返回 (m, 8)：frame, gt_id, cx, cy, w, h, visibility, cls_idx。"""
    rows = _read_rows(path)
    arr = np.array([[float(v) for v in r[:7]] + [CLASSES.index(r[7]) if len(r) > 7 else 0] for r in rows])
    arr[:, 2:6] = tlwh_to_cxcywh(arr[:, 2:6])
    return arr


def write_gt(path: Path, gt: np.ndarray) -> None:
    tl = cxcywh_to_tlwh(gt[:, 2:6])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        for r, b in zip(gt, tl):
            w.writerow([int(r[0]), int(r[1]), *[f"{v:.2f}" for v in b], f"{r[6]:.3f}", CLASSES[int(r[7])]])


def load_sequence(seq_dir: str | Path) -> Sequence:
    d = Path(seq_dir)
    info = yaml.safe_load((d / "seqinfo.yaml").read_text())
    feats = np.load(d / "feats.npy") if (d / "feats.npy").exists() else None
    seq = Sequence(
        name=info.get("name", d.name),
        fps=float(info["fps"]),
        img_w=int(info["img_w"]),
        img_h=int(info["img_h"]),
        tracklets=read_tracklets(d / "tracklets.txt", feats),
        n_frames=int(info.get("n_frames", 0)),
        meta=info,
    )
    if (d / "gt.txt").exists():
        seq.gt = read_gt(d / "gt.txt")
    if (d / "cmc.npz").exists():
        z = np.load(d / "cmc.npz")
        seq.cmc_H, seq.cmc_rho = z["H"], z["rho"]
        seq.cmc_sigma = z["sigma"] if "sigma" in z else None
    if (d / "camera.yaml").exists():
        cam = yaml.safe_load((d / "camera.yaml").read_text())
        cam["Hg"] = np.array(cam["Hg"], float)
        seq.camera = cam
    if (d / "ego.txt").exists():
        e = np.loadtxt(d / "ego.txt", delimiter=",", ndmin=2)
        ego = np.zeros((int(e[:, 0].max()) + 1, 3))
        ego[e[:, 0].astype(int)] = e[:, 1:4]
        seq.ego = ego
    if (d / "frames").is_dir():
        seq.frames_dir = str(d / "frames")
    if not seq.n_frames:
        seq.n_frames = max(tr.t_e for tr in seq.tracklets)
    return seq


def save_sequence(seq: Sequence, seq_dir: str | Path) -> None:
    d = Path(seq_dir)
    d.mkdir(parents=True, exist_ok=True)
    info = {"name": seq.name, "fps": seq.fps, "img_w": seq.img_w, "img_h": seq.img_h, "n_frames": seq.n_frames}
    info.update({k: v for k, v in seq.meta.items() if k not in info})
    (d / "seqinfo.yaml").write_text(yaml.safe_dump(info, sort_keys=False))
    feats = write_tracklets(d / "tracklets.txt", seq.tracklets)
    if feats is not None:
        np.save(d / "feats.npy", feats)
    if seq.gt is not None:
        write_gt(d / "gt.txt", seq.gt)
    if seq.cmc_H is not None:
        np.savez_compressed(d / "cmc.npz", H=seq.cmc_H, rho=seq.cmc_rho,
                            sigma=seq.cmc_sigma if seq.cmc_sigma is not None else np.zeros_like(seq.cmc_rho))
    if seq.camera is not None:
        cam = {k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v) for k, v in seq.camera.items()}
        (d / "camera.yaml").write_text(yaml.safe_dump(cam, sort_keys=False))
    if seq.ego is not None:
        fr = np.arange(len(seq.ego))[1:]
        np.savetxt(d / "ego.txt", np.column_stack([fr, seq.ego[1:]]), delimiter=",", fmt=["%d", "%.4f", "%.4f", "%.6f"])


def list_sequences(root: str | Path, split: str | None = None) -> list[Path]:
    """root 下的序列目录；有 split 时读 root/splits.yaml。"""
    root = Path(root)
    if split:
        names = yaml.safe_load((root / "splits.yaml").read_text())[split]
        return [root / n for n in names]
    return sorted(p for p in root.iterdir() if (p / "seqinfo.yaml").exists())


# ---------- 输出 ----------

def edge_record(e: Edge, trs: list[Tracklet], fps: float) -> dict:
    """§10.4 的一条边记录。"""
    f = lambda v: None if v is None or (isinstance(v, float) and not np.isfinite(v)) else round(float(v), 4)
    return {
        "src": trs[e.i].tid, "dst": trs[e.j].tid,
        "gap_frames": e.gap, "gap_sec": round(e.tau, 3),
        "cost": f(e.C), "prob": f(e.P),
        "motion": {"e_f": f(e.e_f), "e_b": f(e.e_b), "C_m": f(e.C_m), "Q_m": f(e.Q_m), "rho": f(e.rho),
                   "d2": f(e.d2), "trace": f(e.trace)},
        "appearance": {"d_set": f(e.d_set), "d_proto": f(e.d_proto), "C_a": f(e.C_a), "Q_a": f(e.Q_a),
                       "rep_frames_src": [int(v) for v in e.reps_i], "rep_frames_dst": [int(v) for v in e.reps_j]},
        "scale": {"C_s": f(e.C_s), "Q_s": f(e.Q_s)},
        "gap": {"C_t": f(e.C_t), "g": f(e.g)},
        "sufficiency": f(e.E),
        "competition": {"mnn": bool(e.mnn), "A": f(e.A),
                        "competitors": [{"src": trs[a].tid, "dst": trs[b].tid, "cost": f(c)}
                                        for a, b, c in e.competitors]},
        "flags": sorted(e.flags),
        "solver": int(e.x),
        "decision": e.decision,
        "label": int(e.label),
    }


def write_edges_jsonl(path: Path, edges: list[Edge], trs: list[Tracklet], fps: float) -> None:
    with open(path, "w") as f:
        for e in edges:
            f.write(json.dumps(edge_record(e, trs, fps), ensure_ascii=False) + "\n")


def write_merged(path: Path, rows: np.ndarray) -> None:
    """rows: (n, 8) frame, id, cx, cy, w, h, score, is_interp → MOT 风格 txt。"""
    tl = cxcywh_to_tlwh(rows[:, 2:6])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        for r, b in zip(rows, tl):
            w.writerow([int(r[0]), int(r[1]), *[f"{v:.2f}" for v in b], f"{r[6]:.3f}", int(r[7])])
