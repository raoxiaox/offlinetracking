"""合成数据：车载前视相机 + 地面上的车辆。

生成内容：
- 自车位姿（前进、轻微横摆）与相机抖动（俯仰/偏航 AR(1) 噪声）；
- 同向、对向、路边停放车辆的 3D 框，投影为图像框（裁剪到画面，记录截断）；
- 按深度栅格化计算可见度（遮挡）；
- 检测：漏检、遮挡丢失、成段失效（0.2–3 s），框噪声；
- 上游短轨迹：在缺口处断开（小缺口按概率被上游桥接），外加随机断开与少量误检轨迹；
- ReID 特征：每个 ID 一个原型，部分 ID 共享“车型”基向量（相似车），观测噪声随可见度与尺寸变化；
- 帧图像：带纹理的地面（单应投影）+ 远景建筑全景 + 车辆色块，用于真实地跑 CMC。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .seqio import CLASSES
from .structs import Sequence, Tracklet

DIMS = {"car": (4.5, 1.8, 1.5), "van": (5.0, 2.0, 2.1), "truck": (8.0, 2.5, 3.2)}


@dataclass
class SynthParams:
    fps: float = 10.0
    n_frames: int = 240
    img_w: int = 640
    img_h: int = 360
    focal: float = 420.0
    cam_height: float = 1.5
    pitch: float = 0.03
    shake_pitch: float = 0.004
    shake_yaw: float = 0.003
    ego_speed: float = 11.0
    n_same: int = 12
    n_oncoming: int = 5
    n_parked: int = 6
    n_fp: int = 2
    max_depth: float = 110.0
    vis_detect: float = 0.35
    p_miss: float = 0.03
    p_dropout: float = 0.006       # 每帧开始一段失效的概率
    dropout_sec: tuple = (0.2, 3.0)
    p_break: float = 0.006         # 无缺口的随机断开
    p_bridge: float = 0.6          # 1–2 帧小缺口被上游桥接的概率
    feat_dim: int = 64
    model_share: float = 0.45      # 共享车型的 ID 比例
    model_spread: float = 0.45
    feat_noise: tuple = (0.18, 2.0)  # 观测噪声：σ = a + b·(1 − q)²
    occluder_mix: float = 1.5        # 被遮挡时特征混入遮挡者原型的强度 × (1 − vis)
    box_noise: float = 0.02
    render: bool = True
    extra: dict = field(default_factory=dict)


def _ar1(rng, n, sigma, a=0.85):
    x = np.zeros(n)
    for k in range(1, n):
        x[k] = a * x[k - 1] + rng.normal(0, sigma * math.sqrt(1 - a * a))
    return x


class Camera:
    def __init__(self, p: SynthParams):
        self.p = p
        self.K = np.array([[p.focal, 0, p.img_w / 2], [0, p.focal, p.img_h / 2], [0, 0, 1.0]])
        self.G = np.array([[0, -1.0, 0], [0, 0, p.cam_height], [1.0, 0, 0]])  # 自车地面 (x,y,1) → 相机

    def rot(self, pitch, yaw_j):
        cp, sp = math.cos(pitch), math.sin(pitch)
        cj, sj = math.cos(yaw_j), math.sin(yaw_j)
        Rp = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        Ry = np.array([[cj, 0, sj], [0, 1, 0], [-sj, 0, cj]])
        return Rp @ Ry

    def ground_H(self, pitch, yaw_j):
        """自车系地面 (x, y, 1) → 像素 的单应。"""
        return self.K @ self.rot(pitch, yaw_j) @ self.G

    def project(self, pts_ego, pitch, yaw_j):
        """pts_ego (N,3) 自车系 (x 前, y 左, z 上) → (u, v, depth)。"""
        c = np.stack([-pts_ego[:, 1], self.p.cam_height - pts_ego[:, 2], pts_ego[:, 0]], 1)
        c = c @ self.rot(pitch, yaw_j).T
        z = c[:, 2]
        uv = (c @ self.K.T)[:, :2] / np.where(np.abs(z) < 1e-6, 1e-6, z)[:, None]
        return uv, z


def world_to_ego(P, ego):
    x, y, yaw = ego
    c, s = math.cos(yaw), math.sin(yaw)
    d = P[..., :2] - np.array([x, y])
    out = np.empty_like(P)
    out[..., 0] = c * d[..., 0] + s * d[..., 1]
    out[..., 1] = -s * d[..., 0] + c * d[..., 1]
    if P.shape[-1] > 2:
        out[..., 2:] = P[..., 2:]
    return out


def _make_objects(rng, p: SynthParams, ego_x_end):
    objs = []
    T = p.n_frames
    t = np.arange(T + 1) / p.fps

    def lane_path(y0, y1, t_change, dur=3.0):
        s = np.clip((t - t_change) / dur, 0, 1)
        s = s * s * (3 - 2 * s)
        return y0 + (y1 - y0) * s

    for _ in range(p.n_same):
        cls = rng.choice(["car", "car", "car", "van", "truck"])
        lane = rng.choice([-3.5, 0.0, 3.5])
        v = p.ego_speed + rng.normal(0, 2.5)
        x0 = rng.uniform(-25, 200)
        X = x0 + v * t + 0.5 * rng.normal(0, 0.3) * t**2
        Y = np.full_like(t, lane)
        if rng.random() < 0.35:
            Y = lane_path(lane, lane + rng.choice([-3.5, 3.5]), rng.uniform(0, t[-1]))
        objs.append((cls, X, Y))
    for _ in range(p.n_oncoming):
        cls = rng.choice(["car", "car", "van", "truck"])
        lane = rng.choice([7.0, 10.5])
        v = -rng.uniform(7, 13)
        x0 = rng.uniform(60, ego_x_end + 120)
        objs.append((cls, x0 + v * t, np.full_like(t, lane)))
    for _ in range(p.n_parked):
        cls = rng.choice(["car", "car", "van"])
        x0 = rng.uniform(10, ego_x_end + 60)
        objs.append((cls, np.full_like(t, x0), np.full_like(t, -7.5 + rng.normal(0, 0.3))))
    return objs


def _box_corners(X, Y, heading, dims):
    L, W, H = dims
    dx = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * L / 2
    dy = np.array([1, -1, -1, 1, 1, -1, -1, 1]) * W / 2
    dz = np.array([0, 0, 0, 0, 1, 1, 1, 1]) * H
    c, s = math.cos(heading), math.sin(heading)
    return np.stack([X + c * dx - s * dy, Y + s * dx + c * dy, dz], 1)


def generate_sequence(name: str, seed: int, p: SynthParams | None = None):
    """返回 (Sequence, frames 或 None)。frames[t] 为第 t 帧 BGR 图（下标 0 不用）。"""
    p = p or SynthParams()
    rng = np.random.default_rng(seed)
    T, F = p.n_frames, p.fps
    cam = Camera(p)
    t = np.arange(T + 1) / F

    # 自车
    speed = p.ego_speed * (1 + 0.15 * np.sin(2 * np.pi * t / rng.uniform(15, 30) + rng.uniform(0, 6)))
    lat = 0.4 * np.sin(2 * np.pi * t / rng.uniform(6, 12))
    ex = np.concatenate([[0], np.cumsum(speed[1:] / F)])
    yaw = np.arctan2(np.gradient(lat, ex + 1e-9 * np.arange(T + 1)), 1.0)
    ego = np.stack([ex, lat, yaw], 1)
    pitch = p.pitch + _ar1(rng, T + 1, p.shake_pitch)
    yawj = _ar1(rng, T + 1, p.shake_yaw)

    objs = _make_objects(rng, p, ex[-1])
    n_obj = len(objs)
    W, H = p.img_w, p.img_h

    # ---- 投影与可见度 ----
    rec = {}  # (obj, frame) -> dict
    for fr in range(1, T + 1):
        items = []
        for oid, (cls, X, Y) in enumerate(objs):
            k0, k1 = max(fr - 1, 0), min(fr + 1, T)
            ddx, ddy = X[k1] - X[k0], Y[k1] - Y[k0]
            heading = math.atan2(ddy, ddx) if ddx * ddx + ddy * ddy > 1e-6 else 0.0
            corners = world_to_ego(_box_corners(X[fr], Y[fr], heading, DIMS[cls]), ego[fr])
            if corners[:, 0].min() < 2.0:
                continue
            uv, z = cam.project(corners, pitch[fr], yawj[fr])
            if z.min() < 1.0 or z.mean() > p.max_depth:
                continue
            x1, y1 = uv.min(0)
            x2, y2 = uv.max(0)
            cx1, cy1, cx2, cy2 = max(x1, 0), max(y1, 0), min(x2, W - 1), min(y2, H - 1)
            if cx2 - cx1 < 3 or cy2 - cy1 < 3:
                continue
            full = (x2 - x1) * (y2 - y1)
            items.append(dict(oid=oid, cls=cls, depth=float(z.mean()), box=(cx1, cy1, cx2, cy2),
                              full=(x1, y1, x2, y2), trunc=(cx2 - cx1) * (cy2 - cy1) / full))
        # 栅格化可见度：由远到近绘制
        s = 4
        canvas = -np.ones((H // s + 1, W // s + 1), int)
        for it in sorted(items, key=lambda d: -d["depth"]):
            a, b, c, d = (int(v / s) for v in it["box"])
            canvas[b:d + 1, a:c + 1] = it["oid"]
        for it in items:
            a, b, c, d = (int(v / s) for v in it["box"])
            reg = canvas[b:d + 1, a:c + 1]
            it["vis"] = float((reg == it["oid"]).mean()) if reg.size else 0.0
            other = reg[(reg != it["oid"]) & (reg >= 0)]
            it["occ_by"] = int(np.bincount(other).argmax()) if other.size else -1
            rec[(it["oid"], fr)] = it

    # ---- 真值 ----
    gt_rows = []
    for (oid, fr), it in rec.items():
        x1, y1, x2, y2 = it["box"]
        gt_rows.append([fr, oid + 1, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, it["vis"],
                        CLASSES.index(it["cls"])])
    gt = np.array(sorted(gt_rows), float) if gt_rows else np.zeros((0, 8))

    # ---- 特征原型 ----
    D = p.feat_dim
    def unit(v):
        return v / np.linalg.norm(v, axis=-1, keepdims=True)
    n_models = max(1, int(round(n_obj * (1 - p.model_share))))
    bases = unit(rng.normal(size=(n_models, D)))
    model_of = rng.integers(0, n_models, n_obj)
    protos = unit(bases[model_of] + p.model_spread * unit(rng.normal(size=(n_obj, D))))
    drift = unit(rng.normal(size=(n_obj, D)))

    # ---- 检测与上游短轨迹 ----
    tracklets: list[Tracklet] = []
    drop_len = (int(p.dropout_sec[0] * F), int(p.dropout_sec[1] * F))
    for oid in range(n_obj):
        frames = sorted(fr for (o, fr) in rec if o == oid)
        dets = []
        dropout_until = -1
        for fr in frames:
            it = rec[(oid, fr)]
            if fr <= dropout_until:
                continue
            if rng.random() < p.p_dropout:
                dropout_until = fr + int(rng.integers(drop_len[0], drop_len[1] + 1))
                continue
            size = math.sqrt((it["box"][2] - it["box"][0]) * (it["box"][3] - it["box"][1]))
            if it["vis"] < p.vis_detect or size < 6 or rng.random() < p.p_miss:
                continue
            x1, y1, x2, y2 = it["box"]
            w, h = x2 - x1, y2 - y1
            sig = p.box_noise * size * (1 + 1.5 * (1 - it["vis"])) + 0.3
            box = np.array([(x1 + x2) / 2 + rng.normal(0, sig), (y1 + y2) / 2 + rng.normal(0, sig),
                            w * math.exp(rng.normal(0, 0.03)), h * math.exp(rng.normal(0, 0.03))])
            qt = it["vis"] * min(1.0, size / 40) * min(1.0, it["trunc"] + 0.2)
            score = float(np.clip(0.45 + 0.5 * qt + rng.normal(0, 0.05), 0.05, 0.99))
            sigma_f = p.feat_noise[0] + p.feat_noise[1] * (1 - qt) ** 2
            mix = p.occluder_mix * (1 - it["vis"]) * (protos[it["occ_by"]] if it["occ_by"] >= 0 else 0)
            f = unit(protos[oid] + 0.25 * (fr / T) * drift[oid] + mix + sigma_f * unit(rng.normal(size=D)))
            dets.append((fr, box, score, f))
        # 断开成短轨迹
        cur: list = []
        for d in dets:
            if cur:
                g = d[0] - cur[-1][0] - 1
                brk = (g == 0 and rng.random() < p.p_break) or (1 <= g <= 2 and rng.random() > p.p_bridge) or g > 2
                if brk:
                    tracklets.append(_mk(cur, objs[oid][0], oid + 1))
                    cur = []
            cur.append(d)
        if cur:
            tracklets.append(_mk(cur, objs[oid][0], oid + 1))
    tracklets = [tr for tr in tracklets if len(tr) >= 2]

    # 误检短轨迹
    for _ in range(rng.poisson(p.n_fp)):
        L = int(rng.integers(3, 15))
        f0 = int(rng.integers(1, T - L))
        cx, cy, s = rng.uniform(80, W - 80), rng.uniform(H * 0.55, H * 0.85), rng.uniform(15, 40)
        fr = np.arange(f0, f0 + L)
        boxes = np.column_stack([cx + rng.normal(0, 2, L), cy + rng.normal(0, 2, L),
                                 np.full(L, s * 1.3), np.full(L, s)])
        tracklets.append(Tracklet(0, "car", fr, boxes, rng.uniform(0.25, 0.45, L),
                                  unit(rng.normal(size=(L, D)) + unit(rng.normal(size=D))), gt_id=-1))

    tracklets.sort(key=lambda tr: (tr.t_s, tr.t_e))
    for k, tr in enumerate(tracklets):
        tr.tid = k + 1

    camera = {"Hg": cam.ground_H(p.pitch, 0.0), "K": cam.K.tolist(), "height": p.cam_height, "pitch": p.pitch}
    seq = Sequence(name=name, fps=F, img_w=W, img_h=H, tracklets=tracklets, gt=gt, camera=camera, ego=ego,
                   n_frames=T, meta={"synthetic": True, "seed": seed})
    frames = render_frames(seq, objs, rec, pitch, yawj, cam, rng, p) if p.render else None
    return seq, frames


def _mk(dets, cls, gt_id):
    return Tracklet(0, cls, np.array([d[0] for d in dets]), np.array([d[1] for d in dets]),
                    np.array([d[2] for d in dets]), np.array([d[3] for d in dets], np.float32), gt_id=gt_id)


# ---------------- 渲染 ----------------

def _noise_tex(rng, h, w, scales=(2, 8, 32)):
    tex = np.zeros((h, w), np.float32)
    for s in scales:
        small = rng.random((h // s + 2, w // s + 2)).astype(np.float32)
        tex += cv2.resize(small, (w + 2 * s, h + 2 * s), interpolation=cv2.INTER_CUBIC)[s:s + h, s:s + w] / len(scales)
    return tex


def _ground_texture(rng, x_min, x_max, res=8.0):
    w, h = int((x_max - x_min) * res), int(60 * res)
    tex = 70 + 60 * _noise_tex(rng, h, w)
    y0 = -30.0
    for yl in (-5.25, -1.75, 1.75, 5.25, 8.75):
        r = int((yl - y0) * res)
        for xs in np.arange(x_min, x_max, 9.0):
            c = int((xs - x_min) * res)
            cv2.rectangle(tex, (c, r - 1), (c + int(3 * res), r + 1), 230, -1)
    for yl in (-9.0, 12.25):
        r = int((yl - y0) * res)
        cv2.rectangle(tex, (0, r - 1), (w, r + 1), 220, -1)
    for _ in range(int(w * h / 4000)):
        c, r = int(rng.integers(0, w)), int(rng.integers(0, h))
        cv2.circle(tex, (c, r), int(rng.integers(2, 6)), float(rng.uniform(20, 60)), -1)
    Tex = np.array([[1 / res, 0, x_min], [0, 1 / res, y0], [0, 0, 1]])
    return np.clip(tex, 0, 255).astype(np.uint8), Tex


def _panorama(rng, h, w):
    pano = (150 + 40 * _noise_tex(rng, h, w, (4, 16))).astype(np.float32)
    x = 0
    while x < w:
        bw, bh = int(rng.integers(30, 120)), int(rng.integers(h // 6, h // 2))
        g = float(rng.uniform(40, 140))
        cv2.rectangle(pano, (x, h - bh), (x + bw, h), g, -1)
        for wy in range(h - bh + 6, h - 6, 14):
            for wx in range(x + 5, x + bw - 8, 12):
                cv2.rectangle(pano, (wx, wy), (wx + 5, wy + 7), g + rng.uniform(-35, 60), -1)
        x += bw + int(rng.integers(0, 25))
    return np.clip(pano, 0, 255).astype(np.uint8)


def render_frames(seq, objs, rec, pitch, yawj, cam: Camera, rng, p: SynthParams):
    W, H, f = p.img_w, p.img_h, p.focal
    ego = seq.ego
    tex, Tex = _ground_texture(rng, -40.0, ego[-1, 0] + p.max_depth + 40)
    pano_w = W * 6
    pano = _panorama(rng, H, pano_w)
    colors = rng.integers(40, 230, (len(objs), 3))
    frames = [None]
    for fr in range(1, p.n_frames + 1):
        x, y, yaw = ego[fr]
        c, s = math.cos(yaw), math.sin(yaw)
        T1 = np.array([[c, s, -c * x - s * y], [-s, c, s * x - c * y], [0, 0, 1]])
        M = cam.ground_H(pitch[fr], yawj[fr]) @ T1 @ Tex
        ground = cv2.warpPerspective(tex, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=90)
        # 远景：只随旋转平移
        horizon = H / 2 - f * math.tan(pitch[fr])
        dx = -f * (yaw + yawj[fr]) + pano_w / 2 - W / 2
        # 全景底边对齐地平线
        sky = cv2.warpAffine(pano, np.array([[1, 0, -dx], [0, 1, horizon - H]], np.float32), (W, H),
                             borderMode=cv2.BORDER_REFLECT)
        img = ground.copy()
        hr = int(max(0, min(H, horizon + 3)))
        img[:hr] = sky[:hr]
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        items = [rec[(oid, fr)] for oid in range(len(objs)) if (oid, fr) in rec]
        for it in sorted(items, key=lambda d: -d["depth"]):
            x1, y1, x2, y2 = (int(v) for v in it["full"])
            col = tuple(int(v) for v in colors[it["oid"]])
            cv2.rectangle(img, (x1, y1), (x2, y2), col, -1)
            hh, ww = y2 - y1, x2 - x1
            dark = tuple(v // 3 for v in col)
            cv2.rectangle(img, (x1 + ww // 6, y1 + hh // 8), (x2 - ww // 6, y1 + hh // 2), dark, -1)
            cv2.rectangle(img, (x1 + ww // 10, y2 - hh // 4), (x1 + ww // 4, y2 - hh // 8), (30, 30, 200), -1)
            cv2.rectangle(img, (x2 - ww // 4, y2 - hh // 4), (x2 - ww // 10, y2 - hh // 8), (30, 30, 200), -1)
        img = np.clip(img.astype(np.int16) + rng.normal(0, 3, img.shape).astype(np.int16), 0, 255).astype(np.uint8)
        frames.append(img)
    return frames
