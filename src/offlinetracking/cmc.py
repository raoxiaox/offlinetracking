"""模块 0.1：帧间相机运动补偿（§3.1）。

H[t] 是把帧 t 的像素坐标映到帧 t+1 的 3×3 仿射矩阵（最后一行 0,0,1），rho[t] 是它的可靠度。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .structs import Sequence


def estimate_pair(img0: np.ndarray, img1: np.ndarray, boxes0: np.ndarray, cfg: dict):
    """估计 img0→img1 的部分仿射（旋转+缩放+平移）。返回 (H 3×3, rho, 内点残差像素)。"""
    g0 = img0 if img0.ndim == 2 else cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    g1 = img1 if img1.ndim == 2 else cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    h, w = g0.shape
    mask = np.full((h, w), 255, np.uint8)
    ex = cfg.get("box_expand", 0.1)
    for cx, cy, bw, bh in np.asarray(boxes0).reshape(-1, 4):
        x1, y1 = int(cx - bw * (0.5 + ex)), int(cy - bh * (0.5 + ex))
        x2, y2 = int(cx + bw * (0.5 + ex)), int(cy + bh * (0.5 + ex))
        mask[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)] = 0
    pts = cv2.goodFeaturesToTrack(g0, cfg.get("max_corners", 400), cfg.get("quality_level", 0.01),
                                  cfg.get("min_distance", 7), mask=mask)
    eye = np.eye(3)
    if pts is None or len(pts) < cfg.get("n_min", 20):
        return eye, 0.0, np.nan
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, pts, None, winSize=(21, 21), maxLevel=3)
    ok = st.ravel() == 1
    p0, p1 = pts[ok].reshape(-1, 2), nxt[ok].reshape(-1, 2)
    if len(p0) < cfg.get("n_min", 20):
        return eye, 0.0, np.nan
    M, inl = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC,
                                         ransacReprojThreshold=cfg.get("ransac_thresh", 3.0))
    if M is None:
        return eye, 0.0, np.nan
    inl = inl.ravel().astype(bool)
    n_in = int(inl.sum())
    if n_in < cfg.get("n_min", 20):
        return eye, 0.0, np.nan
    H = np.vstack([M, [0, 0, 1]])
    hull = cv2.convexHull(p0[inl].astype(np.float32))
    area = cv2.contourArea(hull)
    rho = (n_in / len(pts)) * min(1.0, area / (cfg.get("hull_alpha", 0.3) * w * h))
    res = np.linalg.norm(p0[inl] @ M[:, :2].T + M[:, 2] - p1[inl], axis=1)
    return H, float(rho), float(np.sqrt(np.mean(res**2)))


def boxes_at_frames(seq: Sequence) -> dict[int, np.ndarray]:
    out: dict[int, list] = {}
    for tr in seq.tracklets:
        for f, b in zip(tr.frames, tr.boxes):
            out.setdefault(int(f), []).append(b)
    return {k: np.array(v) for k, v in out.items()}


def estimate_sequence(seq: Sequence, frames, cfg: dict):
    """frames：可按帧号下标取图的序列（list 或 FrameReader）。"""
    T = seq.n_frames
    H = np.tile(np.eye(3), (T + 2, 1, 1))
    rho = np.ones(T + 2)
    sigma = np.zeros(T + 2)
    boxes = boxes_at_frames(seq)
    prev = frames[1]
    for t in range(1, T):
        cur = frames[t + 1]
        H[t], rho[t], sigma[t] = estimate_pair(prev, cur, boxes.get(t, np.zeros((0, 4))), cfg)
        prev = cur
    sigma = np.where(np.isfinite(sigma), sigma, np.nanmax(np.where(np.isfinite(sigma), sigma, 0)) + 1)
    return H, rho, sigma


class FrameReader:
    def __init__(self, frames_dir: str, gray: bool = True):
        self.d = Path(frames_dir)
        self.gray = gray

    def __getitem__(self, t: int):
        flag = cv2.IMREAD_GRAYSCALE if self.gray else cv2.IMREAD_COLOR
        return cv2.imread(str(self.d / f"{t:06d}.jpg"), flag)


class CMCTrack:
    """累积变换 H(a→b) 与区间可靠度 rho(a→b)。"""

    def __init__(self, H: np.ndarray, rho: np.ndarray, sigma: np.ndarray | None = None):
        n = len(H)
        self.H = H
        self.rho_t = rho
        self.sigma_t = sigma if sigma is not None else np.zeros(n)
        P = np.zeros((n + 1, 3, 3))
        P[0] = P[1] = np.eye(3)
        for t in range(1, n):
            P[t + 1] = H[t] @ P[t]
        self.P = P
        self.Pinv = np.linalg.inv(P)
        self.n = n

    @classmethod
    def identity(cls, n_frames: int):
        return cls(np.tile(np.eye(3), (n_frames + 2, 1, 1)), np.ones(n_frames + 2))

    def _c(self, t):
        return int(min(max(t, 1), self.n))

    def Hab(self, a: int, b: int) -> np.ndarray:
        """帧 a 坐标 → 帧 b 坐标（a > b 时为逆变换）。"""
        if a == b:
            return np.eye(3)
        return self.P[self._c(b)] @ self.Pinv[self._c(a)]

    def step(self, t: int) -> np.ndarray:
        return self.H[self._c(t)] if 1 <= t < self.n else np.eye(3)

    def rho(self, a: int, b: int) -> float:
        lo, hi = sorted((a, b))
        if lo == hi:
            return 1.0
        return float(self.rho_t[self._c(lo):self._c(hi)].min())

    def sigma(self, a: int, b: int) -> float:
        lo, hi = sorted((a, b))
        if lo == hi:
            return 0.0
        return float(np.mean(self.sigma_t[self._c(lo):self._c(hi)] ** 2) ** 0.5)


def apply(H: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    return p @ H[:2, :2].T + H[:2, 2]


def get_cmc(seq: Sequence, cfg: dict) -> CMCTrack:
    c = cfg["cmc"]
    if not c.get("enabled", True) or c.get("source") == "none":
        return CMCTrack.identity(seq.n_frames)
    src = c.get("source", "auto")
    if seq.cmc_H is not None and src in ("auto", "file"):
        return CMCTrack(seq.cmc_H, seq.cmc_rho, seq.cmc_sigma)
    if seq.frames_dir and src in ("auto", "frames"):
        H, rho, sig = estimate_sequence(seq, FrameReader(seq.frames_dir), c)
        seq.cmc_H, seq.cmc_rho, seq.cmc_sigma = H, rho, sig
        return CMCTrack(H, rho, sig)
    seq.meta["cmc_missing"] = True
    return CMCTrack.identity(seq.n_frames)
