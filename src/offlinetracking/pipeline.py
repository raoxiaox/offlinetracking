"""串联模块 0–7。分三步，便于标定时复用：prepare → build_edges → link。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .appearance import AppearanceModel
from .association import associate
from .cmc import CMCTrack, get_cmc
from .competition import competition
from .decision import build_output, decide, merge_chains
from .fusion import fuse, gap_cost, scale_evidence
from .gating import build_candidates
from .motion import end_fits, motion_evidence
from .quality import compute_quality
from .structs import Edge, Sequence


@dataclass
class Result:
    seq: Sequence
    edges: list[Edge]
    gid: np.ndarray
    rows: np.ndarray
    cmc: CMCTrack | None = None
    extra: dict = field(default_factory=dict)


def prepare(seq: Sequence, cfg: dict) -> CMCTrack:
    """模块 0：质量与 CMC。"""
    compute_quality(seq, cfg)
    return get_cmc(seq, cfg)


def build_edges(seq: Sequence, cmc: CMCTrack, cfg: dict, loose: bool = False) -> list[Edge]:
    """模块 1–4 的证据部分：门控、运动、外观、尺度、缺口。"""
    edges = build_candidates(seq.tracklets, cmc, seq.fps, cfg, loose=loose)
    fits = end_fits(seq, cmc, cfg)
    app = AppearanceModel(seq, cfg)
    for e in edges:
        motion_evidence(e, seq, fits, cmc, cfg)
        app.evidence(e)
        scale_evidence(e, seq, cfg)
        gap_cost(e, cfg)
    return edges


def link(seq: Sequence, edges: list[Edge], cfg: dict) -> None:
    """模块 4 融合 → 5 竞争 → 6 关联 → 7 决策。"""
    fuse(edges, cfg)
    C0 = cfg["association"]["theta_link"]
    competition(edges, C0, cfg["competition"]["delta_m"])
    associate(seq.tracklets, edges, cfg)
    decide(edges, cfg)


def run_stwo(seq: Sequence, cfg: dict, cmc: CMCTrack | None = None) -> Result:
    cmc = cmc or prepare(seq, cfg)
    n = len(seq.tracklets)
    if not cfg["pipeline"].get("link", True):
        gid = np.arange(1, n + 1)
        return Result(seq, [], gid, build_output(seq, [], gid, cfg), cmc)
    edges = build_edges(seq, cmc, cfg)
    link(seq, edges, cfg)
    gid = merge_chains(n, [(e.i, e.j) for e in edges if e.decision == "auto"])
    return Result(seq, edges, gid, build_output(seq, edges, gid, cfg), cmc)
