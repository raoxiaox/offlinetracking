"""实验输出目录：outputs/<exp>/<timestamp>/，保存生效配置、git hash 与日志（AGENTS.md §4.1）。"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path

import numpy as np

from .config import save_config


def git_hash() -> str:
    try:
        run = lambda *a: subprocess.check_output(["git", *a], stderr=subprocess.DEVNULL, text=True).strip()
        h, dirty = run("rev-parse", "--short", "HEAD"), run("status", "--porcelain")
        return h + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def make_run_dir(cfg: dict, root: str = "outputs", name: str | None = None) -> Path:
    d = Path(root) / (name or cfg.get("exp_name", "exp")) / time.strftime("%Y%m%d-%H%M%S")
    d.mkdir(parents=True, exist_ok=True)
    save_config(cfg, d / "config.yaml")
    (d / "git.txt").write_text(git_hash() + "\n")
    handler = logging.FileHandler(d / "log.txt")
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)
    return d


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, np.integer):
        return int(o)
    return o


def write_json(path: Path, obj) -> None:
    Path(path).write_text(json.dumps(_clean(obj), ensure_ascii=False, indent=2))
