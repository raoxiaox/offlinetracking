"""配置：YAML 文件逐层合并 + 命令行 `a.b=value` 覆盖。配置就是嵌套 dict。"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def deep_update(base: dict, upd: dict) -> dict:
    """把 upd 递归合并进 base（原地），返回 base。"""
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def set_by_path(cfg: dict, path: str, value: Any) -> None:
    keys = path.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def parse_overrides(items: list[str] | None) -> dict:
    out: dict = {}
    for it in items or []:
        k, _, v = it.partition("=")
        set_by_path(out, k.strip(), yaml.safe_load(v))
    return out


def load_config(*paths: str | Path | None, overrides: list[str] | dict | None = None) -> dict:
    """先读 default.yaml，再依次合并 paths，最后应用 overrides。"""
    cfg = yaml.safe_load(DEFAULT_PATH.read_text())
    for p in paths:
        if p:
            deep_update(cfg, yaml.safe_load(Path(p).read_text()) or {})
    if isinstance(overrides, dict):
        deep_update(cfg, overrides)
    else:
        deep_update(cfg, parse_overrides(overrides))
    return cfg


def per_class(table: dict, cls: str) -> float:
    """按类别取参数，缺省回落到 default。"""
    return float(table.get(cls, table["default"]))


def save_config(cfg: dict, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
