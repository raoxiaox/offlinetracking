"""消融实验：每个配置先在标定集上标定，再在测试集上报告一次。输出 Markdown 表格与 JSON。

    python scripts/ablate.py --data data/synth [--only A0 A9 A10] [--tau-sweep]
"""

from __future__ import annotations

import argparse
import logging

import yaml

from offlinetracking.calibrate import calibrate
from offlinetracking.config import load_config, save_config
from offlinetracking.eval.metrics import sequence_counts, summarize
from offlinetracking.pipeline import run_stwo
from offlinetracking.runs import make_run_dir, write_json
from offlinetracking.seqio import list_sequences, load_sequence


def resolve(table: dict, name: str) -> dict:
    ent = table[name]
    sets = resolve(table, ent["from"]) if "from" in ent else {}
    return {**sets, **ent.get("set", {})}


def run_one(cfg, data, log):
    calib = [load_sequence(p) for p in list_sequences(data, "calib")]
    cal, rep = calibrate(cfg, calib, seed=cfg["seed"])
    counts = []
    for p in list_sequences(data, "test"):
        r = run_stwo(load_sequence(p), cal)
        counts.append(sequence_counts(r, cal))
    return cal, rep, summarize(counts, cal)


def fmt(v, pct=False):
    if v is None or v != v:
        return "–"
    return f"{100 * v:.1f}" if pct else (f"{v:.1f}" if isinstance(v, float) and v > 1 else
                                         f"{v:.3f}" if isinstance(v, float) else str(v))


def row(name, desc, m):
    ma, ml = m.get("mot_auto", {}), m.get("mot_link", {})
    return (f"| {name} | {desc} | {fmt(m['gate_recall'], True)} | {fmt(m['link_precision'], True)} | "
            f"{fmt(m['link_recall'], True)} | {fmt(m['auto_precision'], True)} | {fmt(m['auto_coverage'], True)} | "
            f"{fmt(m['review_per_min'], False)} | {fmt(ma.get('IDF1'), True)} | {ma.get('IDsw', '–')} | "
            f"{fmt(ml.get('IDF1'), True)} | {fmt(m['ece'])} |")


HEADER = ("| 编号 | 配置 | 门控召回% | 边精度% | 边召回% | 自动档精度% | 自动档覆盖% | 复核/分钟 "
          "| IDF1%（仅自动档） | IDsw（仅自动档） | IDF1%（全部求解链接） | ECE |\n"
          "|---|---|---|---|---|---|---|---|---|---|---|---|")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth")
    ap.add_argument("--ablation", default="configs/ablation.yaml")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--tau-sweep", action="store_true", help="另做 τ_max ∈ {0.5,1,2,3} 秒的扫描（基于 A9）")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    table = yaml.safe_load(open(args.ablation))
    names = args.only or list(table)
    run = make_run_dir(load_config(), name="ablation")
    lines, results = [HEADER], {}
    for name in names:
        cfg = load_config(overrides={**{}, **_nest(resolve(table, name))})
        cfg["exp_name"] = name
        cal, rep, m = run_one(cfg, args.data, None)
        save_config(cal, run / f"{name}.yaml")
        results[name] = {"metrics": m, "calibration": rep}
        lines.append(row(name, table[name]["desc"], m))
        print(lines[-1], flush=True)
    if args.tau_sweep:
        lines += ["", "τ_max 扫描（A9）", HEADER]
        for tm in (0.5, 1.0, 2.0, 3.0):
            cfg = load_config(overrides={**_nest(resolve(table, "A9")), "gating": {"tau_max": tm}})
            _, _, m = run_one(cfg, args.data, None)
            results[f"tau_{tm}"] = {"metrics": m}
            lines.append(row(f"τ={tm}s", "A9", m))
            print(lines[-1], flush=True)
    (run / "ablation.md").write_text("\n".join(lines) + "\n")
    write_json(run / "ablation.json", results)
    print(f"\n结果：{run / 'ablation.md'}")


def _nest(flat: dict) -> dict:
    out: dict = {}
    for k, v in flat.items():
        d = out
        ks = k.split(".")
        for kk in ks[:-1]:
            d = d.setdefault(kk, {})
        d[ks[-1]] = v
    return out


if __name__ == "__main__":
    main()
