"""对一个划分（或任意序列目录）跑 STWO，输出边记录、合并结果、指标与审查报告。

    python scripts/run_stwo.py --config configs/calibrated.yaml --data data/synth --split test
输出到 outputs/<exp>/<timestamp>/：config.yaml、git.txt、log.txt、metrics.json、report.html、
<seq>/edges.jsonl、<seq>/merged.txt（frame,id,x,y,w,h,score,is_interp）。
"""

from __future__ import annotations

import argparse
import logging

from offlinetracking.config import load_config
from offlinetracking.eval.metrics import sequence_counts, summarize
from offlinetracking.pipeline import run_stwo
from offlinetracking.runs import make_run_dir, write_json
from offlinetracking.seqio import list_sequences, load_sequence, write_edges_jsonl, write_merged
from offlinetracking.viz.report import build_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/calibrated.yaml")
    ap.add_argument("--data", default="data/synth")
    ap.add_argument("--split", default="test", help="splits.yaml 中的划分名；为空则跑 data 下全部序列")
    ap.add_argument("--max-seqs", type=int, default=0, help="报告里最多放几段序列（0 = 全部）")
    ap.add_argument("--max-cards", type=int, default=25)
    ap.add_argument("--no-report", action="store_true")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, overrides=args.overrides)
    run = make_run_dir(cfg)
    log = logging.getLogger("run")
    results, counts = [], []
    for p in list_sequences(args.data, args.split or None):
        seq = load_sequence(p)
        res = run_stwo(seq, cfg)
        if seq.gt is not None:
            counts.append(sequence_counts(res, cfg))
        d = run / seq.name
        d.mkdir()
        write_edges_jsonl(d / "edges.jsonl", res.edges, seq.tracklets, seq.fps)
        write_merged(d / "merged.txt", res.rows)
        n_auto = sum(e.decision == "auto" for e in res.edges)
        n_rev = sum(e.decision == "review" for e in res.edges)
        log.info("%s: %d 条轨迹 → %d 个 ID，自动 %d，复核 %d", seq.name, len(seq.tracklets), res.gid.max(), n_auto, n_rev)
        print(f"{seq.name}: {len(seq.tracklets)} tracklets → {res.gid.max()} ids, auto {n_auto}, review {n_rev}")
        results.append(res)
    metrics = summarize(counts, cfg) if counts else None
    if metrics:
        write_json(run / "metrics.json", metrics)
        for k in ("auto_precision", "auto_coverage", "link_precision", "link_recall", "gate_recall", "review_per_min"):
            print(f"  {k}: {metrics[k]:.3f}")
        for k in ("mot_raw", "mot_auto", "mot_link"):
            print(f"  {k}: IDF1 {metrics[k]['IDF1']:.3f}, IDsw {metrics[k]['IDsw']}")
    if not args.no_report:
        shown = results[: args.max_seqs] if args.max_seqs else results
        path = build_report(shown, metrics, cfg, run / "report.html", max_cards=args.max_cards)
        print(f"报告：{path}")
    print(f"输出目录：{run}")


if __name__ == "__main__":
    main()
