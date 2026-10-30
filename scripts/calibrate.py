"""在标定集上标定全部阈值与权重，写出 calibrated.yaml。

    python scripts/calibrate.py --data data/synth --split calib --out configs/calibrated.yaml [a.b=v ...]
"""

from __future__ import annotations

import argparse
import logging

from offlinetracking.calibrate import calibrate
from offlinetracking.config import load_config, save_config
from offlinetracking.runs import make_run_dir, write_json
from offlinetracking.seqio import list_sequences, load_sequence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth")
    ap.add_argument("--split", default="calib")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="configs/calibrated.yaml")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config, overrides=args.overrides)
    seqs = [load_sequence(p) for p in list_sequences(args.data, args.split)]
    cal, rep = calibrate(cfg, seqs, seed=cfg["seed"])
    save_config(cal, args.out)
    run = make_run_dir(cal, name=f"calibrate_{cfg.get('exp_name', 'exp')}")
    write_json(run / "calibration_report.json", rep)
    print(f"写出 {args.out}；报告 {run / 'calibration_report.json'}")
    for k, v in rep.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
