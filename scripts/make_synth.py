"""生成合成序列：帧图像 → 真实跑一遍 CMC → 保存为标准序列目录，并按视频划分 calib / test。

    python scripts/make_synth.py --out data/synth --n-seq 12
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path

import cv2
import yaml

from offlinetracking.cmc import estimate_sequence
from offlinetracking.config import load_config
from offlinetracking.seqio import save_sequence
from offlinetracking.synth import SynthParams, generate_sequence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--n-seq", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calib-frac", type=float, default=0.5)
    ap.add_argument("--synth-config", default="configs/synth.yaml")
    ap.add_argument("--no-frames", action="store_true", help="不保存帧图像（CMC 仍会计算并保存）")
    args = ap.parse_args()

    fields = {f.name for f in dataclasses.fields(SynthParams)}
    sp = yaml.safe_load(Path(args.synth_config).read_text()) or {}
    params = SynthParams(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in sp.items() if k in fields})
    cfg = load_config()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = []
    for k in range(args.n_seq):
        t0 = time.time()
        name = f"synth_{k:03d}"
        seq, frames = generate_sequence(name, args.seed * 1000 + k, params)
        seq.cmc_H, seq.cmc_rho, seq.cmc_sigma = estimate_sequence(seq, frames, cfg["cmc"])
        save_sequence(seq, out / name)
        if not args.no_frames:
            fd = out / name / "frames"
            fd.mkdir(exist_ok=True)
            for t in range(1, len(frames)):
                cv2.imwrite(str(fd / f"{t:06d}.jpg"), frames[t], [cv2.IMWRITE_JPEG_QUALITY, 85])
        names.append(name)
        print(f"{name}: {len(seq.tracklets)} tracklets, {int(seq.gt[:, 1].max()) if len(seq.gt) else 0} ids, "
              f"mean rho {seq.cmc_rho[1:seq.n_frames].mean():.2f}, {time.time() - t0:.1f}s")
    n_cal = int(round(len(names) * args.calib_frac))
    (out / "splits.yaml").write_text(yaml.safe_dump({"calib": names[:n_cal], "test": names[n_cal:]}))


if __name__ == "__main__":
    main()
