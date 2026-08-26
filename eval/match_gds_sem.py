#!/usr/bin/env python3
"""
Per-cell GDS -> SEM matcher.

Overlays each solid rectangle (cell) from a GDS layout on top of its
counterpart in the generated SEM image and scores how well the SEM
reproduced it:

    GREEN cell  -> score >= threshold (default 90%): the SEM has material
                   where the GDS says it should be
    RED cell    -> score <  threshold: the cell is missing, shifted, or
                   badly deformed in the SEM

GDS images are upscaled to the SEM's dimensions automatically (465 -> 512
with nearest-neighbour, so rectangle edges stay crisp).

A "cell" is one 8-connected white region in the GDS image. Rectangles that
touch each other in the layout form a single cell — raise --min-area to
drop specks, and note that a red verdict on a large merged cell means some
part of that group is wrong.

Scoring (--metric):
  coverage  (default)  |cell ∩ SEM_material| / |cell|
                       "how much of this rectangle is present in the SEM"
  iou                  |cell ∩ SEM_local| / |cell ∪ SEM_local| over the
                       cell's neighbourhood — also penalises SEM material
                       that bleeds outside the rectangle

--tolerance dilates the SEM material by N px before scoring, forgiving
sub-pixel edge placement and resampling softness (default 1).

Inputs are two directories; filenames are matched by stem, ignoring
ComfyUI's `_00001_` counter suffix.

Usage:
  python match_gds_sem.py --gds-dir gds_2_sem/A/val --sem-dir gds_2_sem/C/val \
      --out matches/gds_vs_gen
  python match_gds_sem.py --root gds_2_sem --split val --threshold 0.85 --annotate
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from matchlib import (GREEN, RED, binarize, caption, connected_components,
                      dilate, gray_to_rgb, load_gray, matched_stems, outline,
                      resize_to, stem_map, tint, write_csv)


def score_cell(cell: np.ndarray, sem_tol: np.ndarray, sem_raw: np.ndarray,
               metric: str) -> float:
    area = int(cell.sum())
    if area == 0:
        return 0.0
    hit = int((cell & sem_tol).sum())
    if metric == "coverage":
        return hit / area
    union = int((cell | (sem_raw & dilate(cell, 2))).sum())
    return hit / max(union, 1)


def match_one(gds_p: Path, sem_p: Path, out_dir: Path, stem: str,
              args) -> dict:
    gds = load_gray(gds_p)
    sem = load_gray(sem_p)
    # everything works at the SEM (generated) resolution
    gds = resize_to(gds, (sem.shape[1], sem.shape[0]), nearest=True)

    gds_mask = binarize(gds, "fixed")
    sem_mask = binarize(sem, "otsu")
    sem_tol = dilate(sem_mask, args.tolerance)

    labels, n, boxes = connected_components(gds_mask)
    base = gray_to_rgb(sem)

    cells = passed = 0
    scores = []
    fails = []
    for i, (x0, y0, x1, y1) in enumerate(boxes, start=1):
        sub = labels[y0:y1, x0:x1] == i
        if int(sub.sum()) < args.min_area:
            continue
        s = score_cell(sub, sem_tol[y0:y1, x0:x1], sem_mask[y0:y1, x0:x1],
                       args.metric)
        ok = s >= args.threshold
        color = GREEN if ok else RED
        cells += 1
        passed += ok
        scores.append(s)
        if not ok:
            fails.append((x0, y0, s))

        view = base[y0:y1, x0:x1]
        tint(view, sub, color, args.alpha)                  # translucent fill
        tint(view, outline(sub, args.outline), color, 1.0)  # solid edge

    rate = passed / cells if cells else 0.0
    head = (f"{stem}   cells {passed}/{cells} matched ({rate:.1%})"
            f"   thr {args.threshold:.0%}   metric {args.metric}")
    img = caption(base, [(head, GREEN if rate >= args.threshold else RED),
                         ("green = cell reproduced in SEM    "
                          "red = below threshold", (200, 200, 200))])

    if args.annotate and fails:
        from PIL import ImageDraw
        d = ImageDraw.Draw(img)
        bar = img.height - base.shape[0]
        for x0, y0, s in fails:
            d.text((x0 + 1, y0 + bar - 10), f"{s:.0%}", fill=RED)

    out_dir.mkdir(parents=True, exist_ok=True)
    img.save(out_dir / f"match_{stem}.png")
    return {"image": stem, "cells": cells, "matched": passed,
            "match_rate": round(rate, 4),
            "mean_cell_score": round(float(np.mean(scores)) if scores else 0.0, 4),
            "min_cell_score": round(float(np.min(scores)) if scores else 0.0, 4)}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("inputs")
    src.add_argument("--gds-dir", type=Path, help="e.g. gds_2_sem/A/val")
    src.add_argument("--sem-dir", type=Path, help="e.g. gds_2_sem/C/val")
    src.add_argument("--root", type=Path, help="gds_2_sem (use with --split)")
    src.add_argument("--split", choices=["train", "val", "both"], default="val")
    ap.add_argument("--out", type=Path, default=Path("matches/gds_vs_gen"),
                    help="output directory for the overlay images")
    ap.add_argument("--threshold", type=float, default=0.90,
                    help="pass mark per cell, 0-1 (default 0.90)")
    ap.add_argument("--metric", choices=["coverage", "iou"], default="coverage")
    ap.add_argument("--tolerance", type=int, default=1,
                    help="px of SEM dilation before scoring (default 1)")
    ap.add_argument("--min-area", type=int, default=20,
                    help="ignore GDS blobs smaller than this many px")
    ap.add_argument("--alpha", type=float, default=0.40,
                    help="opacity of the cell fill (default 0.40)")
    ap.add_argument("--outline", type=int, default=1, help="edge width in px")
    ap.add_argument("--annotate", action="store_true",
                    help="print the score next to each failing cell")
    args = ap.parse_args()

    if args.threshold > 1:
        args.threshold /= 100.0        # accept --threshold 90

    jobs = []
    if args.gds_dir and args.sem_dir:
        jobs.append((args.gds_dir, args.sem_dir, args.out))
    elif args.root:
        splits = ["train", "val"] if args.split == "both" else [args.split]
        for sp in splits:
            jobs.append((args.root / "A" / sp, args.root / "C" / sp,
                         args.out / sp))
    else:
        sys.exit("provide --root (with --split) or both --gds-dir and --sem-dir")

    rows = []
    for gds_dir, sem_dir, out_dir in jobs:
        print(f"matching {gds_dir} (GDS cells) -> {sem_dir} (SEM)")
        gmap, smap = stem_map(gds_dir), stem_map(sem_dir)
        for stem in matched_stems(gmap, smap, "GDS", "SEM"):
            rows.append(match_one(gmap[stem], smap[stem], out_dir, stem, args))

    if not rows:
        sys.exit("no matched pairs found")

    print(f"\n{'image':<26}{'cells':>7}{'matched':>9}{'rate':>9}"
          f"{'mean':>8}{'min':>8}")
    for r in rows:
        print(f"{r['image']:<26}{r['cells']:>7}{r['matched']:>9}"
              f"{r['match_rate']:>8.1%}{r['mean_cell_score']:>8.3f}"
              f"{r['min_cell_score']:>8.3f}")
    tot_c = sum(r["cells"] for r in rows)
    tot_m = sum(r["matched"] for r in rows)
    print("-" * 67)
    print(f"{'total':<26}{tot_c:>7}{tot_m:>9}"
          f"{(tot_m / tot_c if tot_c else 0):>8.1%}"
          f"{np.mean([r['mean_cell_score'] for r in rows]):>8.3f}")

    csv_path = args.out / "cell_match_stats.csv"
    write_csv(csv_path, rows, ["image", "cells", "matched", "match_rate",
                               "mean_cell_score", "min_cell_score"])
    print(f"\noverlays in {args.out}/   stats in {csv_path}")


if __name__ == "__main__":
    main()
