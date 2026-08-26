#!/usr/bin/env python3
"""
Original-vs-generated SEM matcher.

Overlays the generated SEM (C) on top of the real SEM (B) and highlights
where they agree and where they diverge — deliberately tolerant, not
pixel-exact.

Two metrics:

  structure  (default)  Both images are binarised (Otsu) and compared with
                        a --tolerance px slack, so a bar that lands one or
                        two pixels off still counts as a match.
                          GREEN  material present in BOTH (within tolerance)
                          RED    material in only one of the two — i.e. the
                                 generated image added or dropped it
                          untinted  background in both
                        Per-image score = tolerant IoU of the two masks.

  ssim                  Per-pixel local SSIM (11 px window, light blur to
                        stop sensor noise dominating) — compares texture as
                        well as placement.
                          GREEN  local SSIM >= threshold
                          RED    below threshold
                        Scored only where either image has material, unless
                        --include-background is given (flat noisy background
                        scores badly under SSIM and would swamp the map).
                        Per-image score = mean local SSIM over that region.

The caption bar is green when the image's overall score clears --threshold
(default 0.90) and red when it doesn't. B is resampled to C's dimensions if
they differ.

Usage:
  python match_sem_sem.py --ref-dir gds_2_sem/B/val --gen-dir gds_2_sem/C/val \
      --out matches/real_vs_gen
  python match_sem_sem.py --root gds_2_sem --split val --metric ssim
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from matchlib import (GREEN, RED, binarize, blur, caption, dilate,
                      gray_to_rgb, load_gray, local_ssim, matched_stems,
                      resize_to, stem_map, tint, write_csv)


def match_one(ref_p: Path, gen_p: Path, out_dir: Path, stem: str,
              args) -> dict:
    gen = load_gray(gen_p)
    ref = resize_to(load_gray(ref_p), (gen.shape[1], gen.shape[0]),
                    nearest=False)

    # "overlay them on top of each other": 50/50 blend as the backdrop
    base = gray_to_rgb(((1 - args.blend) * ref.astype(np.float64)
                        + args.blend * gen.astype(np.float64)))

    ref_m = binarize(ref, "otsu")
    gen_m = binarize(gen, "otsu")

    if args.metric == "structure":
        tol = args.tolerance
        agree = ((ref_m & dilate(gen_m, tol)) | (gen_m & dilate(ref_m, tol)))
        differ = (ref_m | gen_m) & ~agree
        inter = int((ref_m & dilate(gen_m, tol)).sum())
        union = int((ref_m | gen_m).sum())
        score = inter / max(union, 1)
        extra = float((gen_m & ~dilate(ref_m, tol)).sum()) / gen_m.size * 100
        missing = float((ref_m & ~dilate(gen_m, tol)).sum()) / ref_m.size * 100
        detail = (f"tolerant IoU {score:.1%}   extra {extra:.2f}%   "
                  f"missing {missing:.2f}%   tol {tol}px")
    else:
        x = blur(ref.astype(np.float64) / 255.0, args.blur)
        y = blur(gen.astype(np.float64) / 255.0, args.blur)
        smap = local_ssim(x, y, args.window)
        scope = (np.ones_like(ref_m) if args.include_background
                 else dilate(ref_m | gen_m, 2))
        agree = (smap >= args.threshold) & scope
        differ = (smap < args.threshold) & scope
        score = float(smap[scope].mean()) if scope.any() else 0.0
        extra = float(differ.sum()) / max(int(scope.sum()), 1) * 100
        missing = 0.0
        detail = (f"mean local SSIM {score:.3f}   below-threshold area "
                  f"{extra:.1f}%   window {args.window}px")

    tint(base, agree, GREEN, args.alpha)
    tint(base, differ, RED, args.alpha)

    passed = score >= args.threshold
    img = caption(base, [
        (f"{stem}   {'PASS' if passed else 'FAIL'} @ thr "
         f"{args.threshold:.0%}   {detail}", GREEN if passed else RED),
        (f"green = matches original SEM    red = differs    "
         f"metric {args.metric}", (200, 200, 200))])

    out_dir.mkdir(parents=True, exist_ok=True)
    img.save(out_dir / f"semmatch_{stem}.png")
    return {"image": stem, "score": round(score, 4), "pass": int(passed),
            "differ_pct": round(extra, 3), "missing_pct": round(missing, 3)}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("inputs")
    src.add_argument("--ref-dir", type=Path, help="original SEM, e.g. gds_2_sem/B/val")
    src.add_argument("--gen-dir", type=Path, help="generated SEM, e.g. gds_2_sem/C/val")
    src.add_argument("--root", type=Path, help="gds_2_sem (use with --split)")
    src.add_argument("--split", choices=["train", "val", "both"], default="val")
    ap.add_argument("--out", type=Path, default=Path("matches/real_vs_gen"))
    ap.add_argument("--threshold", type=float, default=0.90)
    ap.add_argument("--metric", choices=["structure", "ssim"], default="structure")
    ap.add_argument("--tolerance", type=int, default=2,
                    help="structure mode: px of slack (default 2)")
    ap.add_argument("--window", type=int, default=11,
                    help="ssim mode: window size in px (default 11)")
    ap.add_argument("--blur", type=float, default=1.0,
                    help="ssim mode: pre-blur sigma to damp sensor noise")
    ap.add_argument("--include-background", action="store_true",
                    help="ssim mode: also score flat background regions")
    ap.add_argument("--alpha", type=float, default=0.40,
                    help="opacity of the green/red highlight (default 0.40)")
    ap.add_argument("--blend", type=float, default=0.50,
                    help="weight of the generated image in the backdrop")
    args = ap.parse_args()

    if args.threshold > 1:
        args.threshold /= 100.0

    jobs = []
    if args.ref_dir and args.gen_dir:
        jobs.append((args.ref_dir, args.gen_dir, args.out))
    elif args.root:
        splits = ["train", "val"] if args.split == "both" else [args.split]
        for sp in splits:
            jobs.append((args.root / "B" / sp, args.root / "C" / sp,
                         args.out / sp))
    else:
        sys.exit("provide --root (with --split) or both --ref-dir and --gen-dir")

    rows = []
    for ref_dir, gen_dir, out_dir in jobs:
        print(f"matching {gen_dir} (generated) -> {ref_dir} (original SEM)")
        rmap, gmap = stem_map(ref_dir), stem_map(gen_dir)
        for stem in matched_stems(rmap, gmap, "B", "C"):
            rows.append(match_one(rmap[stem], gmap[stem], out_dir, stem, args))

    if not rows:
        sys.exit("no matched pairs found")

    print(f"\n{'image':<26}{'score':>9}{'pass':>7}{'differ%':>10}{'missing%':>11}")
    for r in rows:
        print(f"{r['image']:<26}{r['score']:>9.4f}"
              f"{('yes' if r['pass'] else 'no'):>7}"
              f"{r['differ_pct']:>10.2f}{r['missing_pct']:>11.2f}")
    print("-" * 63)
    print(f"{'mean':<26}{np.mean([r['score'] for r in rows]):>9.4f}"
          f"{sum(r['pass'] for r in rows):>4}/{len(rows):<2}")

    csv_path = args.out / "sem_match_stats.csv"
    write_csv(csv_path, rows,
              ["image", "score", "pass", "differ_pct", "missing_pct"])
    print(f"\noverlays in {args.out}/   stats in {csv_path}")


if __name__ == "__main__":
    main()
