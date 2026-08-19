#!/usr/bin/env python3
"""
Visual comparison of FLUX-generated SEM images (C) against original GDS
layouts (A), for the gds_2_sem directory structure:

    gds_2_sem/
      A/train  A/val      original GDS layouts (white rectangles on black)
      C/train  C/val      generated images from the FLUX.2 Klein LoRA

For every matched pair (by filename stem; ComfyUI counter suffixes like
`_00001_` on C images are stripped automatically) two visualizations are
produced, plus a combined side-by-side panel:

  overlay_<name>.png   C rendered semi-transparently on top of A, so the
                       original rectangles show through (--alpha controls
                       C's opacity).
  diff_<name>.png      Structure difference map:
                         WHITE  rectangle pixels present in both A and C
                         GREEN  extra pixels — in C but not in A
                         RED    missing pixels — in A but not in C
                         BLACK  background in both
  panel_<name>.png     [ A | C | overlay | diff ] in one strip.

C images are binarized with Otsu's threshold (they are grayscale SEM
renderings); A images with a fixed mid threshold (they are near-binary).
C is resized to A's dimensions if they differ (e.g. 512 vs 465).

A per-image and mean summary (IoU, extra %, missing %) is printed and
written to stats.csv in the output directory.

Usage:
  python overlay_compare.py --root /path/to/gds_2_sem --split val
  python overlay_compare.py --root /path/to/gds_2_sem --split both \
      --out comparisons --alpha 0.55
  # or explicit directories instead of --root/--split:
  python overlay_compare.py --a-dir gds_2_sem/A/val --c-dir gds_2_sem/C/val
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

WHITE = np.array([255, 255, 255], np.uint8)
GREEN = np.array([0, 200, 70], np.uint8)     # extra: in C, not in A
RED = np.array([230, 40, 40], np.uint8)      # missing: in A, not in C
BLACK = np.array([0, 0, 0], np.uint8)


def stem_map(d: Path) -> dict[str, Path]:
    out = {}
    for p in sorted(d.iterdir()):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        s = re.sub(r"_\d{5}_?$", "", p.stem)   # strip ComfyUI counter suffix
        out[s] = p
    return out


def otsu_threshold(a: np.ndarray) -> float:
    """Otsu's method on a uint8 array (no skimage dependency)."""
    hist = np.bincount(a.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    w = np.cumsum(hist)                    # class-0 weight
    m = np.cumsum(hist * np.arange(256))   # class-0 cumulative mean*weight
    mg = m[-1] / total
    valid = (w > 0) & (w < total)
    between = np.zeros(256)
    between[valid] = (mg * w[valid] - m[valid]) ** 2 / (w[valid] * (total - w[valid]))
    return float(np.argmax(between))


def label_strip(img: Image.Image, text: str) -> Image.Image:
    """Add a small caption bar above an image."""
    bar = 22
    out = Image.new("RGB", (img.width, img.height + bar), (16, 16, 16))
    out.paste(img, (0, bar))
    ImageDraw.Draw(out).text((6, 5), text, fill=(230, 230, 230))
    return out


def compare_pair(a_path: Path, c_path: Path, out_dir: Path, stem: str,
                 alpha: float) -> dict:
    a_img = Image.open(a_path).convert("L")
    c_img = Image.open(c_path).convert("L")
    if c_img.size != a_img.size:
        c_img = c_img.resize(a_img.size, Image.BILINEAR)

    a = np.asarray(a_img, np.uint8)
    c = np.asarray(c_img, np.uint8)

    a_mask = a >= 128                       # GDS is near-binary
    c_mask = c >= otsu_threshold(c)        # SEM rendering -> Otsu

    # ---- 1. transparency overlay: C (alpha) over A ----
    blend = ((1.0 - alpha) * a.astype(np.float32)
             + alpha * c.astype(np.float32)).clip(0, 255).astype(np.uint8)
    overlay = Image.fromarray(blend).convert("RGB")

    # ---- 2. difference map ----
    diff = np.zeros((*a.shape, 3), np.uint8)
    diff[a_mask & c_mask] = WHITE          # rectangles preserved
    diff[c_mask & ~a_mask] = GREEN         # additional area from C
    diff[a_mask & ~c_mask] = RED           # missing area from C
    diff_img = Image.fromarray(diff)

    # ---- 3. side-by-side panel ----
    panels = [label_strip(Image.fromarray(a).convert("RGB"), f"A (GDS): {stem}"),
              label_strip(Image.fromarray(c).convert("RGB"), "C (generated)"),
              label_strip(overlay, f"overlay (C @ {alpha:.0%})"),
              label_strip(diff_img, "diff  white=match green=extra red=missing")]
    gap = 4
    W = sum(p.width for p in panels) + gap * (len(panels) - 1)
    panel = Image.new("RGB", (W, panels[0].height), (40, 40, 40))
    x = 0
    for p in panels:
        panel.paste(p, (x, 0))
        x += p.width + gap

    overlay.save(out_dir / f"overlay_{stem}.png")
    diff_img.save(out_dir / f"diff_{stem}.png")
    panel.save(out_dir / f"panel_{stem}.png")

    inter = int((a_mask & c_mask).sum())
    union = int((a_mask | c_mask).sum())
    n = a_mask.size
    return {
        "image": stem,
        "iou": inter / max(union, 1),
        "extra_pct": 100.0 * int((c_mask & ~a_mask).sum()) / n,
        "missing_pct": 100.0 * int((a_mask & ~c_mask).sum()) / n,
    }


def run_split(a_dir: Path, c_dir: Path, out_dir: Path, alpha: float) -> list[dict]:
    a_map, c_map = stem_map(a_dir), stem_map(c_dir)
    stems = sorted(set(a_map) & set(c_map))
    skipped = sorted(set(c_map) - set(a_map))
    if skipped:
        print(f"  note: {len(skipped)} C image(s) without an A match skipped: "
              f"{skipped[:5]}")
    if not stems:
        print(f"  no matched pairs between {a_dir} and {c_dir}")
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = []
    for s in stems:
        stats.append(compare_pair(a_map[s], c_map[s], out_dir, s, alpha))
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path,
                    help="gds_2_sem directory containing A/ and C/")
    ap.add_argument("--split", choices=["train", "val", "both"], default="val")
    ap.add_argument("--a-dir", type=Path, help="explicit A directory")
    ap.add_argument("--c-dir", type=Path, help="explicit C directory")
    ap.add_argument("--out", type=Path, default=Path("comparisons"))
    ap.add_argument("--alpha", type=float, default=0.55,
                    help="opacity of C in the overlay, 0..1 (default 0.55)")
    args = ap.parse_args()

    jobs = []
    if args.a_dir and args.c_dir:
        jobs.append((args.a_dir, args.c_dir, args.out))
    elif args.root:
        splits = ["train", "val"] if args.split == "both" else [args.split]
        for sp in splits:
            jobs.append((args.root / "A" / sp, args.root / "C" / sp,
                         args.out / sp))
    else:
        sys.exit("provide --root (with --split) or both --a-dir and --c-dir")

    all_stats = []
    for a_dir, c_dir, out_dir in jobs:
        print(f"comparing {c_dir} -> {a_dir}")
        all_stats += run_split(a_dir, c_dir, out_dir, args.alpha)

    if not all_stats:
        sys.exit("no pairs compared")

    print(f"\n{'image':<28}{'IoU':>8}{'extra%':>9}{'missing%':>10}")
    for r in all_stats:
        print(f"{r['image']:<28}{r['iou']:>8.4f}{r['extra_pct']:>9.2f}"
              f"{r['missing_pct']:>10.2f}")
    print("-" * 55)
    print(f"{'mean':<28}{np.mean([r['iou'] for r in all_stats]):>8.4f}"
          f"{np.mean([r['extra_pct'] for r in all_stats]):>9.2f}"
          f"{np.mean([r['missing_pct'] for r in all_stats]):>10.2f}")

    csv_path = (args.out if not (args.a_dir and args.c_dir) else args.out) / "stats.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "iou", "extra_pct", "missing_pct"])
        w.writeheader()
        w.writerows(all_stats)
    print(f"\nstats written to {csv_path}; images in {args.out}/")


if __name__ == "__main__":
    main()
