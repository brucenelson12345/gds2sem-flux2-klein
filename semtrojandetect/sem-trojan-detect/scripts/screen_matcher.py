#!/usr/bin/env python3
"""
screen_matcher — cell-level B vs C SEM comparison report.

Compares the original golden SEM images (B) against the newly captured
suspect SEM images (C), pairing files by filename. Every image is reduced to
its cells (connected bright regions); the two cell sets are matched, and
whatever fails to match is the difference:

    GREEN   present in B, MISSING from C   (material removed)
    RED     present in C, MISSING from B   (material gained)

The overlay puts **B on top of C** — C is the base, B is blended over it,
then unmatched cells are tinted and outlined.

Writes into --out:
    match_report.html   every B and C image plus the overlay, embedded;
                        summary tiles, the accuracy score, per-image table
    match_results.json  the same numbers, machine-readable
    overlays/*.png      the composited overlays on their own

Accuracy is scored on cells:  matched / (matched + missing + gained).

Usage:
  python3 scripts/screen_matcher.py --root /data/incoming/lot42 --out /data/runs/lot42_M
  python3 scripts/screen_matcher.py --b-dir B/val --c-dir C/val --out match_run
  # looser cell pairing, stronger B in the blend:
  python3 scripts/screen_matcher.py --root ... --out ... --match-iou 0.15 --alpha 0.65
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trojanlib.matcher import (MatchParams, match_directories,  # noqa: E402
                               write_match_report)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("inputs")
    src.add_argument("--root", type=Path,
                     help="directory holding B/ and C/ subdirectories")
    src.add_argument("--b-dir", type=Path, help="golden SEM directory")
    src.add_argument("--c-dir", type=Path, help="suspect SEM directory")
    ap.add_argument("--out", required=True, type=Path)

    tune = ap.add_argument_group("matching")
    tune.add_argument("--tolerance", type=int, default=2,
                      help="px of slack when testing cell overlap (default 2)")
    tune.add_argument("--min-area", type=int, default=24,
                      help="ignore cells smaller than this many px (default 24)")
    tune.add_argument("--match-iou", type=float, default=0.25,
                      help="overlap above which two cells are the same cell "
                           "(default 0.25; lower it if a shifted capture "
                           "reports false missing/gained pairs)")

    look = ap.add_argument_group("rendering")
    look.add_argument("--alpha", type=float, default=0.50,
                      help="opacity of B over the C base (default 0.50)")
    look.add_argument("--tint", type=float, default=0.45,
                      help="opacity of the green/red cell fill (default 0.45)")
    look.add_argument("--thumb-width", type=int, default=460,
                      help="px width of images embedded in the report")
    look.add_argument("--lossless", action="store_true",
                      help="embed PNG instead of JPEG — pixel-exact but a much "
                           "larger report file")
    look.add_argument("--no-overlays", action="store_true",
                      help="skip writing overlays/*.png (report still has them)")
    args = ap.parse_args()

    if args.root:
        b_dir, c_dir = args.root / "B", args.root / "C"
    else:
        b_dir, c_dir = args.b_dir, args.c_dir
    for d, lab in ((b_dir, "B"), (c_dir, "C")):
        if not d or not Path(d).is_dir():
            sys.exit(f"need a {lab} directory — pass --root (with B/ and C/) "
                     f"or --b-dir/--c-dir")

    params = MatchParams(args.alpha, args.tint, args.tolerance,
                         args.min_area, args.match_iou)
    print(f"matching {c_dir} (suspect) against {b_dir} (golden)")
    report = match_directories(b_dir, c_dir, args.out, params,
                               save_overlays=not args.no_overlays,
                               thumb_width=args.thumb_width,
                               lossless=args.lossless)
    path = write_match_report(args.out, report)
    print(f"\nresults -> {args.out / 'match_results.json'}")
    print(f"report  -> {path}")


if __name__ == "__main__":
    main()
