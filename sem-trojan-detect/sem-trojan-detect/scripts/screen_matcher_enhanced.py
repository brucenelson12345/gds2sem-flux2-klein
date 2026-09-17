#!/usr/bin/env python3
"""
screen_matcher_enhanced — B vs C with whole-cell events boxed and captioned.

Like screen_matcher.py, this overlays the golden SEM (B) on top of the
suspect SEM (C) and highlights what changed. What it adds is the distinction
between a whole cell changing and only *part* of a cell changing:

    WHOLE CELL — tinted, outlined, boxed, and captioned in the box colour

        missing    RED             a cell in B with no counterpart in C
        addition   GREEN           a cell in C with no counterpart in B
        join       BLUE            cells fused together (should be separated)
        split      YELLOW-ORANGE   a cell broken apart (should be connected)

    PART OF A CELL — tinted only, no box

        red    material lost from a cell that is otherwise still there
        green  material gained on a cell that was already there

        (a cell that got shorter, longer, wider or notched)

That partial case is what the plain matcher cannot see: a shortened cell
still pairs one-to-one with its counterpart and is reported as matched.

Writes into --out:
    enhanced_report.html   every B and C image plus the overlay, embedded,
                           with per-class counts and a per-image table
    enhanced_results.json  every event and parcel box, machine-readable
    overlays/*.png         the composited overlays on their own

Accuracy = clean / (clean + partial + whole-cell events).

Usage:
  python3 scripts/screen_matcher_enhanced.py --root /data/incoming/lot42 \\
      --out /data/runs/lot42_E
  python3 scripts/screen_matcher_enhanced.py --b-dir B/val --c-dir C/val \\
      --out run --min-parcel-area 80
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trojanlib.matcher_enhanced import (  # noqa: E402
    EnhancedParams, match_directories_enhanced, save_results_enhanced,
    write_enhanced_report)


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
                      help="px of slack when comparing shapes (default 2)")
    tune.add_argument("--min-area", type=int, default=24,
                      help="ignore cells smaller than this many px (default 24)")
    tune.add_argument("--min-parcel-area", type=int, default=40,
                      help="ignore shape disagreement smaller than this many px "
                           "(default 40). Raise it if edge noise between two "
                           "captures is reported as partial changes")
    tune.add_argument("--link-cover", type=float, default=0.35,
                      help="containment above which a B cell and a C cell are "
                           "the same cell (default 0.35)")

    look = ap.add_argument_group("rendering")
    look.add_argument("--alpha", type=float, default=0.50,
                      help="opacity of B over the C base (default 0.50)")
    look.add_argument("--tint", type=float, default=0.45,
                      help="opacity of the whole-cell fill (default 0.45)")
    look.add_argument("--parcel-tint", type=float, default=0.55,
                      help="opacity of the partial-cell fill (default 0.55)")
    look.add_argument("--box-width", type=int, default=2)
    look.add_argument("--font-size", type=int, default=13,
                      help="caption size in px (default 13)")
    look.add_argument("--thumb-width", type=int, default=460,
                      help="px width of images embedded in the report")
    look.add_argument("--lossless", action="store_true",
                      help="embed PNG instead of JPEG — pixel-exact, much larger")
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

    params = EnhancedParams(args.alpha, args.tint, args.parcel_tint,
                            args.tolerance, args.min_area,
                            args.min_parcel_area, args.link_cover,
                            args.box_width, args.font_size)
    print(f"matching {c_dir} (suspect) against {b_dir} (golden)")
    report = match_directories_enhanced(
        b_dir, c_dir, args.out, params,
        save_overlays=not args.no_overlays,
        thumb_width=args.thumb_width, lossless=args.lossless)
    results = save_results_enhanced(args.out, report)
    path = write_enhanced_report(args.out, report)
    print(f"\nresults -> {results}")
    print(f"report  -> {path}")


if __name__ == "__main__":
    main()
