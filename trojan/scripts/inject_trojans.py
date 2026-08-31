#!/usr/bin/env python3
"""
Inject synthetic hardware trojans (patterns A-J) into generated SEM images,
to build a labelled TEST SET for the detector.

Reads:
  --gds-dir   GDS layouts (A/val) — used only to place trojans plausibly
              (additions in whitespace, modifications on real cells).
  --sem-dir   generated SEM images (C/val) — the clean images to tamper.
Writes:
  --out-dir   tampered copies (untouched images are copied through too, so
              the detector is tested on a realistic mix of clean + trojaned).
  ground_truth.json  {image: {trojans: [{pattern,name,class,bbox}, ...]}}
              — clean images get an empty list.

Placement is on the GDS-derived cell mask (upscaled to the SEM size), so it
does not "cheat" by using the generated image's own structure.

Usage:
  python inject_trojans.py --gds-dir gds_2_sem/A/val --sem-dir gds_2_sem/C/val \
      --out-dir trojan_testset --rate 0.6 --max-per-image 2 --seed 0
  # force one of every pattern across the set (round-robin), for a demo:
  python inject_trojans.py ... --round-robin
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from matchlib import binarize, load_gray, resize_to, stem_map  # noqa: E402
import patterns as P  # noqa: E402


def inject_one(img_u8, gds_mask, keys, rng, max_per):
    img = img_u8.astype(np.float64)
    o = P.orientation_of(gds_mask)
    bg, fg, sig = P.sample_levels(img, gds_mask)
    placed = []
    n = int(rng.integers(1, max_per + 1))
    for key in keys[:n]:
        insts = P.REGISTRY[key].fn(img, gds_mask, rng, o, bg, fg, sig)
        placed.extend(insts)
    return np.clip(img, 0, 255).astype(np.uint8), placed


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gds-dir", required=True, type=Path)
    ap.add_argument("--sem-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--rate", type=float, default=0.6,
                    help="fraction of images that receive >=1 trojan")
    ap.add_argument("--max-per-image", type=int, default=2)
    ap.add_argument("--round-robin", action="store_true",
                    help="cycle A..J deterministically instead of random draw")
    ap.add_argument("--patterns", default="ABCDEFGHIJ",
                    help="subset of patterns to use (default all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    keys = [k for k in args.patterns if k in P.REGISTRY]
    if not keys:
        sys.exit("no valid patterns selected")
    rng = np.random.default_rng(args.seed)

    gmap, smap = stem_map(args.gds_dir), stem_map(args.sem_dir)
    stems = sorted(set(gmap) & set(smap))
    if not stems:
        sys.exit("no matched GDS/SEM pairs")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gt = {}
    rr = 0
    n_troj = n_clean = 0
    for stem in stems:
        sem = load_gray(smap[stem])
        gds = resize_to(load_gray(gmap[stem]), (sem.shape[1], sem.shape[0]),
                        nearest=True)
        gmask = binarize(gds, "fixed")

        out_name = smap[stem].name           # keep original filename
        do = rng.random() < args.rate
        if do:
            if args.round_robin:
                take = [keys[(rr + i) % len(keys)]
                        for i in range(int(rng.integers(1, args.max_per_image + 1)))]
                rr += len(take)
            else:
                take = list(rng.permutation(keys))
            out_img, placed = inject_one(sem, gmask := gmask, take, rng,
                                         args.max_per_image)
            # drop instances that couldn't be placed (returned nothing)
            placed = [p for p in placed if p]
        else:
            out_img, placed = sem, []

        Image.fromarray(out_img).save(args.out_dir / out_name)
        gt[out_name] = {"trojans": placed}
        if placed:
            n_troj += 1
        else:
            n_clean += 1

    gt_path = args.out_dir / "ground_truth.json"
    meta = {"catalog": P.catalog(),
            "summary": {"images": len(stems), "with_trojan": n_troj,
                        "clean": n_clean,
                        "instances": sum(len(v["trojans"]) for v in gt.values())},
            "images": gt}
    gt_path.write_text(json.dumps(meta, indent=2))

    counts = {}
    for v in gt.values():
        for t in v["trojans"]:
            counts[t["pattern"]] = counts.get(t["pattern"], 0) + 1
    print(f"wrote {len(stems)} images to {args.out_dir} "
          f"({n_troj} trojaned, {n_clean} clean)")
    print("instances per pattern:",
          ", ".join(f"{k}={counts.get(k, 0)}" for k in P.ALL_KEYS))
    print(f"ground truth -> {gt_path}")


if __name__ == "__main__":
    main()
