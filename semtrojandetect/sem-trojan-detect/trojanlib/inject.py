"""
Inject synthetic hardware trojans (patterns A-J) into clean SEM images, to
build a LABELLED test set for the detector.

Placement uses the GDS layout (additions go in whitespace, modifications on
real cells), so the injector never "cheats" by reading the SEM's own
structure. Clean (untampered) images are copied through as negatives, so a
generated set exercises both the detect and the don't-false-alarm paths.

Importable:
    from trojanlib import inject_directory
    summary = inject_directory(gds_dir, sem_dir, out_dir, rate=0.6)

CLI:  python -m trojanlib.inject --gds-dir A/val --sem-dir C/val \
          --out-dir testset --round-robin
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import patterns as P
from .imagelib import binarize, load_gray, resize_to, stem_map


def inject_image(sem_u8: np.ndarray, gds_mask: np.ndarray, keys, rng,
                 max_per: int):
    """Apply up to max_per patterns (from `keys`, in order) to one image.
    Returns (modified uint8 image, list of placed instances)."""
    img = sem_u8.astype(np.float64)
    o = P.orientation_of(gds_mask)
    bg, fg, sig = P.sample_levels(img, gds_mask)
    placed = []
    n = int(rng.integers(1, max_per + 1))
    for key in list(keys)[:n]:
        placed.extend(P.REGISTRY[key].fn(img, gds_mask, rng, o, bg, fg, sig))
    return np.clip(img, 0, 255).astype(np.uint8), placed


def inject_directory(gds_dir, sem_dir, out_dir, rate=0.6, max_per_image=2,
                     round_robin=False, keys=None, seed=0, quiet=False):
    """Tamper a fraction of the SEM images and write ground_truth.json.
    Returns the metadata dict that was written."""
    gds_dir, sem_dir, out_dir = Path(gds_dir), Path(sem_dir), Path(out_dir)
    keys = [k for k in (keys or P.ALL_KEYS) if k in P.REGISTRY]
    if not keys:
        raise ValueError("no valid patterns selected")

    rng = np.random.default_rng(seed)
    gmap, smap = stem_map(gds_dir), stem_map(sem_dir)
    stems = sorted(set(gmap) & set(smap))
    if not stems:
        raise SystemExit(f"no matched GDS/SEM pairs between {gds_dir} and {sem_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    images, rr, n_troj = {}, 0, 0
    for stem in stems:
        sem = load_gray(smap[stem])
        gds = resize_to(load_gray(gmap[stem]), (sem.shape[1], sem.shape[0]),
                        nearest=True)
        gmask = binarize(gds, "fixed")

        if rng.random() < rate:
            if round_robin:
                take = [keys[(rr + i) % len(keys)] for i in range(len(keys))]
                rr += 1
            else:
                take = list(rng.permutation(keys))
            out_img, placed = inject_image(sem, gmask, take, rng, max_per_image)
        else:
            out_img, placed = sem, []

        name = smap[stem].name
        Image.fromarray(out_img).save(out_dir / name)
        images[name] = {"trojans": placed}
        n_troj += bool(placed)

    counts = {k: 0 for k in P.ALL_KEYS}
    for v in images.values():
        for t in v["trojans"]:
            counts[t["pattern"]] += 1

    meta = {"catalog": P.catalog(),
            "summary": {"images": len(stems), "with_trojan": n_troj,
                        "clean": len(stems) - n_troj,
                        "instances": sum(counts.values()),
                        "per_pattern": counts},
            "images": images}
    (out_dir / "ground_truth.json").write_text(json.dumps(meta, indent=2))

    if not quiet:
        print(f"wrote {len(stems)} images to {out_dir} "
              f"({n_troj} trojaned, {len(stems) - n_troj} clean)")
        print("instances per pattern:",
              ", ".join(f"{k}={counts[k]}" for k in P.ALL_KEYS))
        print(f"ground truth -> {out_dir / 'ground_truth.json'}")
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gds-dir", required=True, type=Path)
    ap.add_argument("--sem-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--rate", type=float, default=0.6)
    ap.add_argument("--max-per-image", type=int, default=2)
    ap.add_argument("--round-robin", action="store_true")
    ap.add_argument("--patterns", default="".join(P.ALL_KEYS))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    inject_directory(a.gds_dir, a.sem_dir, a.out_dir, a.rate, a.max_per_image,
                     a.round_robin, list(a.patterns), a.seed)


if __name__ == "__main__":
    main()
