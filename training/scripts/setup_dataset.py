#!/usr/bin/env python3
"""
Set up training against the gds_2_sem directory structure IN PLACE:

    gds_2_sem/
      A/train   GDS images (controls)      A/val   GDS validation images
      B/train   SEM images (targets)       B/val   SEM validation images

This script does three things (no images are modified or moved):

  1. VALIDATE  — checks that every A/train image has a matching B/train
     image (same filename stem, extension may differ), same for val.
  2. CAPTIONS  — (optional, default on) writes one .txt caption per SEM
     image into B/train, with the bar orientation (horizontal/vertical)
     auto-detected from the matching GDS image in A/train. Purely
     additive; skip with --no-captions to rely on the config's
     default_caption instead. Existing .txt files are left untouched
     unless --overwrite-captions is given.
  3. CONFIG    — (optional, with --config) rewrites the `prompts:` block of
     the training YAML so train-time samples use your real A/val
     filenames via --ctrl_img.

Usage:
  python setup_dataset.py --root /path/to/gds_2_sem \
      --config workspace/config.yaml --num-sample-prompts 4
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

CAPTION_TMPL = (
    "{trigger} convert this GDS standard cell layout into a scanning electron "
    "microscope image, preserving the exact position and size of every "
    "{orientation} rectangle, grayscale SEM texture with realistic noise, "
    "edge roughness and slight blur"
)
SAMPLE_TMPL = (
    "{trigger} convert this GDS standard cell layout into a scanning electron "
    "microscope image, preserving the exact position and size of every "
    "rectangle --ctrl_img {ctrl}"
)
# In-container mount point used by run_training.sh
CONTAINER_ROOT = "/workspace/gds_2_sem"


def stem_map(d: Path) -> dict[str, Path]:
    if not d.is_dir():
        sys.exit(f"ERROR: missing directory: {d}")
    return {p.stem: p for p in sorted(d.iterdir())
            if p.suffix.lower() in IMG_EXTS}


def detect_orientation(p: Path) -> str:
    """Bars run vertical -> structure varies along x (column means spread)."""
    a = np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0
    col_var = float(np.var(a.mean(axis=0)))
    row_var = float(np.var(a.mean(axis=1)))
    return "vertical" if col_var >= row_var else "horizontal"


def validate(root: Path) -> tuple[dict, dict, dict, dict]:
    a_train = stem_map(root / "A" / "train")
    b_train = stem_map(root / "B" / "train")
    a_val = stem_map(root / "A" / "val")
    b_val = stem_map(root / "B" / "val")

    ok = True
    for name, x, y in (("train", a_train, b_train), ("val", a_val, b_val)):
        only_a = sorted(set(x) - set(y))
        only_b = sorted(set(y) - set(x))
        print(f"{name}: {len(set(x) & set(y))} matched pairs "
              f"(A: {len(x)} images, B: {len(y)} images)")
        if only_a:
            ok = False
            print(f"  WARNING: in A/{name} but not B/{name}: {only_a[:10]}")
        if only_b:
            ok = False
            print(f"  WARNING: in B/{name} but not A/{name}: {only_b[:10]}")
    if not ok:
        print("  -> unmatched images are silently skipped by ai-toolkit's "
              "control matching; fix the names if unintended.")
    return a_train, b_train, a_val, b_val


def write_captions(a_train: dict, b_train: dict, trigger: str, overwrite: bool):
    written = kept = 0
    for stem in sorted(set(a_train) & set(b_train)):
        txt = b_train[stem].with_suffix(".txt")
        if txt.exists() and not overwrite:
            kept += 1
            continue
        orientation = detect_orientation(a_train[stem])
        txt.write_text(CAPTION_TMPL.format(trigger=trigger,
                                           orientation=orientation) + "\n")
        written += 1
    print(f"captions: wrote {written} .txt files into B/train"
          + (f", kept {kept} existing" if kept else ""))


def patch_config(config: Path, a_val: dict, n: int, trigger: str):
    text = config.read_text()
    names = sorted(a_val)[:n]
    if not names:
        sys.exit("ERROR: no images in A/val to use as sample prompts")
    lines = "\n".join(
        '          - "' + SAMPLE_TMPL.format(
            trigger=trigger,
            ctrl=f"{CONTAINER_ROOT}/A/val/{a_val[s].name}") + '"'
        for s in names)
    new_text, cnt = re.subn(
        r"(^        prompts:\n)(?:^          - .*\n)+",
        r"\1" + lines + "\n",
        text, count=1, flags=re.M)
    if cnt != 1:
        sys.exit(f"ERROR: could not locate the prompts block in {config}")
    config.write_text(new_text)
    print(f"config: sample prompts now use {len(names)} images from A/val "
          f"-> {config}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path,
                    help="path to gds_2_sem")
    ap.add_argument("--config", type=Path, default=None,
                    help="training YAML whose sample prompts should be "
                         "rewritten with real A/val filenames")
    ap.add_argument("--num-sample-prompts", type=int, default=4)
    ap.add_argument("--trigger", default="g2s3m")
    ap.add_argument("--no-captions", action="store_true",
                    help="skip writing .txt captions into B/train")
    ap.add_argument("--overwrite-captions", action="store_true")
    args = ap.parse_args()

    a_train, b_train, a_val, _ = validate(args.root)
    if not (set(a_train) & set(b_train)):
        sys.exit("ERROR: no matched training pairs found")

    if not args.no_captions:
        write_captions(a_train, b_train, args.trigger, args.overwrite_captions)
    if args.config:
        patch_config(args.config, a_val, args.num_sample_prompts, args.trigger)
    print("done.")


if __name__ == "__main__":
    main()
