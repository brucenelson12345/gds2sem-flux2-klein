#!/usr/bin/env python3
"""
Convert injected trojan sets (images + ground_truth.json) into an
Ultralytics YOLO detection dataset, so you can train the `yolo` detector
backend as a learned alternative to the golden-model rules.

Classes are the ten patterns A-J (class ids 0-9, in that order).

Point --inputs at one or more directories, each produced by
inject_trojans.py (an images folder containing ground_truth.json). Clean
images (empty trojan list) are included as negatives — important so the
model learns not to fire on normal layout.

Writes a YOLO dataset:
    out/
      images/{train,val}/*.png
      labels/{train,val}/*.txt        # class cx cy w h  (normalised)
      dataset.yaml                    # ready for `yolo train data=dataset.yaml`

Usage:
  python export_yolo_dataset.py --inputs set1 set2 --out yolo_ds --val-frac 0.2
"""
import argparse
import json
import random
import shutil
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import patterns as P  # noqa: E402

CLASSES = P.ALL_KEYS
CID = {k: i for i, k in enumerate(CLASSES)}


def to_yolo(bbox, w, h):
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) / 2 / w
    cy = (y0 + y1) / 2 / h
    bw = (x1 - x0) / w
    bh = (y1 - y0) / h
    return cx, cy, bw, bh


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", required=True, nargs="+", type=Path,
                    help="one or more inject_trojans.py output directories")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)

    items = []  # (src_image_path, [ (cid, bbox), ... ])
    for d in args.inputs:
        gt = json.loads((d / "ground_truth.json").read_text())["images"]
        for name, rec in gt.items():
            p = d / name
            if not p.exists():
                continue
            items.append((p, [(CID[t["pattern"]], t["bbox"])
                              for t in rec["trojans"]]))
    if not items:
        sys.exit("no images found in --inputs")
    rng.shuffle(items)

    n_val = int(len(items) * args.val_frac)
    n_pos = n_neg = 0
    for idx, (src, objs) in enumerate(items):
        split = "val" if idx < n_val else "train"
        w, h = Image.open(src).size
        # unique-ish destination name (prefix parent set to avoid collisions)
        dst_name = f"{src.parent.name}__{src.name}"
        shutil.copy(src, args.out / "images" / split / dst_name)
        lbl = args.out / "labels" / split / (Path(dst_name).stem + ".txt")
        lines = []
        for cid, bbox in objs:
            cx, cy, bw, bh = to_yolo(bbox, w, h)
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        lbl.write_text("\n".join(lines))          # empty file = negative
        if objs:
            n_pos += 1
        else:
            n_neg += 1

    yaml = (f"# YOLO dataset for SEM trojan detection (patterns A-J)\n"
            f"path: {args.out.resolve()}\n"
            f"train: images/train\nval: images/val\n"
            f"nc: {len(CLASSES)}\n"
            f"names: [{', '.join(CLASSES)}]\n")
    (args.out / "dataset.yaml").write_text(yaml)

    print(f"exported {len(items)} images "
          f"({n_pos} with trojans, {n_neg} clean) -> {args.out}")
    print(f"  train {len(items) - n_val}  val {n_val}")
    print(f"  classes: {CLASSES}")
    print(f"dataset.yaml -> {args.out / 'dataset.yaml'}")
    print("\ntrain with:  python trojan/scripts/train_yolo.py "
          f"--data {args.out / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
