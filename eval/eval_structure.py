#!/usr/bin/env python3
"""
Quantify how well generated SEM images preserve the GDS rectangle structure.

For each triple (GDS control, generated SEM, [optional] real SEM):
  * IoU between the binarized GDS layout and the binarized generated image
    (Otsu threshold) — the "did the rectangles stay put" metric.
  * SSIM between generated and real SEM (if reference provided).

Use this to pick the best checkpoint: run batch_infer.py on val_control with
each saved LoRA step, then compare scores.

  pip install scikit-image numpy pillow    (already in both docker images'
  ecosystems; scikit-image may need adding: pip install scikit-image)

Usage:
  python eval_structure.py --gds-dir dataset/val_control \
      --gen-dir comfy_data/output/gds2sem --ref-dir dataset/val_target
Matching is by filename stem (generated files may have ComfyUI's _00001_
suffix — it is stripped automatically).
"""
import argparse
import re
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from skimage.filters import threshold_otsu
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    raise SystemExit("pip install scikit-image")

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_gray(p: Path, size: int) -> np.ndarray:
    img = Image.open(p).convert("L").resize((size, size), Image.LANCZOS)
    return np.asarray(img, dtype=np.float64) / 255.0


def binarize(a: np.ndarray) -> np.ndarray:
    return a >= threshold_otsu(a)


def stem_map(d: Path) -> dict[str, Path]:
    out = {}
    for p in sorted(d.iterdir()):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        s = re.sub(r"_\d{5}_?$", "", p.stem)  # strip ComfyUI counter suffix
        out[s] = p
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gds-dir", required=True, type=Path)
    ap.add_argument("--gen-dir", required=True, type=Path)
    ap.add_argument("--ref-dir", type=Path, default=None)
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()

    gds, gen = stem_map(args.gds_dir), stem_map(args.gen_dir)
    ref = stem_map(args.ref_dir) if args.ref_dir else {}
    stems = sorted(set(gds) & set(gen))
    if not stems:
        raise SystemExit("no matching stems between gds-dir and gen-dir")

    ious, ssims = [], []
    print(f"{'image':<28}{'IoU(gds,gen)':>14}{'SSIM(gen,ref)':>15}")
    for s in stems:
        g = load_gray(gds[s], args.size)
        y = load_gray(gen[s], args.size)
        m1, m2 = binarize(g), binarize(y)
        iou = (m1 & m2).sum() / max((m1 | m2).sum(), 1)
        ious.append(iou)
        line = f"{s:<28}{iou:>14.4f}"
        if s in ref:
            r = load_gray(ref[s], args.size)
            sv = ssim(y, r, data_range=1.0)
            ssims.append(sv)
            line += f"{sv:>15.4f}"
        print(line)

    print("-" * 57)
    print(f"{'mean':<28}{np.mean(ious):>14.4f}", end="")
    if ssims:
        print(f"{np.mean(ssims):>15.4f}")
    else:
        print()


if __name__ == "__main__":
    main()
