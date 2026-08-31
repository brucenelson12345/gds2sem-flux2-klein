#!/usr/bin/env python3
"""
Hardware-trojan detector for the GDS/SEM prototype.

Compares each suspect SEM image (C) against its golden model — the GDS
layout (A, the designed intent) and the original SEM (B, the known-good
capture) — flags regions that differ, classifies each into one of the ten
trojan patterns A-J, and writes:

  D/results.json          one entry per image:
                            "no_trojan_detected", or a list of detections
                            {pattern, name, class, bbox, confidence}
  D/annotated/<image>     copy with bounding boxes over detected regions
                          (clean images are not re-saved unless --save-clean)

Two backends (--backend):

  golden  (default)  Golden-model differencing + a geometric/photometric
                     rule classifier. Deterministic, needs no training, runs
                     fully offline. This is the working prototype detector.

  yolo               A trained YOLO model (Ultralytics) does detection +
                     classification directly on C. Use export_yolo_dataset.py
                     + train_yolo.py to build one from injected sets, then
                     pass --weights best.pt. Falls back with a clear error if
                     ultralytics or the weights are missing. The golden model
                     is still used to supply B/A context if available.

The golden backend is the reference implementation and the ground the YOLO
model is trained against; RF-DETR can be dropped in the same way as yolo.

Inputs are the A/B/C sibling directories (matched by filename stem,
ComfyUI counter suffixes ignored). B is optional — without it, dopant-class
(intensity-only) trojans can't be seen and A alone supplies golden geometry.

Usage:
  python detect_trojans.py --a-dir A/val --b-dir B/val --c-dir C/val --out D
  python detect_trojans.py --root INPUT_DIR --out D          # INPUT_DIR/{A,B,C}
  python detect_trojans.py --root INPUT_DIR --out D --backend yolo --weights best.pt
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from matchlib import (binarize, blur, connected_components, dilate, erode,  # noqa
                      load_gray, resize_to, stem_map)
import patterns as P  # noqa: E402

# colour per change-class for the annotated boxes (RGB)
CLASS_COLOR = {"addition": (60, 220, 90), "bridge": (255, 170, 0),
               "modification": (0, 180, 255), "deletion": (240, 60, 60)}


# --------------------------------------------------------------------------
# golden-model backend
# --------------------------------------------------------------------------
def region_features(comp_mask, add, rem, inten, golden, golden_labels, o):
    """Compute the features the rule classifier keys on for one region."""
    ys, xs = np.nonzero(comp_mask)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    w, h = x1 - x0, y1 - y0
    area = int(comp_mask.sum())
    bbox_fill = area / max(w * h, 1)
    long_side, short_side = max(w, h), min(w, h)
    aspect = long_side / max(short_side, 1)

    a_frac = float((comp_mask & add).sum()) / max(area, 1)
    r_frac = float((comp_mask & rem).sum()) / max(area, 1)
    i_frac = float((comp_mask & inten).sum()) / max(area, 1)

    # which golden components does this region touch (for bridge detection)?
    halo = dilate(comp_mask, 2) & golden
    touched = set(np.unique(golden_labels[halo])) - {0}

    # arms: does the bbox have both a horizontal and a vertical limb (L-shape)?
    sub = comp_mask[y0:y1, x0:x1]
    col_prof = sub.sum(axis=0)
    row_prof = sub.sum(axis=1)
    has_v_arm = (col_prof > 0.6 * h).any()
    has_h_arm = (row_prof > 0.6 * w).any()

    # internal periodicity (filler cell = several parallel sub-bars)
    prof = col_prof if o == "vertical" else row_prof
    crossings = int(np.sum((prof[1:] > 0) & (prof[:-1] == 0)))

    return dict(bbox=(int(x0), int(y0), int(x1), int(y1)), w=int(w), h=int(h),
                area=area, bbox_fill=bbox_fill, aspect=aspect,
                long_side=int(long_side), short_side=int(short_side),
                a_frac=a_frac, r_frac=r_frac, i_frac=i_frac,
                n_touch=len(touched), has_v_arm=has_v_arm, has_h_arm=has_h_arm,
                crossings=crossings)


def classify(f, o) -> tuple[str, float]:
    """Map region features -> (pattern key, confidence).

    Thresholds calibrated on the injected synthetic set (see the feature
    dump in the README's detector section). Change-class is decided first
    from the addition/deletion/intensity fractions — the reliable signals —
    then additions are separated by size/aspect. n_touch is used only for
    the bridge test, where it's dependable (a bridge straddles two cells);
    it's unreliable for edge-adjacent modifications, so those lean on
    geometry instead.
    """
    a, r, i = f["a_frac"], f["r_frac"], f["i_frac"]
    asp, short, long = f["aspect"], f["short_side"], f["long_side"]
    area, fill = f["area"], f["bbox_fill"]

    # --- intensity-only (dopant): on golden material, no shape change ---
    if i > 0.35 and a < 0.25 and r < 0.25:
        return "H", 0.8
    # --- deletion: golden material missing in the suspect ---
    if r > 0.35 and a < 0.25:
        return "G", 0.85

    # --- additions ---
    # bridge: straddles two distinct golden features with a thin link
    if f["n_touch"] >= 2 and short <= 7:
        return "B", 0.85
    # routing jog: L-shape — both arms present, bbox only partly filled
    if f["has_v_arm"] and f["has_h_arm"] and fill < 0.6:
        return "I", 0.75
    # via/contact: small and roughly square
    if long <= 15 and asp <= 1.7:
        return "C", 0.8
    # filler swap: a large, roughly square block
    if area >= 900 and asp <= 1.9:
        return "F", 0.75
    # parallel route: a very long, very thin added line
    if asp >= 6 and short <= 8:
        return "J", 0.75
    # line widen: a thin strip running along one line's long edge
    if asp >= 2.3 and short <= 8 and area <= 220:
        return "D", 0.65
    # line extend: a chunky block off one line's end (collinear)
    if asp < 2.3 and short >= 10 and area <= 520:
        return "E", 0.65
    # default: an elongated standalone added block -> extra cell
    return "A", 0.7


def detect_golden(a_img, b_img, c_img, args):
    H, W = c_img.shape
    gds = resize_to(a_img, (W, H), nearest=True) if a_img is not None else None
    ref = resize_to(b_img, (W, H), nearest=False) if b_img is not None else None

    c_mask = binarize(c_img, "otsu")
    if gds is not None:
        golden = binarize(gds, "fixed")
    elif ref is not None:
        golden = binarize(ref, "otsu")
    else:
        raise SystemExit("need at least one of A (GDS) or B (SEM) as golden")

    tol = args.tolerance
    added = c_mask & ~dilate(golden, tol)
    removed = golden & ~dilate(c_mask, tol)

    # intensity anomaly: on shared material, C much darker/brighter than B
    intensity = np.zeros_like(c_mask)
    if ref is not None:
        shared = erode(golden & c_mask, 1)
        if shared.any():
            diff = blur(c_img.astype(np.float64) - ref.astype(np.float64), 1.0)
            thr = args.intensity_delta
            intensity = shared & (np.abs(diff) > thr)

    # clean up speckle, then union and label candidate regions
    added = erode(dilate(added, 1), 1)
    removed = erode(dilate(removed, 1), 1)
    anomaly = dilate(added | removed | intensity, args.merge)
    labels, n, boxes = connected_components(anomaly)
    g_labels, _, _ = connected_components(golden)
    o = P.orientation_of(golden)

    dets = []
    for i in range(1, n + 1):
        comp = labels == i
        if int(comp.sum()) < args.min_area:
            continue
        f = region_features(comp, added, removed, intensity, golden,
                            g_labels, o)
        key, conf = classify(f, o)
        if conf < args.min_conf:
            continue
        pat = P.REGISTRY[key]
        dets.append({"pattern": key, "name": pat.name, "class": pat.cls,
                     "bbox": list(f["bbox"]), "confidence": round(float(conf), 3)})
    return dets


# --------------------------------------------------------------------------
# yolo backend (optional; needs a trained model)
# --------------------------------------------------------------------------
def detect_yolo(c_path, model, args):
    res = model.predict(str(c_path), conf=args.min_conf, verbose=False)[0]
    dets = []
    for b in res.boxes:
        key = model.names[int(b.cls)]
        key = key[0].upper() if key else "A"
        pat = P.REGISTRY.get(key)
        if pat is None:
            continue
        x0, y0, x1, y1 = (int(v) for v in b.xyxy[0].tolist())
        dets.append({"pattern": key, "name": pat.name, "class": pat.cls,
                     "bbox": [x0, y0, x1, y1],
                     "confidence": round(float(b.conf), 3)})
    return dets


# --------------------------------------------------------------------------
def annotate(c_img, dets, path):
    im = Image.fromarray(c_img).convert("RGB")
    d = ImageDraw.Draw(im)
    for det in dets:
        x0, y0, x1, y1 = det["bbox"]
        color = CLASS_COLOR.get(det["class"], (255, 255, 0))
        d.rectangle([x0, y0, x1, y1], outline=color, width=2)
        tag = f'{det["pattern"]}:{det["name"]} {det["confidence"]:.0%}'
        ty = max(0, y0 - 11)
        d.rectangle([x0, ty, x0 + 7 * len(tag), ty + 11], fill=(0, 0, 0))
        d.text((x0 + 1, ty), tag, fill=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, help="dir containing A/ B/ C/ subdirs")
    ap.add_argument("--a-dir", type=Path)
    ap.add_argument("--b-dir", type=Path)
    ap.add_argument("--c-dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("D"))
    ap.add_argument("--backend", choices=["golden", "yolo"], default="golden")
    ap.add_argument("--weights", type=Path, help="yolo backend: model .pt")
    ap.add_argument("--tolerance", type=int, default=2,
                    help="golden: px slack between suspect and golden masks")
    ap.add_argument("--intensity-delta", type=float, default=28,
                    help="golden: |C-B| level change flagged as dopant anomaly")
    ap.add_argument("--merge", type=int, default=2,
                    help="golden: px dilation that merges nearby anomaly pixels")
    ap.add_argument("--min-area", type=int, default=24,
                    help="ignore anomaly regions smaller than this (px)")
    ap.add_argument("--min-conf", type=float, default=0.3)
    ap.add_argument("--save-clean", action="store_true",
                    help="also copy images with no detections into annotated/")
    args = ap.parse_args()

    if args.root:
        a_dir, b_dir, c_dir = args.root / "A", args.root / "B", args.root / "C"
    else:
        a_dir, b_dir, c_dir = args.a_dir, args.b_dir, args.c_dir
    if not c_dir or not c_dir.is_dir():
        sys.exit("need --c-dir (suspect SEM) or --root with a C/ subdir")

    cmap = stem_map(c_dir)
    amap = stem_map(a_dir) if a_dir and a_dir.is_dir() else {}
    bmap = stem_map(b_dir) if b_dir and b_dir.is_dir() else {}

    model = None
    if args.backend == "yolo":
        if not args.weights or not args.weights.exists():
            sys.exit("yolo backend needs --weights <trained .pt>")
        try:
            from ultralytics import YOLO
        except ImportError:
            sys.exit("yolo backend needs `pip install ultralytics`")
        model = YOLO(str(args.weights))

    args.out.mkdir(parents=True, exist_ok=True)
    results = {}
    n_flagged = total_det = 0
    per_pattern = {k: 0 for k in P.ALL_KEYS}

    for stem in sorted(cmap):
        c_img = load_gray(cmap[stem])
        a_img = load_gray(amap[stem]) if stem in amap else None
        b_img = load_gray(bmap[stem]) if stem in bmap else None

        if args.backend == "golden":
            dets = detect_golden(a_img, b_img, c_img, args)
        else:
            dets = detect_yolo(cmap[stem], model, args)

        name = cmap[stem].name
        if dets:
            n_flagged += 1
            total_det += len(dets)
            for d in dets:
                per_pattern[d["pattern"]] += 1
            annotate(c_img, dets, args.out / "annotated" / name)
            results[name] = {"status": "trojan_detected",
                             "count": len(dets), "detections": dets}
        else:
            if args.save_clean:
                annotate(c_img, [], args.out / "annotated" / name)
            results[name] = {"status": "no_trojan_detected",
                             "count": 0, "detections": []}

    report = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend,
        "inputs": {"A": str(a_dir), "B": str(b_dir), "C": str(c_dir)},
        "catalog": P.catalog(),
        "summary": {"images": len(cmap), "flagged": n_flagged,
                    "clean": len(cmap) - n_flagged,
                    "detections": total_det,
                    "per_pattern": per_pattern},
        "images": results,
    }
    out_json = args.out / "results.json"
    out_json.write_text(json.dumps(report, indent=2))

    print(f"backend={args.backend}  images={len(cmap)}  "
          f"flagged={n_flagged}  detections={total_det}")
    print("per pattern:",
          ", ".join(f"{k}={per_pattern[k]}" for k in P.ALL_KEYS))
    print(f"results  -> {out_json}")
    print(f"annotated-> {args.out / 'annotated'}/")


if __name__ == "__main__":
    main()
