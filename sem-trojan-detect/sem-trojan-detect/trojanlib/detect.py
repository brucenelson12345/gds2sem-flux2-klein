"""
The trojan detector.

Compares each suspect SEM image (C) against its golden model — the GDS
layout (A, the designed intent) and the original SEM (B, the known-good
capture) — flags regions that differ, and classifies each into a pattern
A-J.

Backends
    golden (default)  Golden-model differencing + a calibrated geometric /
                      photometric rule classifier. Deterministic, no
                      training, fully offline. The reference implementation.
    yolo              A trained Ultralytics model does detection and
                      classification directly on C (no golden reference
                      needed at inference). Build one with
                      scripts/export_yolo_dataset.py + scripts/train_yolo.py.

B is optional: without it, dopant-class (intensity-only) trojans cannot be
seen, since their geometry is unchanged.

Importable:
    from trojanlib import screen_directory
    report = screen_directory(a_dir, b_dir, c_dir, out_dir)

CLI:  python -m trojanlib.detect --root INPUT --out D
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import patterns as P
from .imagelib import (binarize, blur, connected_components, dilate, erode,
                       load_gray, resize_to, stem_map)

# box colour per change-class
CLASS_COLOR = {"addition": (60, 220, 90), "bridge": (255, 170, 0),
               "modification": (0, 180, 255), "deletion": (240, 60, 60)}


@dataclass
class DetectParams:
    """Tuning knobs for the golden backend."""
    tolerance: int = 2          # px slack between suspect and golden masks
    intensity_delta: float = 28  # |C-B| level change flagged as dopant anomaly
    merge: int = 2              # px dilation merging nearby anomaly pixels
    min_area: int = 24          # ignore anomaly regions smaller than this
    min_conf: float = 0.3


# --------------------------------------------------------------------------
# feature extraction + classification
# --------------------------------------------------------------------------
def region_features(comp_mask, add, rem, inten, golden, golden_labels, o):
    """Features the rule classifier keys on, for one anomaly region."""
    ys, xs = np.nonzero(comp_mask)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    w, h = x1 - x0, y1 - y0
    area = int(comp_mask.sum())
    long_side, short_side = max(w, h), min(w, h)

    halo = dilate(comp_mask, 2) & golden          # golden features touched
    touched = set(np.unique(golden_labels[halo])) - {0}

    sub = comp_mask[y0:y1, x0:x1]
    col_prof, row_prof = sub.sum(axis=0), sub.sum(axis=1)
    prof = col_prof if o == "vertical" else row_prof

    return dict(
        bbox=(int(x0), int(y0), int(x1), int(y1)), w=int(w), h=int(h),
        area=area, bbox_fill=area / max(w * h, 1),
        aspect=long_side / max(short_side, 1),
        long_side=int(long_side), short_side=int(short_side),
        a_frac=float((comp_mask & add).sum()) / max(area, 1),
        r_frac=float((comp_mask & rem).sum()) / max(area, 1),
        i_frac=float((comp_mask & inten).sum()) / max(area, 1),
        n_touch=len(touched),
        has_v_arm=bool((col_prof > 0.6 * h).any()),
        has_h_arm=bool((row_prof > 0.6 * w).any()),
        crossings=int(np.sum((prof[1:] > 0) & (prof[:-1] == 0))))


def classify(f, o) -> tuple[str, float]:
    """Region features -> (pattern key, confidence).

    Thresholds calibrated on injected synthetic sets. Change-class is
    decided first from the addition/deletion/intensity fractions — the
    reliable signals — then additions are separated by size and aspect.
    n_touch is used only for the bridge test, where it is dependable (a
    bridge straddles two cells); edge-adjacent modifications lean on
    geometry instead, since their touch count is unreliable.
    """
    a, r, i = f["a_frac"], f["r_frac"], f["i_frac"]
    asp, short, long = f["aspect"], f["short_side"], f["long_side"]
    area, fill = f["area"], f["bbox_fill"]

    if i > 0.35 and a < 0.25 and r < 0.25:
        return "H", 0.8                      # intensity-only: dopant patch
    if r > 0.35 and a < 0.25:
        return "G", 0.85                     # material missing: line cut
    # --- additions ---
    if f["n_touch"] >= 2 and short <= 7:
        return "B", 0.85                     # straddles two cells: bridge
    if f["has_v_arm"] and f["has_h_arm"] and fill < 0.6:
        return "I", 0.75                     # L-shape: routing jog
    if long <= 15 and asp <= 1.7:
        return "C", 0.8                      # small square: via
    if area >= 900 and asp <= 1.9:
        return "F", 0.75                     # large square block: filler
    if asp >= 6 and short <= 8:
        return "J", 0.75                     # very long thin: parallel route
    if asp >= 2.3 and short <= 8 and area <= 220:
        return "D", 0.65                     # thin edge strip: widen
    if asp < 2.3 and short >= 10 and area <= 520:
        return "E", 0.65                     # chunky end block: extend
    return "A", 0.7                          # elongated standalone: extra cell


# --------------------------------------------------------------------------
# backends
# --------------------------------------------------------------------------
def detect_image(c_img, a_img=None, b_img=None, params: DetectParams = None):
    """Golden-model detection on one image. Returns a list of detections."""
    params = params or DetectParams()
    H, W = c_img.shape
    gds = resize_to(a_img, (W, H), nearest=True) if a_img is not None else None
    ref = resize_to(b_img, (W, H), nearest=False) if b_img is not None else None

    c_mask = binarize(c_img, "otsu")
    if gds is not None:
        golden = binarize(gds, "fixed")
    elif ref is not None:
        golden = binarize(ref, "otsu")
    else:
        raise ValueError("need at least one of A (GDS) or B (SEM) as golden")

    added = c_mask & ~dilate(golden, params.tolerance)
    removed = golden & ~dilate(c_mask, params.tolerance)

    intensity = np.zeros_like(c_mask)
    if ref is not None:
        shared = erode(golden & c_mask, 1)
        if shared.any():
            diff = blur(c_img.astype(np.float64) - ref.astype(np.float64), 1.0)
            intensity = shared & (np.abs(diff) > params.intensity_delta)

    added = erode(dilate(added, 1), 1)          # despeckle
    removed = erode(dilate(removed, 1), 1)
    anomaly = dilate(added | removed | intensity, params.merge)

    labels, n, _ = connected_components(anomaly)
    g_labels, _, _ = connected_components(golden)
    o = P.orientation_of(golden)

    dets = []
    for idx in range(1, n + 1):
        comp = labels == idx
        if int(comp.sum()) < params.min_area:
            continue
        f = region_features(comp, added, removed, intensity, golden,
                            g_labels, o)
        key, conf = classify(f, o)
        if conf < params.min_conf:
            continue
        pat = P.REGISTRY[key]
        dets.append({"pattern": key, "name": pat.name, "class": pat.cls,
                     "bbox": list(f["bbox"]),
                     "confidence": round(float(conf), 3)})
    return dets


def detect_image_yolo(c_path, model, min_conf: float):
    res = model.predict(str(c_path), conf=min_conf, verbose=False)[0]
    dets = []
    for b in res.boxes:
        raw = model.names[int(b.cls)]
        key = (raw[0].upper() if raw else "A")
        pat = P.REGISTRY.get(key)
        if pat is None:
            continue
        x0, y0, x1, y1 = (int(v) for v in b.xyxy[0].tolist())
        dets.append({"pattern": key, "name": pat.name, "class": pat.cls,
                     "bbox": [x0, y0, x1, y1],
                     "confidence": round(float(b.conf), 3)})
    return dets


# --------------------------------------------------------------------------
def annotate(c_img, dets, path: Path):
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


def screen_directory(a_dir, b_dir, c_dir, out_dir, backend="golden",
                     weights=None, params: DetectParams = None,
                     save_clean=False, quiet=False):
    """Screen every image in c_dir against its golden counterparts.
    Writes results.json + annotated/ into out_dir; returns the report dict."""
    params = params or DetectParams()
    c_dir, out_dir = Path(c_dir), Path(out_dir)
    cmap = stem_map(c_dir)
    amap = stem_map(Path(a_dir)) if a_dir and Path(a_dir).is_dir() else {}
    bmap = stem_map(Path(b_dir)) if b_dir and Path(b_dir).is_dir() else {}
    if not cmap:
        raise SystemExit(f"no images found in {c_dir}")

    model = None
    if backend == "yolo":
        if not weights or not Path(weights).exists():
            raise SystemExit("yolo backend needs --weights <trained .pt>")
        try:
            from ultralytics import YOLO
        except ImportError:
            raise SystemExit("yolo backend needs `pip install ultralytics`")
        model = YOLO(str(weights))

    out_dir.mkdir(parents=True, exist_ok=True)
    results, n_flagged, total = {}, 0, 0
    per_pattern = {k: 0 for k in P.ALL_KEYS}

    for stem in sorted(cmap):
        c_img = load_gray(cmap[stem])
        if backend == "golden":
            dets = detect_image(c_img,
                                load_gray(amap[stem]) if stem in amap else None,
                                load_gray(bmap[stem]) if stem in bmap else None,
                                params)
        else:
            dets = detect_image_yolo(cmap[stem], model, params.min_conf)

        name = cmap[stem].name
        if dets:
            n_flagged += 1
            total += len(dets)
            for d in dets:
                per_pattern[d["pattern"]] += 1
            annotate(c_img, dets, out_dir / "annotated" / name)
            results[name] = {"status": "trojan_detected",
                             "count": len(dets), "detections": dets}
        else:
            if save_clean:
                annotate(c_img, [], out_dir / "annotated" / name)
            results[name] = {"status": "no_trojan_detected",
                             "count": 0, "detections": []}

    report = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "inputs": {"A": str(a_dir), "B": str(b_dir), "C": str(c_dir)},
        "catalog": P.catalog(),
        "summary": {"images": len(cmap), "flagged": n_flagged,
                    "clean": len(cmap) - n_flagged, "detections": total,
                    "per_pattern": per_pattern},
        "images": results,
    }
    (out_dir / "results.json").write_text(json.dumps(report, indent=2))

    if not quiet:
        print(f"backend={backend}  images={len(cmap)}  "
              f"flagged={n_flagged}  detections={total}")
        print("per pattern:",
              ", ".join(f"{k}={per_pattern[k]}" for k in P.ALL_KEYS))
        print(f"results  -> {out_dir / 'results.json'}")
        print(f"annotated-> {out_dir / 'annotated'}/")
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, help="dir containing A/ B/ C/")
    ap.add_argument("--a-dir", type=Path)
    ap.add_argument("--b-dir", type=Path)
    ap.add_argument("--c-dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("D"))
    ap.add_argument("--backend", choices=["golden", "yolo"], default="golden")
    ap.add_argument("--weights", type=Path)
    ap.add_argument("--tolerance", type=int, default=2)
    ap.add_argument("--intensity-delta", type=float, default=28)
    ap.add_argument("--merge", type=int, default=2)
    ap.add_argument("--min-area", type=int, default=24)
    ap.add_argument("--min-conf", type=float, default=0.3)
    ap.add_argument("--save-clean", action="store_true")
    a = ap.parse_args(argv)

    if a.root:
        a_dir, b_dir, c_dir = a.root / "A", a.root / "B", a.root / "C"
    else:
        a_dir, b_dir, c_dir = a.a_dir, a.b_dir, a.c_dir
    if not c_dir or not Path(c_dir).is_dir():
        raise SystemExit("need --c-dir (suspect SEM) or --root with a C/ subdir")

    screen_directory(a_dir, b_dir, c_dir, a.out, a.backend, a.weights,
                     DetectParams(a.tolerance, a.intensity_delta, a.merge,
                                  a.min_area, a.min_conf),
                     a.save_clean)


if __name__ == "__main__":
    main()
