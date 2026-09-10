"""
B ↔ C SEM matcher — cell-level differences, connectivity changes, and
grouped trojan regions.

Every image is reduced to its *cells* (8-connected bright regions). The B
and C cell sets are linked into an overlap graph, and each connected
component of that graph says what happened to those cells:

    1 B ↔ 1 C     matched, left untinted
    1 B,  no C    MISSING from C   — material removed          RED
    no B,  1 C    GAINED in C      — material added            GREEN
    n B ↔ 1 C     MERGED           — cells that should be separated   LIGHT BLUE
    1 B ↔ n C     SPLIT            — cells that should be connected   LIGHT BLUE
    n B ↔ n C     TANGLED          — connectivity rewired             LIGHT BLUE

Linking uses *containment* (overlap over the smaller cell) rather than IoU,
because a merged cell is far larger than either cell it swallowed — their
IoU is low while containment is near 1.

Anomalies that sit near each other are then clustered into **trojan
regions**, drawn as a yellow labelled box and classified from the mix of
changes inside, one-to-one with the five layout patterns in
`trojanlib.gds_trojans`:

    only additions          → Trojan A   inserted_cluster
    only removals           → Trojan B   depopulated
    only merges             → Trojan C   merged_pair
    only splits             → Trojan D   severed_net
    any mixture             → Trojan E   rerouted_block

The overlay composites **B on top of C**: C is the base, B is blended over
it at `alpha`, then anomalies are tinted and the regions boxed.

Colour note: red=missing / green=gained matches gds2sem's overlay_compare.
`--legacy-colors` restores the earlier green=missing / red=gained scheme.

Accuracy is scored on cells:

    accuracy = matched / (matched + missing + gained + connectivity)

Importable:
    from trojanlib.matcher import match_directories, write_match_report

CLI:  python -m trojanlib.matcher --b-dir B/val --c-dir C/val --out match_run
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .gds_trojans import ALL_PATTERNS, PATTERNS, label_of
from .imagelib import (binarize, connected_components, dilate, erode,
                       load_gray, resize_to, stem_map)

RED = (240, 70, 60)          # in B, missing from C
GREEN = (60, 220, 110)       # in C, gained
BLUE = (110, 200, 255)       # connectivity: should be connected / separated
YELLOW = (255, 205, 40)      # trojan region box

# component kind -> (colour key, edit kind used for region classification)
KINDS = {
    "missing": ("red", "remove"),
    "gained": ("green", "add"),
    "merged": ("blue", "merge"),
    "split": ("blue", "split"),
    "tangled": ("blue", "tangle"),
}


@dataclass
class MatchParams:
    alpha: float = 0.50        # opacity of B over the C base
    tint: float = 0.45         # opacity of the anomaly fill
    tolerance: int = 2         # px of slack when testing overlap
    min_area: int = 24         # ignore blobs smaller than this (px)
    link_cover: float = 0.35   # containment above which two cells are linked
    group_gap: int = 56        # px within which anomalies join one region
    legacy_colors: bool = False


def _colors(p: MatchParams):
    """(missing, gained) — swapped under --legacy-colors."""
    return (GREEN, RED) if p.legacy_colors else (RED, GREEN)


# --------------------------------------------------------------------------
# union-find
# --------------------------------------------------------------------------
class _UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra

    def groups(self):
        out = {}
        for k in list(self.p):
            out.setdefault(self.find(k), []).append(k)
        return list(out.values())


# --------------------------------------------------------------------------
# cells + overlap graph
# --------------------------------------------------------------------------
def _cells(mask, min_area):
    labels, n, boxes = connected_components(mask)
    out = {}
    for i in range(1, n + 1):
        x0, y0, x1, y1 = boxes[i - 1]
        area = int((labels[y0:y1, x0:x1] == i).sum())
        if area >= min_area:
            out[i] = {"bbox": (int(x0), int(y0), int(x1), int(y1)), "area": area}
    return labels, out


def _bbox_near(p, q, pad):
    return not (p[2] + pad <= q[0] or q[2] + pad <= p[0]
                or p[3] + pad <= q[1] or q[3] + pad <= p[1])


def _union_box(p, q):
    return (min(p[0], q[0]), min(p[1], q[1]), max(p[2], q[2]), max(p[3], q[3]))


def analyse(b_mask, c_mask, params: MatchParams):
    """Overlap-graph analysis. Returns (components, b_lab, c_lab, counts).

    Each component: {kind, b_ids, c_ids, bbox, iou}.
    """
    b_lab, b_cells = _cells(b_mask, params.min_area)
    c_lab, c_cells = _cells(c_mask, params.min_area)
    tol = params.tolerance

    uf = _UF()
    for i in b_cells:
        uf.find(("b", i))
    for j in c_cells:
        uf.find(("c", j))

    pair_iou = {}
    for i, bi in b_cells.items():
        for j, cj in c_cells.items():
            if not _bbox_near(bi["bbox"], cj["bbox"], tol + 1):
                continue
            box = _union_box(bi["bbox"], cj["bbox"])
            x0, y0, x1, y1 = box
            a = b_lab[y0:y1, x0:x1] == i
            c = c_lab[y0:y1, x0:x1] == j
            inter = int((a & (dilate(c, tol) if tol else c)).sum())
            if not inter:
                continue
            # containment, not IoU: a merged cell contains each part it absorbed
            cover = inter / max(1, min(bi["area"], cj["area"]))
            if cover >= params.link_cover:
                uf.union(("b", i), ("c", j))
                union_px = int((a | c).sum())
                pair_iou[(i, j)] = inter / union_px if union_px else 0.0

    comps = []
    counts = {k: 0 for k in KINDS}
    counts["matched"] = 0
    for nodes in uf.groups():
        b_ids = sorted(n[1] for n in nodes if n[0] == "b")
        c_ids = sorted(n[1] for n in nodes if n[0] == "c")
        nb, nc = len(b_ids), len(c_ids)
        if nb == 1 and nc == 1:
            kind = "matched"
        elif nb >= 1 and nc == 0:
            kind = "missing"
        elif nb == 0 and nc >= 1:
            kind = "gained"
        elif nb >= 2 and nc == 1:
            kind = "merged"
        elif nb == 1 and nc >= 2:
            kind = "split"
        else:
            kind = "tangled"

        boxes = ([b_cells[i]["bbox"] for i in b_ids]
                 + [c_cells[j]["bbox"] for j in c_ids])
        bbox = boxes[0]
        for bx in boxes[1:]:
            bbox = _union_box(bbox, bx)
        comps.append({"kind": kind, "b_ids": b_ids, "c_ids": c_ids,
                      "bbox": tuple(int(v) for v in bbox),
                      "iou": round(pair_iou.get((b_ids[0], c_ids[0]), 0.0), 4)
                      if kind == "matched" else None})
        counts[kind] += 1

    return comps, b_lab, c_lab, counts, len(b_cells), len(c_cells)


# --------------------------------------------------------------------------
# trojan regions
# --------------------------------------------------------------------------
def group_regions(comps, params: MatchParams, shape):
    """Cluster nearby anomalies into labelled trojan regions."""
    anomalies = [c for c in comps if c["kind"] != "matched"]
    if not anomalies:
        return []
    uf = _UF()
    for idx in range(len(anomalies)):
        uf.find(idx)
    for a in range(len(anomalies)):
        for b in range(a + 1, len(anomalies)):
            if _bbox_near(anomalies[a]["bbox"], anomalies[b]["bbox"],
                          params.group_gap):
                uf.union(a, b)

    h, w = shape
    regions = []
    for members in uf.groups():
        parts = [anomalies[i] for i in members]
        bbox = parts[0]["bbox"]
        for p in parts[1:]:
            bbox = _union_box(bbox, p["bbox"])
        pad = 10
        bbox = (max(0, bbox[0] - pad), max(0, bbox[1] - pad),
                min(w, bbox[2] + pad), min(h, bbox[3] + pad))

        kinds = set()
        for p in parts:
            k = KINDS[p["kind"]][1]
            kinds |= ({"merge", "split"} if k == "tangle" else {k})
        if kinds == {"add"}:
            key = "A"
        elif kinds == {"remove"}:
            key = "B"
        elif kinds == {"merge"}:
            key = "C"
        elif kinds == {"split"}:
            key = "D"
        else:
            key = "E"

        regions.append({
            "pattern": key, "label": label_of(key), "name": PATTERNS[key][0],
            "bbox": [int(v) for v in bbox],
            "kinds": sorted(kinds),
            "anomalies": len(parts),
            "breakdown": {k: sum(1 for p in parts if p["kind"] == k)
                          for k in KINDS if any(p["kind"] == k for p in parts)},
        })
    regions.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))
    return regions


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------
def _tint(rgb, mask, color, a):
    if mask.any():
        rgb[mask] = (1 - a) * rgb[mask] + a * np.array(color, np.float64)


def render_overlay(b_img, c_img, comps, regions, b_lab, c_lab,
                   params: MatchParams):
    """C as the base, B blended over it, anomalies tinted, regions boxed."""
    miss_col, gain_col = _colors(params)
    palette = {"red": miss_col, "green": gain_col, "blue": BLUE}

    base = ((1 - params.alpha) * c_img.astype(np.float64)
            + params.alpha * b_img.astype(np.float64))
    rgb = np.repeat(base[:, :, None], 3, axis=2)

    for comp in comps:
        if comp["kind"] == "matched":
            continue
        color = palette[KINDS[comp["kind"]][0]]
        x0, y0, x1, y1 = comp["bbox"]
        sub = np.zeros((y1 - y0, x1 - x0), bool)
        for i in comp["b_ids"]:
            sub |= b_lab[y0:y1, x0:x1] == i
        for j in comp["c_ids"]:
            sub |= c_lab[y0:y1, x0:x1] == j
        view = rgb[y0:y1, x0:x1]
        _tint(view, sub, color, params.tint)
        _tint(view, sub & ~erode(sub, 1), color, 1.0)     # solid outline

    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    if regions:
        d = ImageDraw.Draw(img)
        for r in regions:
            x0, y0, x1, y1 = r["bbox"]
            d.rectangle([x0, y0, x1, y1], outline=YELLOW, width=2)
            tag = r["label"]
            tw = 7 * len(tag) + 6
            ty = y0 - 12 if y0 >= 12 else y1
            d.rectangle([x0, ty, x0 + tw, ty + 12], fill=(0, 0, 0))
            d.text((x0 + 3, ty + 1), tag, fill=YELLOW)
    return img


def _b64(img: Image.Image, width: int, lossless: bool = False):
    """(base64, mime). SEM grain compresses badly as PNG, so JPEG by default."""
    if width and img.width > width:
        img = img.resize((width, round(img.height * width / img.width)),
                         Image.LANCZOS)
    buf = io.BytesIO()
    if lossless:
        img.convert("RGB").save(buf, "PNG", optimize=True)
        return base64.b64encode(buf.getvalue()).decode(), "png"
    img.convert("RGB").save(buf, "JPEG", quality=86, subsampling=0)
    return base64.b64encode(buf.getvalue()).decode(), "jpeg"


# --------------------------------------------------------------------------
# optional scoring against the GDS injection ground truth
# --------------------------------------------------------------------------
def _iou_box(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    ua = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / ua if ua else 0.0


def _cover_box(a, b):
    """Intersection over the SMALLER box. A ground-truth region includes the
    untouched neighbours around the edit, so a detection drawn tightly around
    the changed cells sits inside it — IoU would call that a miss."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / smaller if smaller else 0.0


def score_against_truth(report, truth_path, cover=0.5):
    """Match detected regions to gds_trojans.json regions; adds 'truth' block.

    A ground-truth region may legitimately be found as several nearby
    detections (three inserted cells, say), so every detection lying inside a
    region is consumed by it: the region counts once as found, and those
    detections are not also counted as false positives.
    """
    truth = json.loads(Path(truth_path).read_text())
    timgs = truth.get("images", {})
    by_stem = {Path(k).stem: v for k, v in timgs.items()}

    tp = fp = fn = 0
    label_ok = 0
    fragmented = 0
    confusion = {k: {j: 0 for j in ALL_PATTERNS} for k in ALL_PATTERNS}
    for stem, v in report["images"].items():
        gt = by_stem.get(stem)
        if gt is None:
            continue
        gw, gh = gt.get("size", [None, None])
        cw, ch = v["size"]
        sx = cw / gw if gw else 1.0
        sy = ch / gh if gh else 1.0
        gts = [{"pattern": r["pattern"],
                "bbox": [r["bbox"][0] * sx, r["bbox"][1] * sy,
                         r["bbox"][2] * sx, r["bbox"][3] * sy]}
               for r in gt["trojans"]]
        dets = list(v["regions"])
        used = set()
        for g in gts:
            inside = [k for k, d in enumerate(dets)
                      if k not in used and _cover_box(g["bbox"], d["bbox"]) >= cover]
            if not inside:
                fn += 1
                continue
            used.update(inside)
            tp += 1
            if len(inside) > 1:
                fragmented += 1
            # the largest detection in the region carries the predicted label
            best = max(inside, key=lambda k: ((dets[k]["bbox"][2] - dets[k]["bbox"][0])
                                              * (dets[k]["bbox"][3] - dets[k]["bbox"][1])))
            pred = dets[best]["pattern"]
            confusion[g["pattern"]][pred] += 1
            label_ok += pred == g["pattern"]
        fp += len(dets) - len(used)

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    report["truth"] = {
        "source": str(truth_path), "cover": cover, "fragmented": fragmented,
        "regions_found": tp, "missed": fn, "spurious": fp,
        "precision": round(prec, 4), "recall": round(rec, 4),
        "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
        "label_accuracy": round(label_ok / tp, 4) if tp else 0.0,
        "confusion": confusion,
    }
    return report


# --------------------------------------------------------------------------
# directory pass
# --------------------------------------------------------------------------
def match_directories(b_dir, c_dir, out_dir, params: MatchParams = None,
                      save_overlays=True, thumb_width=460,
                      lossless=False, quiet=False):
    params = params or MatchParams()
    b_dir, c_dir, out_dir = Path(b_dir), Path(c_dir), Path(out_dir)
    bmap, cmap = stem_map(b_dir), stem_map(c_dir)
    stems = sorted(set(bmap) & set(cmap))
    only_b = sorted(set(bmap) - set(cmap))
    only_c = sorted(set(cmap) - set(bmap))
    if not stems:
        raise SystemExit(f"no matching filenames between {b_dir} and {c_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if save_overlays:
        (out_dir / "overlays").mkdir(exist_ok=True)

    images = {}
    tot = {"matched": 0, "missing": 0, "gained": 0,
           "merged": 0, "split": 0, "tangled": 0,
           "b_cells": 0, "c_cells": 0, "regions": 0}
    per_pattern = {k: 0 for k in ALL_PATTERNS}

    for stem in stems:
        c_img = load_gray(cmap[stem])
        b_img = resize_to(load_gray(bmap[stem]),
                          (c_img.shape[1], c_img.shape[0]), nearest=False)
        b_mask, c_mask = binarize(b_img, "otsu"), binarize(c_img, "otsu")

        comps, b_lab, c_lab, counts, nb, nc = analyse(b_mask, c_mask, params)
        regions = group_regions(comps, params, c_img.shape)
        conn = counts["merged"] + counts["split"] + counts["tangled"]
        defects = counts["missing"] + counts["gained"] + conn
        denom = counts["matched"] + defects
        acc = counts["matched"] / denom if denom else 1.0

        inter = int((b_mask & c_mask).sum())
        union = int((b_mask | c_mask).sum())

        overlay = render_overlay(b_img, c_img, comps, regions, b_lab, c_lab,
                                 params)
        if save_overlays:
            overlay.save(out_dir / "overlays" / f"match_{stem}.png")

        images[stem] = {
            "b_file": bmap[stem].name, "c_file": cmap[stem].name,
            "size": [int(c_img.shape[1]), int(c_img.shape[0])],
            "b_cells": nb, "c_cells": nc,
            "matched": counts["matched"],
            "missing_from_c": counts["missing"], "gained_in_c": counts["gained"],
            "merged": counts["merged"], "split": counts["split"],
            "tangled": counts["tangled"], "connectivity": conn,
            "cell_accuracy": round(acc, 4),
            "pixel_iou": round(inter / union if union else 1.0, 4),
            "regions": regions,
            "anomaly_boxes": [{"kind": c["kind"], "bbox": list(c["bbox"])}
                              for c in comps if c["kind"] != "matched"],
            "_thumbs": (_b64(Image.fromarray(b_img), thumb_width, lossless),
                        _b64(Image.fromarray(c_img), thumb_width, lossless),
                        _b64(overlay, thumb_width, lossless)),
        }
        for k in ("matched", "missing", "gained", "merged", "split", "tangled"):
            tot[k] += counts[k]
        tot["b_cells"] += nb
        tot["c_cells"] += nc
        tot["regions"] += len(regions)
        for r in regions:
            per_pattern[r["pattern"]] += 1
        if not quiet:
            tags = " ".join(r["label"].replace("Trojan ", "T") for r in regions)
            print(f"  {stem:<24} B{nb:>4} C{nc:>4}  ok {counts['matched']:>4}"
                  f"  miss {counts['missing']:>3}  gain {counts['gained']:>3}"
                  f"  conn {conn:>3}  acc {acc:>6.1%}"
                  + (f"  [{tags}]" if tags else ""))

    conn_tot = tot["merged"] + tot["split"] + tot["tangled"]
    denom = tot["matched"] + tot["missing"] + tot["gained"] + conn_tot
    overall = tot["matched"] / denom if denom else 1.0
    accs = [v["cell_accuracy"] for v in images.values()]

    report = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "inputs": {"B": str(b_dir), "C": str(c_dir)},
        "params": asdict(params),
        "catalog": [{"pattern": k, "label": label_of(k), "name": PATTERNS[k][0],
                     "description": PATTERNS[k][1]} for k in ALL_PATTERNS],
        "summary": {
            "pairs": len(stems),
            "b_cells": tot["b_cells"], "c_cells": tot["c_cells"],
            "matched": tot["matched"],
            "missing_from_c": tot["missing"], "gained_in_c": tot["gained"],
            "merged": tot["merged"], "split": tot["split"],
            "tangled": tot["tangled"], "connectivity": conn_tot,
            "trojan_regions": tot["regions"], "per_pattern": per_pattern,
            "images_with_regions": sum(1 for v in images.values() if v["regions"]),
            "overall_accuracy": round(overall, 4),
            "mean_image_accuracy": round(float(np.mean(accs)), 4),
            "worst_image": min(images, key=lambda k: images[k]["cell_accuracy"]),
            "unpaired_in_b": only_b, "unpaired_in_c": only_c,
        },
        "images": images,
    }

    if not quiet:
        s = report["summary"]
        print(f"\npairs {s['pairs']}   matched {s['matched']}   "
              f"missing {s['missing_from_c']}   gained {s['gained_in_c']}   "
              f"connectivity {s['connectivity']}")
        print(f"trojan regions {s['trojan_regions']} — "
              + ", ".join(f"{label_of(k)} {per_pattern[k]}" for k in ALL_PATTERNS))
        print(f"overall cell accuracy {s['overall_accuracy']:.2%}   "
              f"mean per-image {s['mean_image_accuracy']:.2%}")
    return report


def save_results(out_dir, report):
    slim = {k: {kk: vv for kk, vv in v.items() if kk != "_thumbs"}
            for k, v in report["images"].items()}
    p = Path(out_dir) / "match_results.json"
    p.write_text(json.dumps({**report, "images": slim}, indent=2))
    return p


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def write_match_report(out_dir, report: dict) -> Path:
    out_dir = Path(out_dir)
    s = report["summary"]
    acc = s["overall_accuracy"]
    legacy = report.get("params", {}).get("legacy_colors", False)
    miss_css, gain_css = ("g", "r") if legacy else ("r", "g")
    verdict = ("clean" if s["trojan_regions"] == 0 else
               f"{s['trojan_regions']} trojan region"
               f"{'s' if s['trojan_regions'] != 1 else ''} found")
    vclass = "ok" if s["trojan_regions"] == 0 else "bad"

    legend = (f"<span class='key {miss_css}'>&#9632; missing from C</span>"
              f"<span class='key {gain_css}'>&#9632; gained in C</span>"
              f"<span class='key b'>&#9632; should be connected / separated</span>"
              f"<span class='key y'>&#9633; trojan region</span>")

    pat_rows = "".join(
        f"<tr><td class='y'><b>{html.escape(label_of(k))}</b></td>"
        f"<td>{html.escape(PATTERNS[k][0])}</td>"
        f"<td>{html.escape(PATTERNS[k][1])}</td>"
        f"<td class='n'>{s['per_pattern'][k]}</td></tr>" for k in ALL_PATTERNS)

    truth_block = ""
    if "truth" in report:
        t = report["truth"]
        conf = t["confusion"]
        head = "".join(f"<th class='n'>{k}</th>" for k in ALL_PATTERNS)
        body = "".join(
            f"<tr><td class='y'>{k}</td>"
            + "".join(f"<td class='n'>{conf[k][j]}</td>" for j in ALL_PATTERNS)
            + "</tr>" for k in ALL_PATTERNS if sum(conf[k].values()))
        truth_block = f"""
<h2>Scored against the layout ground truth</h2>
<p class="sub">Detected regions matched to <code>{html.escape(t['source'])}</code>
  at containment ≥ {t['cover']}.</p>
<div class="tiles">
  <div class="tile acc"><div class="k">Region recall</div><div class="v">{t['recall']:.1%}</div></div>
  <div class="tile"><div class="k">Precision</div><div class="v">{t['precision']:.1%}</div></div>
  <div class="tile"><div class="k">Label accuracy</div><div class="v">{t['label_accuracy']:.1%}</div></div>
  <div class="tile"><div class="k">Found / missed / spurious</div>
    <div class="v">{t['regions_found']}·{t['missed']}·{t['spurious']}</div></div>
</div>
<table><thead><tr><th>true \\ predicted</th>{head}</tr></thead><tbody>{body}</tbody></table>"""

    cards = []
    for stem, v in sorted(report["images"].items(),
                          key=lambda kv: kv[1]["cell_accuracy"]):
        (b64b, mb), (b64c, mc), (b64o, mo) = v["_thumbs"]
        chips = (f"<span class='chip {miss_css}'>{v['missing_from_c']} missing</span>"
                 f"<span class='chip {gain_css}'>{v['gained_in_c']} gained</span>"
                 f"<span class='chip b'>{v['connectivity']} connectivity</span>"
                 f"<span class='chip'>{v['matched']} matched</span>"
                 f"<span class='chip acc'>{v['cell_accuracy']:.1%} accuracy</span>")
        tro = "".join(f"<span class='chip y'>{html.escape(r['label'])} · "
                      f"{html.escape(r['name'])}</span>" for r in v["regions"])
        cards.append(f"""
      <section class="card">
        <div class="cardhead"><h3>{html.escape(stem)}</h3>
          <div class="chips">{chips}</div></div>
        {f'<div class="chips troj">{tro}</div>' if tro else ''}
        <div class="trio">
          <figure><img src="data:image/{mb};base64,{b64b}" alt="B golden SEM">
            <figcaption>B · golden — {html.escape(v['b_file'])} · {v['b_cells']} cells</figcaption></figure>
          <figure><img src="data:image/{mc};base64,{b64c}" alt="C suspect SEM">
            <figcaption>C · suspect — {html.escape(v['c_file'])} · {v['c_cells']} cells</figcaption></figure>
          <figure><img src="data:image/{mo};base64,{b64o}" alt="B over C with differences highlighted">
            <figcaption><b>B over C</b> — {legend}</figcaption></figure>
        </div>
      </section>""")

    unpaired = ""
    if s["unpaired_in_b"] or s["unpaired_in_c"]:
        unpaired = (f"<p class='warn'>Unpaired files skipped — only in B: "
                    f"{html.escape(', '.join(s['unpaired_in_b']) or 'none')}; "
                    f"only in C: "
                    f"{html.escape(', '.join(s['unpaired_in_c']) or 'none')}.</p>")

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>B vs C SEM match report</title><style>
 :root{{color-scheme:dark}}
 body{{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:0;
      background:#111;color:#ddd;line-height:1.55}}
 .wrap{{max-width:1180px;margin:0 auto;padding:28px 22px 72px}}
 h1{{color:#fff;margin:0 0 6px;font-size:26px}}
 h2{{color:#fff;font-size:18px;margin:34px 0 12px;border-bottom:1px solid #333;
     padding-bottom:8px}}
 h3{{color:#fff;margin:0;font-size:16px;font-family:ui-monospace,monospace}}
 .sub{{color:#9aa;margin:0 0 18px;font-size:14px}}
 .ok{{color:#3fbf6f}} .bad{{color:#ffcd28;font-weight:700}}
 .g{{color:#3cdc6e}} .r{{color:#ff6a5c}} .b{{color:#6ec8ff}} .y{{color:#ffcd28}}
 .warn{{color:#e0b050;font-size:14px}}
 .keys{{display:flex;flex-wrap:wrap;gap:12px;margin:0 0 16px;font-size:12.5px;
       font-family:ui-monospace,monospace}}
 .key{{white-space:nowrap}}
 .tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(148px,1fr));
        gap:12px;margin:18px 0 6px}}
 .tile{{background:#181818;border:1px solid #2c2c2c;border-radius:8px;padding:14px 16px}}
 .tile .k{{font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:#8a9a9f}}
 .tile .v{{font-size:25px;font-weight:700;color:#fff;font-variant-numeric:tabular-nums}}
 .tile.red .v{{color:#ff6a5c}} .tile.green .v{{color:#3cdc6e}}
 .tile.blue .v{{color:#6ec8ff}} .tile.yellow .v{{color:#ffcd28}}
 .tile.acc .v{{color:#66c6ff}}
 .meter{{height:8px;border-radius:4px;background:#2a2a2a;overflow:hidden;margin-top:10px}}
 .meter i{{display:block;height:100%;background:linear-gradient(90deg,#3cdc6e,#66c6ff)}}
 table{{border-collapse:collapse;width:100%;font-size:13.5px;margin-top:6px}}
 th,td{{text-align:left;padding:8px 12px;border-bottom:1px solid #262626}}
 th{{color:#8a9a9f;font-size:11px;letter-spacing:.09em;text-transform:uppercase}}
 td.n,th.n{{font-variant-numeric:tabular-nums;font-family:ui-monospace,monospace}}
 .card{{background:#161616;border:1px solid #2a2a2a;border-radius:10px;
       padding:16px 18px;margin:16px 0}}
 .cardhead{{display:flex;flex-wrap:wrap;gap:10px;align-items:center;
           justify-content:space-between;margin-bottom:10px}}
 .chips{{display:flex;flex-wrap:wrap;gap:6px}} .chips.troj{{margin-bottom:12px}}
 .chip{{font-size:11.5px;font-family:ui-monospace,monospace;border:1px solid #3a3a3a;
       border-radius:999px;padding:2px 10px;color:#bbb}}
 .chip.g{{color:#3cdc6e;border-color:#245c37}} .chip.r{{color:#ff6a5c;border-color:#6b2b25}}
 .chip.b{{color:#6ec8ff;border-color:#245066}} .chip.acc{{color:#66c6ff;border-color:#245066}}
 .chip.y{{color:#ffcd28;border-color:#6b5a12;background:#241f08}}
 .trio{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}}
 figure{{margin:0}} figure img{{width:100%;border-radius:6px;display:block;
        border:1px solid #2c2c2c;background:#000}}
 figcaption{{font-size:11.5px;color:#98a6ab;margin-top:6px;font-family:ui-monospace,monospace}}
 figcaption .key{{margin-right:8px}}
 code{{font-family:ui-monospace,monospace;background:#1d1d1d;padding:1px 5px;
      border-radius:3px;font-size:12.5px}}
</style></head><body><div class="wrap">

<h1>B vs C SEM match report</h1>
<p class="sub">Golden <code>{html.escape(report['inputs']['B'])}</code> compared against
  suspect <code>{html.escape(report['inputs']['C'])}</code> ·
  generated {html.escape(report['generated'])} ·
  verdict <span class="{vclass}">{verdict}</span></p>
<div class="keys">{legend}</div>

<div class="tiles">
  <div class="tile acc"><div class="k">Cell accuracy</div><div class="v">{acc:.1%}</div>
    <div class="meter"><i style="width:{max(0.0, min(1.0, acc))*100:.1f}%"></i></div></div>
  <div class="tile"><div class="k">Pairs</div><div class="v">{s['pairs']}</div></div>
  <div class="tile"><div class="k">Matched</div><div class="v">{s['matched']}</div></div>
  <div class="tile {'green' if legacy else 'red'}"><div class="k">Missing from C</div>
    <div class="v">{s['missing_from_c']}</div></div>
  <div class="tile {'red' if legacy else 'green'}"><div class="k">Gained in C</div>
    <div class="v">{s['gained_in_c']}</div></div>
  <div class="tile blue"><div class="k">Connectivity</div><div class="v">{s['connectivity']}</div></div>
  <div class="tile yellow"><div class="k">Trojan regions</div><div class="v">{s['trojan_regions']}</div></div>
</div>
<p class="sub">Cell accuracy is
  <code>matched / (matched + missing + gained + connectivity)</code> —
  {s['matched']} / ({s['matched']} + {s['missing_from_c']} + {s['gained_in_c']}
  + {s['connectivity']}). Mean per-image {s['mean_image_accuracy']:.1%};
  weakest image <code>{html.escape(str(s['worst_image']))}</code>.
  Connectivity splits into {s['merged']} merged, {s['split']} split,
  {s['tangled']} tangled. B holds {s['b_cells']} cells, C holds {s['c_cells']};
  {s['images_with_regions']} of {s['pairs']} images carry a trojan region.</p>
{unpaired}

<h2>Trojan regions by pattern</h2>
<table><thead><tr><th>Label</th><th>Pattern</th><th>What it looks like</th>
  <th class="n">Detected</th></tr></thead><tbody>{pat_rows}</tbody></table>
{truth_block}

<h2>Per-image summary</h2>
<table>
  <thead><tr><th>Image</th><th class="n">B</th><th class="n">C</th><th class="n">Matched</th>
    <th class="n">Missing</th><th class="n">Gained</th><th class="n">Conn.</th>
    <th class="n">Accuracy</th><th>Regions</th></tr></thead>
  <tbody>
  {''.join(
      f"<tr><td>{html.escape(k)}</td><td class='n'>{v['b_cells']}</td>"
      f"<td class='n'>{v['c_cells']}</td><td class='n'>{v['matched']}</td>"
      f"<td class='n {miss_css}'>{v['missing_from_c']}</td>"
      f"<td class='n {gain_css}'>{v['gained_in_c']}</td>"
      f"<td class='n b'>{v['connectivity']}</td>"
      f"<td class='n'>{v['cell_accuracy']:.1%}</td>"
      f"<td class='y'>{html.escape(' '.join(r['label'].replace('Trojan ', 'T') for r in v['regions']))}</td></tr>"
      for k, v in sorted(report['images'].items(),
                         key=lambda kv: kv[1]['cell_accuracy']))}
  </tbody>
</table>

<h2>Every pair · B, C, and B over C</h2>
{''.join(cards)}

</div></body></html>"""
    p = out_dir / "match_report.html"
    p.write_text(page)
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--b-dir", required=True, type=Path)
    ap.add_argument("--c-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--alpha", type=float, default=0.50)
    ap.add_argument("--tint", type=float, default=0.45)
    ap.add_argument("--tolerance", type=int, default=2)
    ap.add_argument("--min-area", type=int, default=24)
    ap.add_argument("--link-cover", type=float, default=0.35)
    ap.add_argument("--group-gap", type=int, default=56)
    ap.add_argument("--legacy-colors", action="store_true")
    ap.add_argument("--thumb-width", type=int, default=460)
    ap.add_argument("--lossless", action="store_true")
    ap.add_argument("--truth", type=Path)
    a = ap.parse_args(argv)
    rep = match_directories(a.b_dir, a.c_dir, a.out,
                            MatchParams(a.alpha, a.tint, a.tolerance,
                                        a.min_area, a.link_cover, a.group_gap,
                                        a.legacy_colors),
                            thumb_width=a.thumb_width, lossless=a.lossless)
    if a.truth:
        score_against_truth(rep, a.truth)
    save_results(a.out, rep)
    print(f"report -> {write_match_report(a.out, rep)}")


if __name__ == "__main__":
    main()
