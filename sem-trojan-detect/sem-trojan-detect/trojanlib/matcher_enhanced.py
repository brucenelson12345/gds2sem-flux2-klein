"""
Enhanced B ↔ C matcher — whole-cell events boxed, partial changes tinted.

The plain `trojanlib.matcher` answers "which cells changed". This one adds
the distinction that matters when you are looking at an overlay and deciding
what to trust: **was a whole cell affected, or only part of one?**

    WHOLE CELL           tinted, outlined, boxed and captioned
      missing    red             a cell in B with no counterpart in C
      addition   green           a cell in C with no counterpart in B
      join       blue            n cells in B fused into 1 in C
      split      yellow-orange   1 cell in B broken into n in C

    PARCEL OF A CELL     tinted only, no box
      a matched pair whose shapes disagree — the cell got shorter, longer,
      wider or notched. Same red (material lost) and green (material gained)
      as the whole-cell cases, because it is the same kind of change; the
      absence of a box is what says "part of a cell, not all of it".

That partial case is invisible to the plain matcher: a shortened cell still
links one-to-one and is reported as matched. Here the two shapes are
differenced inside the pair, so the change shows up.

No trojan-region grouping here — that is the plain matcher's job, and its
yellow region boxes would collide with the yellow-orange split boxes. This
matcher is purely per-cell.

Importable:
    from trojanlib.matcher_enhanced import match_directories_enhanced

CLI:
    python -m trojanlib.matcher_enhanced --b-dir B --c-dir C --out run
"""
from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .imagelib import (binarize, connected_components, dilate, erode,
                       load_gray, resize_to, stem_map)
from .matcher import _b64, _bbox_near, _cells, _union_box, _UF

# ---- the four whole-cell classes, and the two parcel classes ---------------
CLASSES = {
    "missing":  {"color": (230, 60, 50),   "caption": "missing",
                 "desc": "a cell in B with no counterpart in C"},
    "addition": {"color": (55, 215, 105),  "caption": "addition",
                 "desc": "a cell in C with no counterpart in B"},
    "join":     {"color": (70, 150, 255),  "caption": "join",
                 "desc": "cells fused together — should be separated"},
    "split":    {"color": (255, 168, 32),  "caption": "split",
                 "desc": "a cell broken apart — should be connected"},
}
# parcels reuse the missing/addition hues but are never boxed
PARCEL_LOST = CLASSES["missing"]["color"]
PARCEL_GAINED = CLASSES["addition"]["color"]


@dataclass
class EnhancedParams:
    alpha: float = 0.50          # opacity of B over the C base
    tint: float = 0.45           # opacity of the whole-cell fill
    parcel_tint: float = 0.55    # parcels are small: tint them a touch harder
    tolerance: int = 2           # px of slack when comparing shapes
    min_area: int = 24           # ignore cells smaller than this
    min_parcel_area: int = 40    # ignore shape disagreement smaller than this
    link_cover: float = 0.35     # containment above which two cells link
    box_width: int = 2
    font_size: int = 13


def _font(size):
    try:
        return ImageFont.load_default(size=size)     # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def _open(mask, r=1):
    """Morphological opening — drops the 1-2 px slivers that two independent
    renders always disagree on along every edge."""
    return dilate(erode(mask, r), r)


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------
def analyse_enhanced(b_mask, c_mask, params: EnhancedParams):
    """Returns (events, parcels, b_lab, c_lab, counts, n_b, n_c).

    events  — whole-cell changes: {kind, bbox, b_ids, c_ids}
    parcels — partial changes inside a matched pair:
              {kind: lost|gained, bbox, area, b_id, c_id}
    """
    b_lab, b_cells = _cells(b_mask, params.min_area)
    c_lab, c_cells = _cells(c_mask, params.min_area)
    tol = params.tolerance

    uf = _UF()
    for i in b_cells:
        uf.find(("b", i))
    for j in c_cells:
        uf.find(("c", j))

    for i, bi in b_cells.items():
        for j, cj in c_cells.items():
            if not _bbox_near(bi["bbox"], cj["bbox"], tol + 1):
                continue
            x0, y0, x1, y1 = _union_box(bi["bbox"], cj["bbox"])
            a = b_lab[y0:y1, x0:x1] == i
            c = c_lab[y0:y1, x0:x1] == j
            inter = int((a & (dilate(c, tol) if tol else c)).sum())
            if inter and inter / max(1, min(bi["area"], cj["area"])) >= params.link_cover:
                uf.union(("b", i), ("c", j))

    events, parcels = [], []
    counts = {k: 0 for k in CLASSES}
    counts.update({"clean": 0, "parcel_pairs": 0,
                   "parcel_lost": 0, "parcel_gained": 0})

    for nodes in uf.groups():
        b_ids = sorted(n[1] for n in nodes if n[0] == "b")
        c_ids = sorted(n[1] for n in nodes if n[0] == "c")
        nb, nc = len(b_ids), len(c_ids)

        boxes = ([b_cells[i]["bbox"] for i in b_ids]
                 + [c_cells[j]["bbox"] for j in c_ids])
        bbox = boxes[0]
        for bx in boxes[1:]:
            bbox = _union_box(bbox, bx)

        # ---- whole-cell events ------------------------------------------
        if nb >= 1 and nc == 0:
            kind = "missing"
        elif nb == 0 and nc >= 1:
            kind = "addition"
        elif nb >= 2 and nc == 1:
            kind = "join"
        elif nb == 1 and nc >= 2:
            kind = "split"
        elif nb >= 2 and nc >= 2:
            # rewired: call it by the direction it moved in
            kind = "join" if nc < nb else "split"
        else:
            kind = None                                  # 1 <-> 1, examine it

        if kind is not None:
            events.append({"kind": kind, "bbox": [int(v) for v in bbox],
                           "b_ids": b_ids, "c_ids": c_ids,
                           "caption": CLASSES[kind]["caption"]})
            counts[kind] += 1
            continue

        # ---- a matched pair: difference the two shapes -------------------
        i, j = b_ids[0], c_ids[0]
        px0, py0, px1, py1 = bbox
        pad = tol + 2
        px0, py0 = max(0, px0 - pad), max(0, py0 - pad)
        px1 = min(b_mask.shape[1], px1 + pad)
        py1 = min(b_mask.shape[0], py1 + pad)
        a = b_lab[py0:py1, px0:px1] == i
        c = c_lab[py0:py1, px0:px1] == j
        lost = _open(a & ~dilate(c, tol))        # in B, gone from C
        gained = _open(c & ~dilate(a, tol))      # in C, not in B

        hit = False
        for sub, pkind in ((lost, "lost"), (gained, "gained")):
            area = int(sub.sum())
            if area < params.min_parcel_area:
                continue
            ys, xs = np.nonzero(sub)
            parcels.append({
                "kind": pkind, "area": area, "b_id": i, "c_id": j,
                "bbox": [int(px0 + xs.min()), int(py0 + ys.min()),
                         int(px0 + xs.max()) + 1, int(py0 + ys.max()) + 1],
                "_mask_origin": (px0, py0), "_mask": sub,
            })
            counts["parcel_" + pkind] += 1
            hit = True
        counts["parcel_pairs" if hit else "clean"] += 1

    return events, parcels, b_lab, c_lab, counts, len(b_cells), len(c_cells)


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------
def _tint(rgb, mask, color, a):
    if mask.any():
        rgb[mask] = (1 - a) * rgb[mask] + a * np.array(color, np.float64)


def render_enhanced(b_img, c_img, events, parcels, b_lab, c_lab,
                    params: EnhancedParams):
    """C as the base, B blended over it. Parcels tinted; whole-cell events
    tinted, outlined, boxed and captioned in their class colour."""
    base = ((1 - params.alpha) * c_img.astype(np.float64)
            + params.alpha * b_img.astype(np.float64))
    rgb = np.repeat(base[:, :, None], 3, axis=2)

    # parcels first, so a whole-cell box drawn later always reads on top
    for p in parcels:
        col = PARCEL_LOST if p["kind"] == "lost" else PARCEL_GAINED
        ox, oy = p["_mask_origin"]
        m = p["_mask"]
        view = rgb[oy:oy + m.shape[0], ox:ox + m.shape[1]]
        _tint(view, m, col, params.parcel_tint)

    for ev in events:
        col = CLASSES[ev["kind"]]["color"]
        x0, y0, x1, y1 = ev["bbox"]
        sub = np.zeros((y1 - y0, x1 - x0), bool)
        for i in ev["b_ids"]:
            sub |= b_lab[y0:y1, x0:x1] == i
        for j in ev["c_ids"]:
            sub |= c_lab[y0:y1, x0:x1] == j
        view = rgb[y0:y1, x0:x1]
        _tint(view, sub, col, params.tint)
        _tint(view, sub & ~erode(sub, 1), col, 1.0)

    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    font = _font(params.font_size)
    W, H = img.size
    for ev in events:
        col = CLASSES[ev["kind"]]["color"]
        cap = ev["caption"]
        x0, y0, x1, y1 = ev["bbox"]
        pad = 3
        bx0, by0 = max(0, x0 - pad), max(0, y0 - pad)
        bx1, by1 = min(W - 1, x1 + pad), min(H - 1, y1 + pad)
        d.rectangle([bx0, by0, bx1, by1], outline=col, width=params.box_width)

        # caption in the box colour, on a dark plate so it reads over SEM grain
        try:
            tb = d.textbbox((0, 0), cap, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
        except Exception:                                      # noqa: BLE001
            tw, th = 7 * len(cap), 11
        ty = by0 - (th + 4)
        if ty < 0:                                   # no room above: go inside
            ty = by0 + 1
        tx = min(bx0, W - (tw + 6))
        d.rectangle([tx, ty, tx + tw + 5, ty + th + 3], fill=(0, 0, 0))
        d.text((tx + 3, ty + 1), cap, fill=col, font=font)
    return img


# --------------------------------------------------------------------------
# directory pass
# --------------------------------------------------------------------------
def match_directories_enhanced(b_dir, c_dir, out_dir,
                               params: EnhancedParams = None,
                               save_overlays=True, thumb_width=460,
                               lossless=False, quiet=False):
    params = params or EnhancedParams()
    b_dir, c_dir, out_dir = Path(b_dir), Path(c_dir), Path(out_dir)
    bmap, cmap = stem_map(b_dir), stem_map(c_dir)
    stems = sorted(set(bmap) & set(cmap))
    if not stems:
        raise SystemExit(f"no matching filenames between {b_dir} and {c_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if save_overlays:
        (out_dir / "overlays").mkdir(exist_ok=True)

    images = {}
    tot = {k: 0 for k in CLASSES}
    tot.update({"clean": 0, "parcel_pairs": 0, "parcel_lost": 0,
                "parcel_gained": 0, "b_cells": 0, "c_cells": 0})

    for stem in stems:
        c_img = load_gray(cmap[stem])
        b_img = resize_to(load_gray(bmap[stem]),
                          (c_img.shape[1], c_img.shape[0]), nearest=False)
        b_mask, c_mask = binarize(b_img, "otsu"), binarize(c_img, "otsu")

        events, parcels, b_lab, c_lab, counts, nb, nc = analyse_enhanced(
            b_mask, c_mask, params)
        overlay = render_enhanced(b_img, c_img, events, parcels,
                                  b_lab, c_lab, params)
        if save_overlays:
            overlay.save(out_dir / "overlays" / f"enh_{stem}.png")

        whole = sum(counts[k] for k in CLASSES)
        denom = counts["clean"] + counts["parcel_pairs"] + whole
        acc = counts["clean"] / denom if denom else 1.0

        images[stem] = {
            "b_file": bmap[stem].name, "c_file": cmap[stem].name,
            "size": [int(c_img.shape[1]), int(c_img.shape[0])],
            "b_cells": nb, "c_cells": nc,
            **{k: counts[k] for k in CLASSES},
            "clean": counts["clean"],
            "parcel_pairs": counts["parcel_pairs"],
            "parcel_lost": counts["parcel_lost"],
            "parcel_gained": counts["parcel_gained"],
            "whole_cell_events": whole,
            "cell_accuracy": round(acc, 4),
            "events": [{k: v for k, v in e.items()} for e in events],
            "parcels": [{k: v for k, v in p.items()
                         if not k.startswith("_")} for p in parcels],
            "_thumbs": (_b64(Image.fromarray(b_img), thumb_width, lossless),
                        _b64(Image.fromarray(c_img), thumb_width, lossless),
                        _b64(overlay, thumb_width, lossless)),
        }
        for k in list(CLASSES) + ["clean", "parcel_pairs", "parcel_lost",
                                  "parcel_gained"]:
            tot[k] += counts[k]
        tot["b_cells"] += nb
        tot["c_cells"] += nc

        if not quiet:
            bits = " ".join(f"{CLASSES[k]['caption']} {counts[k]}"
                            for k in CLASSES if counts[k])
            par = (f"  parcels {counts['parcel_lost']}L/"
                   f"{counts['parcel_gained']}G" if counts["parcel_pairs"] else "")
            print(f"  {stem:<24} B{nb:>4} C{nc:>4}  clean {counts['clean']:>4}"
                  f"  acc {acc:>6.1%}  {bits}{par}")

    whole_tot = sum(tot[k] for k in CLASSES)
    denom = tot["clean"] + tot["parcel_pairs"] + whole_tot
    overall = tot["clean"] / denom if denom else 1.0
    accs = [v["cell_accuracy"] for v in images.values()]

    report = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "matcher": "enhanced",
        "inputs": {"B": str(b_dir), "C": str(c_dir)},
        "params": asdict(params),
        "classes": {k: {"caption": v["caption"], "color": list(v["color"]),
                        "description": v["desc"]} for k, v in CLASSES.items()},
        "summary": {
            "pairs": len(stems),
            "b_cells": tot["b_cells"], "c_cells": tot["c_cells"],
            "clean": tot["clean"],
            **{k: tot[k] for k in CLASSES},
            "whole_cell_events": whole_tot,
            "parcel_pairs": tot["parcel_pairs"],
            "parcel_lost": tot["parcel_lost"],
            "parcel_gained": tot["parcel_gained"],
            "overall_accuracy": round(overall, 4),
            "mean_image_accuracy": round(float(np.mean(accs)), 4),
            "worst_image": min(images, key=lambda k: images[k]["cell_accuracy"]),
        },
        "images": images,
    }

    if not quiet:
        s = report["summary"]
        print(f"\npairs {s['pairs']}   clean {s['clean']}   "
              + "   ".join(f"{CLASSES[k]['caption']} {s[k]}" for k in CLASSES))
        print(f"partial-cell changes: {s['parcel_pairs']} pair(s) "
              f"({s['parcel_lost']} lost, {s['parcel_gained']} gained)")
        print(f"overall cell accuracy {s['overall_accuracy']:.2%}   "
              f"mean per-image {s['mean_image_accuracy']:.2%}")
    return report


def save_results_enhanced(out_dir, report):
    slim = {k: {kk: vv for kk, vv in v.items() if kk != "_thumbs"}
            for k, v in report["images"].items()}
    p = Path(out_dir) / "enhanced_results.json"
    p.write_text(json.dumps({**report, "images": slim}, indent=2))
    return p


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def _hex(c):
    return "#%02x%02x%02x" % tuple(c)


def write_enhanced_report(out_dir, report: dict) -> Path:
    out_dir = Path(out_dir)
    s = report["summary"]
    acc = s["overall_accuracy"]
    whole = s["whole_cell_events"]
    verdict = "clean" if whole == 0 and s["parcel_pairs"] == 0 else (
        f"{whole} whole-cell event{'s' if whole != 1 else ''}"
        + (f", {s['parcel_pairs']} partial" if s["parcel_pairs"] else ""))

    legend = "".join(
        f"<span class='key' style='color:{_hex(v['color'])}'>&#9632; "
        f"{html.escape(v['caption'])} — {html.escape(v['desc'])}</span>"
        for v in CLASSES.values())
    legend += ("<span class='key' style='color:#9aa'>&#9642; tint with no box "
               "— only part of a cell changed</span>")

    tiles = "".join(
        f"<div class='tile'><div class='k'>{html.escape(v['caption'])}</div>"
        f"<div class='v' style='color:{_hex(v['color'])}'>{s[k]}</div></div>"
        for k, v in CLASSES.items())

    cards = []
    for stem, v in sorted(report["images"].items(),
                          key=lambda kv: kv[1]["cell_accuracy"]):
        (b64b, mb), (b64c, mc), (b64o, mo) = v["_thumbs"]
        chips = "".join(
            f"<span class='chip' style='color:{_hex(CLASSES[k]['color'])};"
            f"border-color:{_hex(CLASSES[k]['color'])}44'>"
            f"{v[k]} {html.escape(CLASSES[k]['caption'])}</span>"
            for k in CLASSES if v[k])
        if v["parcel_pairs"]:
            chips += (f"<span class='chip'>{v['parcel_lost']} partial lost · "
                      f"{v['parcel_gained']} partial gained</span>")
        chips += (f"<span class='chip acc'>{v['cell_accuracy']:.1%} clean</span>")
        cards.append(f"""
      <section class="card">
        <div class="cardhead"><h3>{html.escape(stem)}</h3>
          <div class="chips">{chips}</div></div>
        <div class="trio">
          <figure><img src="data:image/{mb};base64,{b64b}" alt="B golden SEM">
            <figcaption>B · golden — {html.escape(v['b_file'])} · {v['b_cells']} cells</figcaption></figure>
          <figure><img src="data:image/{mc};base64,{b64c}" alt="C suspect SEM">
            <figcaption>C · suspect — {html.escape(v['c_file'])} · {v['c_cells']} cells</figcaption></figure>
          <figure><img src="data:image/{mo};base64,{b64o}" alt="B over C, differences boxed">
            <figcaption><b>B over C</b> — boxed = whole cell, tint only = partial</figcaption></figure>
        </div>
      </section>""")

    rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td class='n'>{v['b_cells']}</td>"
        f"<td class='n'>{v['c_cells']}</td><td class='n'>{v['clean']}</td>"
        + "".join(f"<td class='n' style='color:{_hex(CLASSES[c]['color'])}'>"
                  f"{v[c]}</td>" for c in CLASSES)
        + f"<td class='n'>{v['parcel_lost']}/{v['parcel_gained']}</td>"
        f"<td class='n'>{v['cell_accuracy']:.1%}</td></tr>"
        for k, v in sorted(report["images"].items(),
                           key=lambda kv: kv[1]["cell_accuracy"]))
    head = "".join(f"<th class='n'>{html.escape(v['caption'])}</th>"
                   for v in CLASSES.values())

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Enhanced B vs C match report</title><style>
 :root{{color-scheme:dark}}
 body{{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:0;
      background:#111;color:#ddd;line-height:1.55}}
 .wrap{{max-width:1180px;margin:0 auto;padding:28px 22px 72px}}
 h1{{color:#fff;margin:0 0 6px;font-size:26px}}
 h2{{color:#fff;font-size:18px;margin:34px 0 12px;border-bottom:1px solid #333;
     padding-bottom:8px}}
 h3{{color:#fff;margin:0;font-size:16px;font-family:ui-monospace,monospace}}
 .sub{{color:#9aa;margin:0 0 18px;font-size:14px}}
 .keys{{display:flex;flex-direction:column;gap:4px;margin:0 0 18px;
       font-size:12.5px;font-family:ui-monospace,monospace}}
 .tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
        gap:12px;margin:18px 0 6px}}
 .tile{{background:#181818;border:1px solid #2c2c2c;border-radius:8px;padding:14px 16px}}
 .tile .k{{font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:#8a9a9f}}
 .tile .v{{font-size:25px;font-weight:700;color:#fff;font-variant-numeric:tabular-nums}}
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
 .chips{{display:flex;flex-wrap:wrap;gap:6px}}
 .chip{{font-size:11.5px;font-family:ui-monospace,monospace;border:1px solid #3a3a3a;
       border-radius:999px;padding:2px 10px;color:#bbb}}
 .chip.acc{{color:#66c6ff;border-color:#245066}}
 .trio{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}}
 figure{{margin:0}} figure img{{width:100%;border-radius:6px;display:block;
        border:1px solid #2c2c2c;background:#000}}
 figcaption{{font-size:11.5px;color:#98a6ab;margin-top:6px;font-family:ui-monospace,monospace}}
 code{{font-family:ui-monospace,monospace;background:#1d1d1d;padding:1px 5px;
      border-radius:3px;font-size:12.5px}}
</style></head><body><div class="wrap">

<h1>Enhanced B vs C match report</h1>
<p class="sub">Golden <code>{html.escape(report['inputs']['B'])}</code> vs suspect
  <code>{html.escape(report['inputs']['C'])}</code> · generated
  {html.escape(report['generated'])} · verdict <b>{html.escape(verdict)}</b></p>
<div class="keys">{legend}</div>

<div class="tiles">
  <div class="tile acc"><div class="k">Clean cells</div><div class="v">{acc:.1%}</div>
    <div class="meter"><i style="width:{max(0.0, min(1.0, acc))*100:.1f}%"></i></div></div>
  <div class="tile"><div class="k">Pairs</div><div class="v">{s['pairs']}</div></div>
  {tiles}
  <div class="tile"><div class="k">Partial cells</div><div class="v">{s['parcel_pairs']}</div></div>
</div>
<p class="sub">A boxed, captioned cell is a <b>whole-cell</b> event; a tinted
  region with no box is <b>part</b> of a cell that changed shape — the cell is
  still there, it just got shorter, longer, wider or notched. Accuracy is
  <code>clean / (clean + partial + whole-cell events)</code> —
  {s['clean']} / ({s['clean']} + {s['parcel_pairs']} + {whole}). Mean per-image
  {s['mean_image_accuracy']:.1%}; weakest <code>{html.escape(str(s['worst_image']))}</code>.
  B holds {s['b_cells']} cells, C holds {s['c_cells']}.</p>

<h2>Per-image summary</h2>
<table><thead><tr><th>Image</th><th class="n">B</th><th class="n">C</th>
  <th class="n">Clean</th>{head}<th class="n">Partial L/G</th>
  <th class="n">Accuracy</th></tr></thead><tbody>{rows}</tbody></table>

<h2>Every pair · B, C, and B over C</h2>
{''.join(cards)}

</div></body></html>"""
    p = out_dir / "enhanced_report.html"
    p.write_text(page)
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--b-dir", required=True, type=Path)
    ap.add_argument("--c-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--alpha", type=float, default=0.50)
    ap.add_argument("--tint", type=float, default=0.45)
    ap.add_argument("--parcel-tint", type=float, default=0.55)
    ap.add_argument("--tolerance", type=int, default=2)
    ap.add_argument("--min-area", type=int, default=24)
    ap.add_argument("--min-parcel-area", type=int, default=40)
    ap.add_argument("--link-cover", type=float, default=0.35)
    ap.add_argument("--thumb-width", type=int, default=460)
    ap.add_argument("--lossless", action="store_true")
    a = ap.parse_args(argv)
    rep = match_directories_enhanced(
        a.b_dir, a.c_dir, a.out,
        EnhancedParams(a.alpha, a.tint, a.parcel_tint, a.tolerance,
                       a.min_area, a.min_parcel_area, a.link_cover),
        thumb_width=a.thumb_width, lossless=a.lossless)
    save_results_enhanced(a.out, rep)
    print(f"report -> {write_enhanced_report(a.out, rep)}")


if __name__ == "__main__":
    main()
