"""
B ↔ C SEM matcher — cell-level differences between the original golden SEM
(B) and the newly captured suspect SEM (C).

Each image is reduced to its *cells* (8-connected bright regions), the two
cell sets are matched one-to-one by overlap, and whatever fails to match is
what changed:

    GREEN   a cell present in B but MISSING from C   (material removed)
    RED     a cell present in C but MISSING from B   (material gained)
    matched cells are left untinted

Note the convention is the reverse of gds2sem's overlay_compare: there,
green marked extra material. Here red marks gained material, because gained
material is the suspicious direction when screening a returned chip.

The overlay composites **B on top of C**: C is the base, B is blended over
it at `alpha`, then the unmatched cells are tinted and outlined.

Accuracy is scored on cells, not pixels — a Jaccard over the two cell sets:

    accuracy = matched / (matched + missing + gained)

so a perfect reproduction scores 1.0, and every cell that appears on one
side only costs the same regardless of its size. Pixel IoU is reported
alongside it as a secondary, area-weighted view.

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
from PIL import Image

from .imagelib import (binarize, connected_components, dilate, erode,
                       load_gray, resize_to, stem_map)

GREEN = (60, 220, 110)     # in B, missing from C
RED = (240, 70, 60)        # in C, missing from B


@dataclass
class MatchParams:
    """Tuning for the cell match."""
    alpha: float = 0.50        # opacity of B over the C base
    tint: float = 0.45         # opacity of the green/red cell tint
    tolerance: int = 2         # px of slack when testing cell overlap
    min_area: int = 24         # ignore blobs smaller than this (px)
    match_iou: float = 0.25    # IoU above which two cells are "the same cell"


# --------------------------------------------------------------------------
# core matching
# --------------------------------------------------------------------------
def _cells(mask: np.ndarray, min_area: int):
    """[(label_id, bbox, area)] for components at least min_area px."""
    labels, n, boxes = connected_components(mask)
    out = []
    for i in range(1, n + 1):
        x0, y0, x1, y1 = boxes[i - 1]
        area = int((labels[y0:y1, x0:x1] == i).sum())
        if area >= min_area:
            out.append((i, (x0, y0, x1, y1), area))
    return labels, out


def _iou(a_lab, a_id, b_lab, b_id, box, tol):
    """IoU of two labelled cells, computed on the union of their bboxes."""
    x0, y0, x1, y1 = box
    a = a_lab[y0:y1, x0:x1] == a_id
    b = b_lab[y0:y1, x0:x1] == b_id
    if tol:
        inter = int((a & dilate(b, tol)).sum())
    else:
        inter = int((a & b).sum())
    union = int((a | b).sum())
    return inter / union if union else 0.0


def _union_box(p, q):
    return (min(p[0], q[0]), min(p[1], q[1]), max(p[2], q[2]), max(p[3], q[3]))


def _overlaps(p, q, pad=0):
    return not (p[2] + pad <= q[0] or q[2] + pad <= p[0]
                or p[3] + pad <= q[1] or q[3] + pad <= p[1])


def match_masks(b_mask, c_mask, params: MatchParams):
    """Greedy one-to-one cell match. Returns (matched, missing, gained, info)
    where missing/gained are lists of (labels_array, label_id, bbox)."""
    b_lab, b_cells = _cells(b_mask, params.min_area)
    c_lab, c_cells = _cells(c_mask, params.min_area)

    # candidate pairs: only cells whose bboxes come near each other
    pairs = []
    for bi, bbox, _ in b_cells:
        for ci, cbox, _ in c_cells:
            if _overlaps(bbox, cbox, params.tolerance + 1):
                v = _iou(b_lab, bi, c_lab, ci, _union_box(bbox, cbox),
                         params.tolerance)
                if v >= params.match_iou:
                    pairs.append((v, bi, ci))
    pairs.sort(reverse=True)                       # greedy, best overlap first

    used_b, used_c, matched = set(), set(), []
    for v, bi, ci in pairs:
        if bi in used_b or ci in used_c:
            continue
        used_b.add(bi)
        used_c.add(ci)
        matched.append((bi, ci, round(float(v), 4)))

    missing = [(b_lab, i, box) for i, box, _ in b_cells if i not in used_b]
    gained = [(c_lab, i, box) for i, box, _ in c_cells if i not in used_c]
    return matched, missing, gained, {"b_cells": len(b_cells),
                                      "c_cells": len(c_cells)}


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------
def _tint(rgb, mask, color, a):
    if mask.any():
        rgb[mask] = (1 - a) * rgb[mask] + a * np.array(color, np.float64)


def render_overlay(b_img, c_img, missing, gained, params: MatchParams):
    """C as the base, B blended on top, unmatched cells tinted and outlined."""
    base = ((1 - params.alpha) * c_img.astype(np.float64)
            + params.alpha * b_img.astype(np.float64))
    rgb = np.repeat(base[:, :, None], 3, axis=2)

    for cells, color in ((missing, GREEN), (gained, RED)):
        for lab, i, (x0, y0, x1, y1) in cells:
            sub = lab[y0:y1, x0:x1] == i
            view = rgb[y0:y1, x0:x1]
            _tint(view, sub, color, params.tint)
            _tint(view, sub & ~erode(sub, 1), color, 1.0)   # solid outline
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def _b64(img: Image.Image, width: int, lossless: bool = False) -> tuple[str, str]:
    """(base64, mime). SEM grain compresses badly as PNG — a 12-pair report is
    ~8 MB lossless vs well under 1 MB as JPEG — so JPEG is the default and
    lossless is opt-in for pixel-exact review."""
    if width and img.width > width:
        img = img.resize((width, round(img.height * width / img.width)),
                         Image.LANCZOS)
    buf = io.BytesIO()
    if lossless:
        img.convert("RGB").save(buf, "PNG", optimize=True)
        mime = "png"
    else:
        img.convert("RGB").save(buf, "JPEG", quality=86, subsampling=0)
        mime = "jpeg"
    return base64.b64encode(buf.getvalue()).decode(), mime


# --------------------------------------------------------------------------
# directory pass
# --------------------------------------------------------------------------
def match_directories(b_dir, c_dir, out_dir, params: MatchParams = None,
                      save_overlays=True, thumb_width=460,
                      lossless=False, quiet=False):
    """Match every B/C pair, write overlays, and return the report dict."""
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

    images, tot = {}, {"matched": 0, "missing": 0, "gained": 0,
                       "b_cells": 0, "c_cells": 0}
    for stem in stems:
        c_img = load_gray(cmap[stem])
        b_img = resize_to(load_gray(bmap[stem]),
                          (c_img.shape[1], c_img.shape[0]), nearest=False)
        b_mask, c_mask = binarize(b_img, "otsu"), binarize(c_img, "otsu")

        matched, missing, gained, info = match_masks(b_mask, c_mask, params)
        n_m, n_miss, n_gain = len(matched), len(missing), len(gained)
        denom = n_m + n_miss + n_gain
        acc = n_m / denom if denom else 1.0

        inter = int((b_mask & c_mask).sum())
        union = int((b_mask | c_mask).sum())
        pix_iou = inter / union if union else 1.0

        overlay = render_overlay(b_img, c_img, missing, gained, params)
        name = cmap[stem].name
        if save_overlays:
            overlay.save(out_dir / "overlays" / f"match_{stem}.png")

        images[stem] = {
            "b_file": bmap[stem].name, "c_file": name,
            "b_cells": info["b_cells"], "c_cells": info["c_cells"],
            "matched": n_m, "missing_from_c": n_miss, "gained_in_c": n_gain,
            "cell_accuracy": round(acc, 4), "pixel_iou": round(pix_iou, 4),
            "missing_boxes": [[int(v) for v in box] for _, _, box in missing],
            "gained_boxes": [[int(v) for v in box] for _, _, box in gained],
            "_thumbs": (_b64(Image.fromarray(b_img), thumb_width, lossless),
                        _b64(Image.fromarray(c_img), thumb_width, lossless),
                        _b64(overlay, thumb_width, lossless)),
        }
        tot["matched"] += n_m
        tot["missing"] += n_miss
        tot["gained"] += n_gain
        tot["b_cells"] += info["b_cells"]
        tot["c_cells"] += info["c_cells"]
        if not quiet:
            print(f"  {stem:<26} B {info['b_cells']:>4}  C {info['c_cells']:>4}"
                  f"  matched {n_m:>4}  missing {n_miss:>3}  gained {n_gain:>3}"
                  f"  acc {acc:>6.1%}")

    d = tot["matched"] + tot["missing"] + tot["gained"]
    overall = tot["matched"] / d if d else 1.0
    accs = [v["cell_accuracy"] for v in images.values()]

    report = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "inputs": {"B": str(b_dir), "C": str(c_dir)},
        "params": asdict(params),
        "summary": {
            "pairs": len(stems),
            "b_cells": tot["b_cells"], "c_cells": tot["c_cells"],
            "matched": tot["matched"],
            "missing_from_c": tot["missing"], "gained_in_c": tot["gained"],
            "overall_accuracy": round(overall, 4),
            "mean_image_accuracy": round(float(np.mean(accs)), 4),
            "worst_image": min(images, key=lambda k: images[k]["cell_accuracy"]),
            "unpaired_in_b": only_b, "unpaired_in_c": only_c,
        },
        "images": images,
    }

    slim = {k: {kk: vv for kk, vv in v.items() if kk != "_thumbs"}
            for k, v in images.items()}
    (out_dir / "match_results.json").write_text(
        json.dumps({**report, "images": slim}, indent=2))
    if not quiet:
        s = report["summary"]
        print(f"\npairs {s['pairs']}   matched {s['matched']}   "
              f"missing {s['missing_from_c']}   gained {s['gained_in_c']}")
        print(f"overall cell accuracy {s['overall_accuracy']:.2%}   "
              f"mean per-image {s['mean_image_accuracy']:.2%}")
        if only_b or only_c:
            print(f"unpaired — B only: {len(only_b)}, C only: {len(only_c)}")
    return report


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def write_match_report(out_dir, report: dict) -> Path:
    out_dir = Path(out_dir)
    s = report["summary"]
    acc = s["overall_accuracy"]
    verdict = ("clean" if s["missing_from_c"] == 0 and s["gained_in_c"] == 0
               else "differences found")
    vclass = "ok" if verdict == "clean" else "bad"

    rows = []
    for stem, v in sorted(report["images"].items(),
                          key=lambda kv: kv[1]["cell_accuracy"]):
        (b64b, mb), (b64c, mc), (b64o, mo) = v["_thumbs"]
        chips = (f"<span class='chip green'>{v['missing_from_c']} missing "
                 f"from C</span><span class='chip red'>{v['gained_in_c']} "
                 f"gained in C</span><span class='chip'>{v['matched']} "
                 f"matched</span>")
        rows.append(f"""
      <section class="card">
        <div class="cardhead">
          <h3>{html.escape(stem)}</h3>
          <div class="chips">{chips}
            <span class="chip acc">{v['cell_accuracy']:.1%} cell accuracy</span>
            <span class="chip">{v['pixel_iou']:.1%} pixel IoU</span>
          </div>
        </div>
        <div class="trio">
          <figure><img src="data:image/{mb};base64,{b64b}" alt="B, golden SEM">
            <figcaption>B · golden — {html.escape(v['b_file'])} · {v['b_cells']} cells</figcaption></figure>
          <figure><img src="data:image/{mc};base64,{b64c}" alt="C, suspect SEM">
            <figcaption>C · suspect — {html.escape(v['c_file'])} · {v['c_cells']} cells</figcaption></figure>
          <figure><img src="data:image/{mo};base64,{b64o}" alt="B over C with differences highlighted">
            <figcaption><b>B over C</b> — <span class="g">green missing from C</span>,
              <span class="r">red gained in C</span></figcaption></figure>
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
 h2{{color:#fff;font-size:18px;margin:34px 0 12px;
     border-bottom:1px solid #333;padding-bottom:8px}}
 h3{{color:#fff;margin:0;font-size:16px;font-family:ui-monospace,monospace}}
 .sub{{color:#9aa;margin:0 0 20px;font-size:14px}}
 .ok{{color:#3fbf6f}} .bad{{color:#ff7a6e;font-weight:700}}
 .g{{color:#3cdc6e}} .r{{color:#ff6a5c}}
 .warn{{color:#e0b050;font-size:14px}}
 .tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
        gap:12px;margin:18px 0 6px}}
 .tile{{background:#181818;border:1px solid #2c2c2c;border-radius:8px;padding:14px 16px}}
 .tile .k{{font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:#8a9a9f}}
 .tile .v{{font-size:26px;font-weight:700;color:#fff;font-variant-numeric:tabular-nums}}
 .tile.green .v{{color:#3cdc6e}} .tile.red .v{{color:#ff6a5c}}
 .tile.acc .v{{color:#66c6ff}}
 .meter{{height:8px;border-radius:4px;background:#2a2a2a;overflow:hidden;margin-top:10px}}
 .meter i{{display:block;height:100%;background:linear-gradient(90deg,#3cdc6e,#66c6ff)}}
 table{{border-collapse:collapse;width:100%;font-size:13.5px;margin-top:6px}}
 th,td{{text-align:left;padding:8px 12px;border-bottom:1px solid #262626}}
 th{{color:#8a9a9f;font-size:11px;letter-spacing:.09em;text-transform:uppercase}}
 td.n{{font-variant-numeric:tabular-nums;font-family:ui-monospace,monospace}}
 .card{{background:#161616;border:1px solid #2a2a2a;border-radius:10px;
       padding:16px 18px;margin:16px 0}}
 .cardhead{{display:flex;flex-wrap:wrap;gap:10px;align-items:center;
           justify-content:space-between;margin-bottom:12px}}
 .chips{{display:flex;flex-wrap:wrap;gap:6px}}
 .chip{{font-size:11.5px;font-family:ui-monospace,monospace;border:1px solid #3a3a3a;
       border-radius:999px;padding:2px 10px;color:#bbb}}
 .chip.green{{color:#3cdc6e;border-color:#245c37}}
 .chip.red{{color:#ff6a5c;border-color:#6b2b25}}
 .chip.acc{{color:#66c6ff;border-color:#245066}}
 .trio{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}}
 figure{{margin:0}} figure img{{width:100%;border-radius:6px;display:block;
        border:1px solid #2c2c2c;background:#000}}
 figcaption{{font-size:12px;color:#98a6ab;margin-top:6px;
            font-family:ui-monospace,monospace}}
 code{{font-family:ui-monospace,monospace;background:#1d1d1d;padding:1px 5px;
      border-radius:3px;font-size:12.5px}}
</style></head><body><div class="wrap">

<h1>B vs C SEM match report</h1>
<p class="sub">Golden <code>{html.escape(report['inputs']['B'])}</code> compared against
  suspect <code>{html.escape(report['inputs']['C'])}</code> ·
  generated {html.escape(report['generated'])} ·
  verdict <span class="{vclass}">{verdict}</span></p>

<div class="tiles">
  <div class="tile acc"><div class="k">Cell accuracy</div><div class="v">{acc:.1%}</div>
    <div class="meter"><i style="width:{max(0.0, min(1.0, acc))*100:.1f}%"></i></div></div>
  <div class="tile"><div class="k">Pairs</div><div class="v">{s['pairs']}</div></div>
  <div class="tile"><div class="k">Matched cells</div><div class="v">{s['matched']}</div></div>
  <div class="tile green"><div class="k">Missing from C</div><div class="v">{s['missing_from_c']}</div></div>
  <div class="tile red"><div class="k">Gained in C</div><div class="v">{s['gained_in_c']}</div></div>
</div>
<p class="sub">Cell accuracy is
  <code>matched / (matched + missing + gained)</code> over every cell in the set —
  {s['matched']} / ({s['matched']} + {s['missing_from_c']} + {s['gained_in_c']}).
  Mean per-image accuracy is {s['mean_image_accuracy']:.1%};
  the weakest image is <code>{html.escape(str(s['worst_image']))}</code>.
  B holds {s['b_cells']} cells, C holds {s['c_cells']}.</p>
{unpaired}

<h2>Per-image summary</h2>
<table>
  <thead><tr><th>Image</th><th>B cells</th><th>C cells</th><th>Matched</th>
    <th>Missing from C</th><th>Gained in C</th><th>Cell accuracy</th><th>Pixel IoU</th></tr></thead>
  <tbody>
  {''.join(
      f"<tr><td>{html.escape(k)}</td><td class='n'>{v['b_cells']}</td>"
      f"<td class='n'>{v['c_cells']}</td><td class='n'>{v['matched']}</td>"
      f"<td class='n g'>{v['missing_from_c']}</td>"
      f"<td class='n r'>{v['gained_in_c']}</td>"
      f"<td class='n'>{v['cell_accuracy']:.1%}</td>"
      f"<td class='n'>{v['pixel_iou']:.1%}</td></tr>"
      for k, v in sorted(report['images'].items(),
                         key=lambda kv: kv[1]['cell_accuracy']))}
  </tbody>
</table>

<h2>Every pair · B, C, and B over C</h2>
{''.join(rows)}

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
    ap.add_argument("--match-iou", type=float, default=0.25)
    ap.add_argument("--thumb-width", type=int, default=460)
    ap.add_argument("--lossless", action="store_true")
    a = ap.parse_args(argv)
    rep = match_directories(a.b_dir, a.c_dir, a.out,
                            MatchParams(a.alpha, a.tint, a.tolerance,
                                        a.min_area, a.match_iou),
                            thumb_width=a.thumb_width,
                            lossless=a.lossless)
    print(f"report -> {write_match_report(a.out, rep)}")


if __name__ == "__main__":
    main()
