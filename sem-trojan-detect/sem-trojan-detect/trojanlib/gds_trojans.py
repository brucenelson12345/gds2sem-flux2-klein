"""
GDS-level trojan injection — five trojan patterns (Trojan A … Trojan E)
stamped into a directory of GDS layout images.

Where `trojanlib.inject` tampers *SEM* images with single-feature defects,
this module works one level up, on the **layout**, and produces a
*trojan region*: a small group of neighbouring cells in which one or more
cells have been added, removed, merged or split. That group — modified
cells plus the untouched neighbours around them — is what gets labelled as
the trojan, which is the unit an analyst (or a detector) is actually asked
to find.

The five patterns, and the primitive edits each is built from:

    A  inserted_cluster   2-3 new cells placed in the group's whitespace
    B  depopulated        1-2 existing cells deleted from the group
    C  merged_pair        two adjacent cells bridged into one
    D  severed_net        one cell cut into two separated pieces
    E  rerouted_block     a mixture: one added, one removed, one pair bridged

They are deliberately one-to-one with what a B-vs-C comparison can
distinguish downstream: additions, removals, and the two connectivity
changes (something that should be separated, something that should be
connected). `trojanlib.matcher` re-derives the same five labels from the
image difference alone, so the injector doubles as ground truth.

Typical use — build a labelled training set from 70 clean layouts:

    python -m trojanlib.gds_trojans --gds-dir A/train --out-dir A_trojan \\
        --rate 0.7 --max-per-image 2

Then render A_trojan through gds2sem to get the tampered SEM set, and
compare it against the SEM rendered from the clean layouts.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
from PIL import Image

from .imagelib import binarize, connected_components, dilate, load_gray, stem_map
from .patterns import find_whitespace_box, orientation_of, rect

PATTERNS = {
    "A": ("inserted_cluster", "extra cells inserted into the group's whitespace"),
    "B": ("depopulated", "existing cells deleted from the group"),
    "C": ("merged_pair", "two adjacent cells bridged into one"),
    "D": ("severed_net", "one cell cut into two separated pieces"),
    "E": ("rerouted_block", "mixed: a cell added, a cell removed, a pair bridged"),
}
ALL_PATTERNS = list("ABCDE")


def label_of(key: str) -> str:
    return f"Trojan {key}"


@dataclass
class Edit:
    kind: str                      # add | remove | merge | split
    bbox: list                     # [x0, y0, x1, y1]


@dataclass
class TrojanRegion:
    pattern: str                   # A..E
    label: str                     # "Trojan A"
    name: str                      # inserted_cluster, ...
    bbox: list                     # region box: modified cells + neighbours
    edits: list = field(default_factory=list)
    member_boxes: list = field(default_factory=list)


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------
def _centre(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _dist(a, b):
    (ax, ay), (bx, by) = _centre(a), _centre(b)
    return float(np.hypot(ax - bx, ay - by))


def _union(boxes):
    xs0 = min(b[0] for b in boxes); ys0 = min(b[1] for b in boxes)
    xs1 = max(b[2] for b in boxes); ys1 = max(b[3] for b in boxes)
    return [int(xs0), int(ys0), int(xs1), int(ys1)]


def _pad(box, p, shape):
    h, w = shape
    return [max(0, box[0] - p), max(0, box[1] - p),
            min(w, box[2] + p), min(h, box[3] + p)]


def _cells(mask, min_area):
    """[(label_id, bbox)] for components of at least min_area px."""
    labels, n, boxes = connected_components(mask)
    out = []
    for i in range(1, n + 1):
        x0, y0, x1, y1 = boxes[i - 1]
        if int((labels[y0:y1, x0:x1] == i).sum()) >= min_area:
            out.append((i, [int(x0), int(y0), int(x1), int(y1)]))
    return labels, out


def _neighbourhood(cells, rng, size):
    """A seed cell plus its nearest neighbours — the trojan's host group."""
    if len(cells) < 2:
        return []
    seed = cells[int(rng.integers(len(cells)))]
    ranked = sorted((c for c in cells if c[0] != seed[0]),
                    key=lambda c: _dist(seed[1], c[1]))
    return [seed] + ranked[:max(0, size - 1)]


# --------------------------------------------------------------------------
# primitive edits (operate on a boolean layout mask, in place)
# --------------------------------------------------------------------------
def _add_cell(mask, group, rng, o):
    """New cell in whitespace near the group; returns its bbox or None."""
    gbox = _union([b for _, b in group])
    long_side = int(rng.integers(26, 56))
    short = int(rng.integers(9, 15))
    bw, bh = (short, long_side) if o == "vertical" else (long_side, short)
    h, w = mask.shape
    reach = 38          # keep insertions inside the host group's vicinity,
    for _ in range(80): # so one trojan reads as one cluster downstream
        cx = int(rng.integers(max(0, gbox[0] - reach), min(w - bw, gbox[2] + reach) + 1))
        cy = int(rng.integers(max(0, gbox[1] - reach), min(h - bh, gbox[3] + reach) + 1))
        probe = rect(h, w, cx - 3, cy - 3, cx + bw + 3, cy + bh + 3)
        if not (mask & probe).any():
            mask |= rect(h, w, cx, cy, cx + bw, cy + bh)
            return [cx, cy, cx + bw, cy + bh]
    loc = find_whitespace_box(mask, rng, bw, bh)   # fall back to anywhere
    if loc is None:
        return None
    cx, cy = loc
    mask |= rect(h, w, cx, cy, cx + bw, cy + bh)
    return [cx, cy, cx + bw, cy + bh]


def _remove_cell(mask, labels, cell):
    """Delete a whole cell. Returns its bbox."""
    i, (x0, y0, x1, y1) = cell
    sub = labels[y0:y1, x0:x1] == i
    mask[y0:y1, x0:x1] &= ~sub
    return [x0, y0, x1, y1]


def _merge_pair(mask, group, rng, o, max_gap=26):
    """Bridge the closest pair of cells in the group. Returns the bridge bbox."""
    h, w = mask.shape
    best = None
    for a in range(len(group)):
        for b in range(a + 1, len(group)):
            ba, bb = group[a][1], group[b][1]
            if o == "vertical":
                gap = max(ba[0], bb[0]) - min(ba[2], bb[2])
                span = min(ba[3], bb[3]) - max(ba[1], bb[1])
            else:
                gap = max(ba[1], bb[1]) - min(ba[3], bb[3])
                span = min(ba[2], bb[2]) - max(ba[0], bb[0])
            if 0 < gap <= max_gap and span > 10:
                if best is None or gap < best[0]:
                    best = (gap, ba, bb)
    if best is None:
        return None
    _, ba, bb = best
    if o == "vertical":
        x0, x1 = min(ba[2], bb[2]), max(ba[0], bb[0])
        lo, hi = max(ba[1], bb[1]), min(ba[3], bb[3])
        yy = int((lo + hi) // 2)
        thick = int(rng.integers(4, 8))
        box = [x0, yy - thick // 2, x1, yy - thick // 2 + thick]
    else:
        y0, y1 = min(ba[3], bb[3]), max(ba[1], bb[1])
        lo, hi = max(ba[0], bb[0]), min(ba[2], bb[2])
        xx = int((lo + hi) // 2)
        thick = int(rng.integers(4, 8))
        box = [xx - thick // 2, y0, xx - thick // 2 + thick, y1]
    mask |= rect(h, w, *box)
    return box


def _split_cell(mask, labels, group, rng, o, min_len=44):
    """Cut a long cell in the group into two pieces. Returns the cut bbox."""
    h, w = mask.shape
    cands = [c for c in group
             if (c[1][3] - c[1][1] if o == "vertical" else c[1][2] - c[1][0]) >= min_len]
    if not cands:
        return None
    i, (x0, y0, x1, y1) = cands[int(rng.integers(len(cands)))]
    gap = int(rng.integers(9, 16))           # wide enough to separate cleanly
    if o == "vertical":
        yy = int(rng.integers(y0 + 14, max(y0 + 15, y1 - 14 - gap)))
        cut = rect(h, w, x0 - 2, yy, x1 + 2, yy + gap)
        box = [x0, yy, x1, yy + gap]
    else:
        xx = int(rng.integers(x0 + 14, max(x0 + 15, x1 - 14 - gap)))
        cut = rect(h, w, xx, y0 - 2, xx + gap, y1 + 2)
        box = [xx, y0, xx + gap, y1]
    sub = labels == i
    mask &= ~(cut & dilate(sub, 2))
    return box


# --------------------------------------------------------------------------
# the five recipes
# --------------------------------------------------------------------------
def _apply(key, mask, labels, group, rng, o):
    """Run one pattern over a host group. Returns [Edit] (possibly empty)."""
    edits = []
    if key == "A":
        for _ in range(int(rng.integers(2, 4))):
            b = _add_cell(mask, group, rng, o)
            if b:
                edits.append(Edit("add", b))

    elif key == "B":
        victims = list(group)
        rng.shuffle(victims)
        for cell in victims[:int(rng.integers(1, 3))]:
            edits.append(Edit("remove", _remove_cell(mask, labels, cell)))

    elif key == "C":
        b = _merge_pair(mask, group, rng, o)
        if b:
            edits.append(Edit("merge", b))

    elif key == "D":
        b = _split_cell(mask, labels, group, rng, o)
        if b:
            edits.append(Edit("split", b))

    elif key == "E":
        b = _add_cell(mask, group, rng, o)
        if b:
            edits.append(Edit("add", b))
        b = _merge_pair(mask, group, rng, o)
        if b:
            edits.append(Edit("merge", b))
        victims = [c for c in group]
        rng.shuffle(victims)
        if victims:
            edits.append(Edit("remove", _remove_cell(mask, labels, victims[0])))
    return edits


def inject_image(gds_u8: np.ndarray, keys, rng, group_size=4, pad=14,
                 min_area=20):
    """Stamp the given patterns into one layout. Returns (image, [TrojanRegion])."""
    mask = binarize(gds_u8, "fixed").copy()
    o = orientation_of(mask)
    regions = []

    for key in keys:
        labels, cells = _cells(mask, min_area)      # re-label after each edit
        group = _neighbourhood(cells, rng, group_size)
        if not group:
            continue
        edits = _apply(key, mask, labels, group, rng, o)
        if not edits:
            continue                                 # pattern could not place
        member_boxes = [b for _, b in group]
        box = _pad(_union(member_boxes + [e.bbox for e in edits]), pad, mask.shape)
        name, _ = PATTERNS[key]
        regions.append(TrojanRegion(key, label_of(key), name, box,
                                    [asdict(e) for e in edits], member_boxes))

    out = np.where(mask, np.uint8(255), np.uint8(0))
    return out, regions


# --------------------------------------------------------------------------
# directory pass
# --------------------------------------------------------------------------
def inject_directory(gds_dir, out_dir, rate=0.7, max_per_image=2,
                     patterns=None, group_size=4, seed=0, round_robin=True,
                     quiet=False):
    """Tamper a fraction of a GDS directory and write gds_trojans.json."""
    gds_dir, out_dir = Path(gds_dir), Path(out_dir)
    keys = [k for k in (patterns or ALL_PATTERNS) if k in PATTERNS]
    if not keys:
        raise ValueError("no valid patterns selected")
    rng = np.random.default_rng(seed)
    gmap = stem_map(gds_dir)
    if not gmap:
        raise SystemExit(f"no images found in {gds_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    images, rr, n_troj = {}, 0, 0
    counts = {k: 0 for k in ALL_PATTERNS}
    for stem in sorted(gmap):
        src = gmap[stem]
        gds = load_gray(src)
        if rng.random() < rate:
            n = int(rng.integers(1, max_per_image + 1))
            if round_robin:
                take = [keys[(rr + i) % len(keys)] for i in range(n)]
                rr += n
            else:
                take = [keys[int(rng.integers(len(keys)))] for _ in range(n)]
            out_img, regions = inject_image(gds, take, rng, group_size)
        else:
            out_img, regions = gds, []

        Image.fromarray(out_img).save(out_dir / src.name)
        images[src.name] = {"size": [int(gds.shape[1]), int(gds.shape[0])],
                            "trojans": [asdict(r) for r in regions]}
        if regions:
            n_troj += 1
        for r in regions:
            counts[r.pattern] += 1
        if not quiet and regions:
            print(f"  {stem:<26} " + ", ".join(
                f"{r.label} ({r.name}, {len(r.edits)} edit"
                f"{'s' if len(r.edits) != 1 else ''})" for r in regions))

    meta = {
        "catalog": [{"pattern": k, "label": label_of(k),
                     "name": PATTERNS[k][0], "description": PATTERNS[k][1]}
                    for k in ALL_PATTERNS],
        "summary": {"images": len(gmap), "with_trojan": n_troj,
                    "clean": len(gmap) - n_troj,
                    "regions": sum(counts.values()),
                    "per_pattern": counts},
        "images": images,
    }
    (out_dir / "gds_trojans.json").write_text(json.dumps(meta, indent=2))

    if not quiet:
        print(f"\nwrote {len(gmap)} layouts to {out_dir} "
              f"({n_troj} with trojans, {len(gmap) - n_troj} clean)")
        print("regions per pattern: "
              + ", ".join(f"{k}={counts[k]}" for k in ALL_PATTERNS))
        print(f"ground truth -> {out_dir / 'gds_trojans.json'}")
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gds-dir", required=True, type=Path,
                    help="clean GDS layouts (e.g. gds_2_sem/A/train)")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--rate", type=float, default=0.7,
                    help="fraction of layouts that receive >=1 trojan")
    ap.add_argument("--max-per-image", type=int, default=2)
    ap.add_argument("--patterns", default="ABCDE")
    ap.add_argument("--group-size", type=int, default=4,
                    help="cells per trojan region (modified + neighbours)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--random-choice", action="store_true",
                    help="draw patterns at random instead of round-robin")
    a = ap.parse_args(argv)
    inject_directory(a.gds_dir, a.out_dir, a.rate, a.max_per_image,
                     list(a.patterns), a.group_size, a.seed,
                     round_robin=not a.random_choice)


if __name__ == "__main__":
    main()
