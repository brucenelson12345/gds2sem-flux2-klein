"""
Ten hardware-trojan patterns (A-J) for the GDS/SEM prototype.

Each pattern is a small, physically-motivated layout modification rendered
in the same SEM style as the dataset (bright poly/metal features ~ value
160 on a ~60 background, with grain and slight edge blur so an inserted
feature blends the way a real one would). Grounded in the failure modes the
reference article calls out: insertions in unused whitespace, added
gates/cells, extra routing, and deletions/modifications of existing cells.

The taxonomy deliberately spans the three change classes a golden-model
comparison must catch:

  ADDITIONS (new material where the golden model has none)
    A  extra_cell        a whole standard-cell-sized block in whitespace
    C  extra_via         a small square contact/via blob
    F  filler_swap       a dense block of thin bars (capacitive/filler cell)
    I  routing_jog       an L-shaped connector added into free space
    J  parallel_route    a thin redundant line beside an existing one

  BRIDGES (new material joining two golden features -> shorts)
    B  bridge_short      a thin bar linking two adjacent parallel lines

  MODIFICATIONS (existing golden feature altered in place)
    D  line_widen        a run of one line made wider than golden
    E  line_extend       one line pushed longer than golden
    H  dopant_patch      an intensity-only patch (geometry unchanged) —
                         the hardest case, mimics a dopant-level trojan

  DELETIONS (golden material removed)
    G  line_cut          a notch/gap cut out of an existing line

Each pattern returns one or more instances, every instance a dict:
    {"pattern": "A", "name": "extra_cell", "class": "addition",
     "bbox": [x0, y0, x1, y1]}        # pixel coords, exclusive x1/y1
so the injector can write ground-truth boxes and the detector can be scored
against them.

Rendering and geometry helpers live in eval/matchlib.py; this module only
adds SEM-texture synthesis and the placement logic.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

# reuse the eval helpers (dilate/erode/connected_components/blur)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from matchlib import blur, connected_components, dilate, erode  # noqa: E402


# --------------------------------------------------------------------------
# SEM texture synthesis
# --------------------------------------------------------------------------
def sample_levels(img: np.ndarray, fg_mask: np.ndarray) -> tuple[float, float, float]:
    """Estimate (background level, foreground level, noise sigma) from an
    image so injected material matches the surrounding SEM statistics."""
    if fg_mask.any():
        fg = float(np.median(img[fg_mask]))
        sig = float(np.std(img[fg_mask])) or 12.0
    else:
        fg, sig = 175.0, 12.0
    bg_mask = ~dilate(fg_mask, 2)
    bg = float(np.median(img[bg_mask])) if bg_mask.any() else 60.0
    return bg, fg, sig


def paint(img: np.ndarray, shape: np.ndarray, fg: float, sig: float,
          rng, edge_blur: float = 0.7) -> None:
    """Alpha-composite a boolean `shape` onto img as bright SEM material,
    with grain and a soft edge, modified in place (img is float)."""
    if not shape.any():
        return
    feature = np.full(img.shape, fg, np.float64)
    feature += rng.normal(0, sig, img.shape)
    # brighten edges slightly, like SEM charging at feature boundaries
    edge = shape & ~erode(shape, 1)
    feature[edge] += sig * 1.5
    alpha = blur(shape.astype(np.float64), edge_blur)
    np.clip(feature, 0, 255, out=feature)
    img[:] = (1.0 - alpha) * img + alpha * feature


def carve(img: np.ndarray, shape: np.ndarray, bg: float, sig: float,
          rng, edge_blur: float = 0.7) -> None:
    """Inverse of paint: replace material with background (a deletion)."""
    if not shape.any():
        return
    filler = np.full(img.shape, bg, np.float64) + rng.normal(0, sig, img.shape)
    alpha = blur(shape.astype(np.float64), edge_blur)
    np.clip(filler, 0, 255, out=filler)
    img[:] = (1.0 - alpha) * img + alpha * filler


# --------------------------------------------------------------------------
# geometry / placement helpers
# --------------------------------------------------------------------------
def rect(h: int, w: int, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    m = np.zeros((h, w), bool)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 > x0 and y1 > y0:
        m[y0:y1, x0:x1] = True
    return m


def cells_and_boxes(gds_mask: np.ndarray, min_area: int = 20):
    labels, n, boxes = connected_components(gds_mask)
    keep = [(i + 1, b) for i, b in enumerate(boxes)
            if (labels == i + 1).sum() >= min_area]
    return labels, keep


def find_whitespace_box(gds_mask: np.ndarray, rng, bw: int, bh: int,
                        tries: int = 60):
    """Find a bw x bh box that doesn't overlap existing material (dilated)."""
    h, w = gds_mask.shape
    occupied = dilate(gds_mask, 3)
    for _ in range(tries):
        x0 = rng.integers(0, max(1, w - bw))
        y0 = rng.integers(0, max(1, h - bh))
        if not occupied[y0:y0 + bh, x0:x0 + bw].any():
            return int(x0), int(y0)
    return None


def orientation_of(gds_mask: np.ndarray) -> str:
    col_var = float(np.var(gds_mask.mean(axis=0)))
    row_var = float(np.var(gds_mask.mean(axis=1)))
    return "vertical" if col_var >= row_var else "horizontal"


# --------------------------------------------------------------------------
# pattern registry
# --------------------------------------------------------------------------
@dataclass
class Pattern:
    key: str
    name: str
    cls: str
    fn: Callable
    desc: str = ""


REGISTRY: dict[str, Pattern] = {}


def register(key, name, cls, desc=""):
    def deco(fn):
        REGISTRY[key] = Pattern(key, name, cls, fn, desc)
        return fn
    return deco


def _instance(key, bbox):
    p = REGISTRY[key]
    return {"pattern": key, "name": p.name, "class": p.cls,
            "bbox": [int(v) for v in bbox]}


# ---- ADDITIONS -----------------------------------------------------------
@register("A", "extra_cell", "addition",
          "standard-cell-sized block inserted in unused whitespace")
def p_extra_cell(img, gds, rng, o, bg, fg, sig):
    long = rng.integers(28, 55)
    short = rng.integers(10, 18)
    bw, bh = (short, long) if o == "vertical" else (long, short)
    loc = find_whitespace_box(gds, rng, bw, bh)
    if loc is None:
        return []
    x0, y0 = loc
    paint(img, rect(*img.shape, x0, y0, x0 + bw, y0 + bh), fg, sig, rng)
    return [_instance("A", (x0, y0, x0 + bw, y0 + bh))]


@register("C", "extra_via", "addition",
          "small square contact/via blob added between features")
def p_extra_via(img, gds, rng, o, bg, fg, sig):
    s = int(rng.integers(6, 11))
    loc = find_whitespace_box(gds, rng, s, s)
    if loc is None:
        return []
    x0, y0 = loc
    paint(img, rect(*img.shape, x0, y0, x0 + s, y0 + s), fg, sig, rng, 0.5)
    return [_instance("C", (x0, y0, x0 + s, y0 + s))]


@register("F", "filler_swap", "addition",
          "dense block of thin bars replacing a filler cell")
def p_filler_swap(img, gds, rng, o, bg, fg, sig):
    bw, bh = (rng.integers(26, 40), rng.integers(26, 40))
    loc = find_whitespace_box(gds, rng, int(bw), int(bh))
    if loc is None:
        return []
    x0, y0 = loc
    block = np.zeros(img.shape, bool)
    if o == "vertical":
        for xx in range(x0 + 1, x0 + bw, 4):
            block |= rect(*img.shape, xx, y0, xx + 2, y0 + bh)
    else:
        for yy in range(y0 + 1, y0 + bh, 4):
            block |= rect(*img.shape, x0, yy, x0 + bw, yy + 2)
    paint(img, block, fg, sig, rng, 0.5)
    return [_instance("F", (x0, y0, x0 + int(bw), y0 + int(bh)))]


@register("I", "routing_jog", "addition",
          "L-shaped routing connector added into free space")
def p_routing_jog(img, gds, rng, o, bg, fg, sig):
    L = int(rng.integers(20, 34))
    t = 3
    loc = find_whitespace_box(gds, rng, L, L)
    if loc is None:
        return []
    x0, y0 = loc
    shape = (rect(*img.shape, x0, y0, x0 + L, y0 + t)
             | rect(*img.shape, x0, y0, x0 + t, y0 + L))
    paint(img, shape, fg, sig, rng, 0.5)
    return [_instance("I", (x0, y0, x0 + L, y0 + L))]


@register("J", "parallel_route", "addition",
          "thin redundant line added beside an existing one")
def p_parallel_route(img, gds, rng, o, bg, fg, sig):
    _, keep = cells_and_boxes(gds)
    rng.shuffle(keep)
    for _, (x0, y0, x1, y1) in keep:
        if o == "vertical" and (y1 - y0) > 20:
            for dx in (x1 + 3, x0 - 6):
                nb = rect(*img.shape, dx, y0, dx + 3, y1)
                if not dilate(gds, 2)[nb].any() and nb.any():
                    paint(img, nb, fg, sig, rng, 0.5)
                    return [_instance("J", (dx, y0, dx + 3, y1))]
        elif o == "horizontal" and (x1 - x0) > 20:
            for dy in (y1 + 3, y0 - 6):
                nb = rect(*img.shape, x0, dy, x1, dy + 3)
                if not dilate(gds, 2)[nb].any() and nb.any():
                    paint(img, nb, fg, sig, rng, 0.5)
                    return [_instance("J", (x0, dy, x1, dy + 3))]
    return []


# ---- BRIDGE --------------------------------------------------------------
@register("B", "bridge_short", "bridge",
          "thin bar bridging two adjacent parallel lines (a short)")
def p_bridge_short(img, gds, rng, o, bg, fg, sig):
    _, keep = cells_and_boxes(gds)
    boxes = [b for _, b in keep]
    rng.shuffle(boxes)
    for i in range(len(boxes)):
        x0, y0, x1, y1 = boxes[i]
        for j in range(len(boxes)):
            if i == j:
                continue
            a0, b0, a1, b1 = boxes[j]
            if o == "vertical":
                gap = a0 - x1
                if 3 <= gap <= 22 and min(y1, b1) - max(y0, b0) > 8:
                    yy = int((max(y0, b0) + min(y1, b1)) // 2)
                    br = rect(*img.shape, x1, yy - 1, a0, yy + 2)
                    paint(img, br, fg, sig, rng, 0.5)
                    return [_instance("B", (x1, yy - 2, a0, yy + 3))]
            else:
                gap = b0 - y1
                if 3 <= gap <= 22 and min(x1, a1) - max(x0, a0) > 8:
                    xx = int((max(x0, a0) + min(x1, a1)) // 2)
                    br = rect(*img.shape, xx - 1, y1, xx + 2, b0)
                    paint(img, br, fg, sig, rng, 0.5)
                    return [_instance("B", (xx - 2, y1, xx + 3, b0))]
    return []


# ---- MODIFICATIONS -------------------------------------------------------
@register("D", "line_widen", "modification",
          "a run of one line made wider than the golden layout")
def p_line_widen(img, gds, rng, o, bg, fg, sig):
    _, keep = cells_and_boxes(gds)
    rng.shuffle(keep)
    for _, (x0, y0, x1, y1) in keep:
        if o == "vertical" and (y1 - y0) > 24:
            yy = int(rng.integers(y0, max(y0 + 1, y1 - 16)))
            seg = rect(*img.shape, x1, yy, x1 + 3, yy + 16)
            if not dilate(gds, 1)[seg].any():
                paint(img, seg, fg, sig, rng, 0.5)
                return [_instance("D", (x1 - 1, yy, x1 + 3, yy + 16))]
        elif o == "horizontal" and (x1 - x0) > 24:
            xx = int(rng.integers(x0, max(x0 + 1, x1 - 16)))
            seg = rect(*img.shape, xx, y1, xx + 16, y1 + 3)
            if not dilate(gds, 1)[seg].any():
                paint(img, seg, fg, sig, rng, 0.5)
                return [_instance("D", (xx, y1 - 1, xx + 16, y1 + 3))]
    return []


@register("E", "line_extend", "modification",
          "one line pushed longer than the golden layout")
def p_line_extend(img, gds, rng, o, bg, fg, sig):
    # An extension abuts its own line's end, so test collisions against the
    # UNDILATED golden mask (the segment must not sit on any material) plus a
    # clear margin just beyond the segment's far end (no other cell there).
    _, keep = cells_and_boxes(gds)
    rng.shuffle(keep)
    for _, (x0, y0, x1, y1) in keep:
        for ext in range(18, 6, -2):
            if o == "vertical":
                cands = [((x0, y1, x1, y1 + ext), (x0, y1 + ext, x1, y1 + ext + 2)),
                         ((x0, y0 - ext, x1, y0), (x0, y0 - ext - 2, x1, y0 - ext))]
            else:
                cands = [((x1, y0, x1 + ext, y1), (x1 + ext, y0, x1 + ext + 2, y1)),
                         ((x0 - ext, y0, x0, y1), (x0 - ext - 2, y0, x0 - ext, y1))]
            for (bx0, by0, bx1, by1), (mx0, my0, mx1, my1) in cands:
                seg = rect(*img.shape, bx0, by0, bx1, by1)
                margin = rect(*img.shape, mx0, my0, mx1, my1)
                if seg.sum() >= 12 and not gds[seg].any() and not gds[margin].any():
                    paint(img, seg, fg, sig, rng)
                    return [_instance("E", (bx0, by0, bx1, by1))]
    return []


@register("H", "dopant_patch", "modification",
          "intensity-only patch on a line (geometry unchanged; dopant-level)")
def p_dopant_patch(img, gds, rng, o, bg, fg, sig):
    _, keep = cells_and_boxes(gds)
    rng.shuffle(keep)
    for _, (x0, y0, x1, y1) in keep:
        if (x1 - x0) * (y1 - y0) < 60:
            continue
        pw = min(x1 - x0, int(rng.integers(8, 16)))
        ph = min(y1 - y0, int(rng.integers(8, 16)))
        px = int(rng.integers(x0, x1 - pw + 1))
        py = int(rng.integers(y0, y1 - ph + 1))
        patch = rect(*img.shape, px, py, px + pw, py + ph) & (gds > 0)
        # darken the material ~25%: a contrast anomaly without a shape change
        alpha = blur(patch.astype(np.float64), 0.6)
        img[:] = (1.0 - alpha * 0.28) * img
        return [_instance("H", (px, py, px + pw, py + ph))]
    return []


# ---- DELETION ------------------------------------------------------------
@register("G", "line_cut", "deletion",
          "a notch/gap cut out of an existing line")
def p_line_cut(img, gds, rng, o, bg, fg, sig):
    _, keep = cells_and_boxes(gds)
    rng.shuffle(keep)
    for _, (x0, y0, x1, y1) in keep:
        if o == "vertical" and (y1 - y0) > 26:
            yy = int(rng.integers(y0 + 8, y1 - 12))
            cut = rect(*img.shape, x0 - 1, yy, x1 + 1, yy + int(rng.integers(6, 12)))
            carve(img, cut & dilate(gds > 0, 1), bg, sig, rng, 0.5)
            b = (x0, yy, x1, yy + 10)
            return [_instance("G", b)]
        elif o == "horizontal" and (x1 - x0) > 26:
            xx = int(rng.integers(x0 + 8, x1 - 12))
            cut = rect(*img.shape, xx, y0 - 1, xx + int(rng.integers(6, 12)), y1 + 1)
            carve(img, cut & dilate(gds > 0, 1), bg, sig, rng, 0.5)
            return [_instance("G", (xx, y0, xx + 10, y1))]
    return []


ALL_KEYS = list("ABCDEFGHIJ")


def catalog() -> list[dict]:
    return [{"pattern": k, "name": REGISTRY[k].name,
             "class": REGISTRY[k].cls, "description": REGISTRY[k].desc}
            for k in ALL_KEYS]
