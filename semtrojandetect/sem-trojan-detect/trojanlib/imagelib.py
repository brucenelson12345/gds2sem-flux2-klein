"""
Shared image helpers for the GDS/SEM matchers (match_gds_sem.py,
match_sem_sem.py) and general eval use.

OpenCV is used when importable (fast paths for connected components,
morphology and blur); every function has a pure-numpy fallback so the
scripts still run on a machine where only numpy + Pillow are installed.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

try:
    import cv2
except ImportError:  # pure-numpy fallbacks are used instead
    cv2 = None

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

GREEN = (0, 200, 70)
RED = (230, 40, 40)
GREY = (150, 150, 150)


# --------------------------------------------------------------------------
# io / pairing
# --------------------------------------------------------------------------
def stem_map(d: Path) -> dict[str, Path]:
    """Map filename stem -> path, stripping ComfyUI counter suffixes
    (`cell_007_00001_.png` -> `cell_007`)."""
    if not d.is_dir():
        raise SystemExit(f"ERROR: not a directory: {d}")
    out: dict[str, Path] = {}
    for p in sorted(d.iterdir()):
        if p.suffix.lower() in IMG_EXTS:
            out[re.sub(r"_\d{5}_?$", "", p.stem)] = p
    return out


def matched_stems(a: dict, b: dict, a_name="A", b_name="B") -> list[str]:
    stems = sorted(set(a) & set(b))
    for missing, where in ((sorted(set(a) - set(b)), b_name),
                           (sorted(set(b) - set(a)), a_name)):
        if missing:
            print(f"  note: {len(missing)} image(s) with no counterpart in "
                  f"{where}, skipped: {missing[:5]}")
    return stems


def load_gray(p: Path) -> np.ndarray:
    return np.asarray(Image.open(p).convert("L"), dtype=np.uint8)


def resize_to(a: np.ndarray, size: tuple[int, int], nearest: bool) -> np.ndarray:
    """size = (width, height). Nearest for binary/GDS art, bicubic for photos."""
    if (a.shape[1], a.shape[0]) == size:
        return a
    mode = Image.NEAREST if nearest else Image.BICUBIC
    return np.asarray(Image.fromarray(a).resize(size, mode), dtype=np.uint8)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------
# thresholding
# --------------------------------------------------------------------------
def otsu_threshold(a: np.ndarray) -> float:
    hist = np.bincount(a.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    w0 = np.cumsum(hist)
    m0 = np.cumsum(hist * np.arange(256))
    mg = m0[-1] / total
    valid = (w0 > 0) & (w0 < total)
    between = np.zeros(256)
    between[valid] = ((mg * w0[valid] - m0[valid]) ** 2
                      / (w0[valid] * (total - w0[valid])))
    return float(np.argmax(between))


def binarize(a: np.ndarray, method: str = "otsu") -> np.ndarray:
    """'fixed' (mid-grey, for near-binary GDS art) or 'otsu' (for SEM)."""
    return a >= (128 if method == "fixed" else otsu_threshold(a))


# --------------------------------------------------------------------------
# morphology
# --------------------------------------------------------------------------
def _shift(m: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift a boolean array, filling vacated space with False."""
    out = np.zeros_like(m)
    ys, yd = (slice(max(0, -dy), m.shape[0] - max(0, dy)),
              slice(max(0, dy), m.shape[0] - max(0, -dy)))
    xs, xd = (slice(max(0, -dx), m.shape[1] - max(0, dx)),
              slice(max(0, dx), m.shape[1] - max(0, -dx)))
    out[yd, xd] = m[ys, xs]
    return out


def dilate(m: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return m
    if cv2 is not None:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * r + 1, 2 * r + 1))
        return cv2.dilate(m.astype(np.uint8), k).astype(bool)
    out = m.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out |= _shift(m, dy, dx)
    return out


def erode(m: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return m
    if cv2 is not None:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * r + 1, 2 * r + 1))
        return cv2.erode(m.astype(np.uint8), k).astype(bool)
    return ~dilate(~m, r)


def outline(m: np.ndarray, width: int = 1) -> np.ndarray:
    return m & ~erode(m, width)


# --------------------------------------------------------------------------
# connected components  ->  (labels, count, bboxes)
# bbox = (x0, y0, x1, y1) with exclusive x1/y1, so arr[y0:y1, x0:x1] slices it
# --------------------------------------------------------------------------
def connected_components(mask: np.ndarray):
    if cv2 is not None:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8)
        boxes = []
        for i in range(1, n):
            x, y, w, h = stats[i, :4]
            boxes.append((int(x), int(y), int(x + w), int(y + h)))
        return labels, n - 1, boxes
    return _cc_numpy(mask)


def _cc_numpy(mask: np.ndarray):
    """Two-pass union-find labeler (8-connected). Slower than OpenCV's —
    used only when cv2 is unavailable."""
    h, w = mask.shape
    labels = np.zeros((h, w), np.int32)
    parent = [0]

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    nxt = 1
    for y in range(h):
        for x in np.nonzero(mask[y])[0]:
            neigh = [labels[y + dy, x + dx]
                     for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1))
                     if 0 <= y + dy < h and 0 <= x + dx < w
                     and labels[y + dy, x + dx]]
            if neigh:
                m = min(neigh)
                labels[y, x] = m
                for nb in neigh:
                    union(m, nb)
            else:
                labels[y, x] = nxt
                parent.append(nxt)
                nxt += 1

    # resolve equivalences and compact the label ids
    roots = np.array([find(i) for i in range(nxt)], np.int32)
    uniq = {r: i + 1 for i, r in enumerate(sorted(set(roots[1:].tolist())))}
    remap = np.zeros(nxt, np.int32)
    for i in range(1, nxt):
        remap[i] = uniq[roots[i]]
    labels = remap[labels]

    n = len(uniq)
    boxes = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(labels == i)
        boxes.append((int(xs.min()), int(ys.min()),
                      int(xs.max()) + 1, int(ys.max()) + 1))
    return labels, n, boxes


# --------------------------------------------------------------------------
# local statistics (for the SEM<->SEM similarity map)
# --------------------------------------------------------------------------
def box_mean(a: np.ndarray, w: int) -> np.ndarray:
    """Mean over a w x w window, edge-padded, via an integral image."""
    r = w // 2
    p = np.pad(a.astype(np.float64), r, mode="edge")
    ii = np.zeros((p.shape[0] + 1, p.shape[1] + 1), np.float64)
    ii[1:, 1:] = p.cumsum(0).cumsum(1)
    h, wd = a.shape
    s = (ii[w:w + h, w:w + wd] - ii[0:h, w:w + wd]
         - ii[w:w + h, 0:wd] + ii[0:h, 0:wd])
    return s / (w * w)


def blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur on a float image (3 box passes approximate it without cv2)."""
    if sigma <= 0:
        return a
    if cv2 is not None:
        return cv2.GaussianBlur(a, (0, 0), sigma)
    w = max(3, int(round(sigma * 2)) * 2 + 1)
    out = a
    for _ in range(3):
        out = box_mean(out, w)
    return out


def local_ssim(x: np.ndarray, y: np.ndarray, w: int = 11) -> np.ndarray:
    """Per-pixel SSIM map for float images in [0, 1]."""
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mx, my = box_mean(x, w), box_mean(y, w)
    vx = np.maximum(box_mean(x * x, w) - mx * mx, 0)
    vy = np.maximum(box_mean(y * y, w) - my * my, 0)
    vxy = box_mean(x * y, w) - mx * my
    return (((2 * mx * my + c1) * (2 * vxy + c2))
            / ((mx * mx + my * my + c1) * (vx + vy + c2)))


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------
def gray_to_rgb(a: np.ndarray) -> np.ndarray:
    return np.repeat(a[:, :, None].astype(np.float64), 3, axis=2)


def tint(rgb: np.ndarray, mask: np.ndarray, color, alpha: float) -> None:
    """Alpha-blend a flat colour into rgb (float 0-255, modified in place)."""
    if not mask.any():
        return
    c = np.array(color, np.float64)
    rgb[mask] = (1.0 - alpha) * rgb[mask] + alpha * c


def caption(rgb: np.ndarray, lines: list[tuple[str, tuple]],
            bar: int = 0) -> Image.Image:
    """Add a caption bar above the image; each line is (text, colour)."""
    img = Image.fromarray(rgb.clip(0, 255).astype(np.uint8))
    bar = bar or (14 * len(lines) + 8)
    out = Image.new("RGB", (img.width, img.height + bar), (16, 16, 16))
    out.paste(img, (0, bar))
    d = ImageDraw.Draw(out)
    for i, (text, color) in enumerate(lines):
        d.text((6, 4 + 12 * i), text, fill=color)
    return out
