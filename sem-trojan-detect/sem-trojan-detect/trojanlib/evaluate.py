"""
Score detector output (results.json) against injection ground truth
(ground_truth.json from trojanlib.inject).

Reports image-level detection (is a trojaned image flagged?), instance-level
localisation at an IoU threshold, classification accuracy over the localised
regions, a per-pattern table, and a confusion matrix.

Importable:
    from trojanlib import evaluate
    text, stats = evaluate("ground_truth.json", "results.json", iou=0.3)

CLI:  python -m trojanlib.evaluate --truth ground_truth.json --results results.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import patterns as P

KEYS = P.ALL_KEYS


def iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua else 0.0


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def evaluate(truth_path, results_path, iou_thresh: float = 0.3):
    """Returns (human-readable report text, stats dict)."""
    truth = json.loads(Path(truth_path).read_text())["images"]
    res = json.loads(Path(results_path).read_text())["images"]

    img = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    inst = {"tp": 0, "fp": 0, "fn": 0}
    class_correct = 0
    per = {k: [0, 0, 0] for k in KEYS}                    # tp, fp, fn
    confusion = {k: {j: 0 for j in KEYS} for k in KEYS}

    for name, gt in truth.items():
        gts = list(gt["trojans"])
        dets = list(res.get(name, {}).get("detections", []))
        has, flagged = bool(gts), bool(dets)
        img["tp"] += has and flagged
        img["fn"] += has and not flagged
        img["fp"] += (not has) and flagged
        img["tn"] += (not has) and not flagged

        used = set()
        for g in gts:                                    # greedy 1:1 by IoU
            best_j, best_v = -1, iou_thresh
            for j, d in enumerate(dets):
                if j in used:
                    continue
                v = iou(g["bbox"], d["bbox"])
                if v >= best_v:
                    best_v, best_j = v, j
            if best_j >= 0:
                used.add(best_j)
                inst["tp"] += 1
                per[g["pattern"]][0] += 1
                pred = dets[best_j]["pattern"]
                confusion[g["pattern"]][pred] += 1
                class_correct += pred == g["pattern"]
            else:
                inst["fn"] += 1
                per[g["pattern"]][2] += 1
        for j, d in enumerate(dets):
            if j not in used:
                inst["fp"] += 1
                per[d["pattern"]][1] += 1

    L = []
    L.append("=" * 60)
    L.append("IMAGE-LEVEL  (trojan present vs flagged)")
    ip, ir, if_ = prf(img["tp"], img["fp"], img["fn"])
    L.append(f"  flagged-correct {img['tp']}  missed {img['fn']}  "
             f"false-alarm {img['fp']}  clean-ok {img['tn']}")
    L.append(f"  precision {ip:.3f}  recall {ir:.3f}  F1 {if_:.3f}")

    L.append(f"\nINSTANCE-LEVEL  (localised @ IoU >= {iou_thresh}, any class)")
    np_, nr, nf = prf(inst["tp"], inst["fp"], inst["fn"])
    L.append(f"  found {inst['tp']}  missed {inst['fn']}  spurious {inst['fp']}")
    L.append(f"  precision {np_:.3f}  recall {nr:.3f}  F1 {nf:.3f}")

    acc = class_correct / inst["tp"] if inst["tp"] else 0.0
    if inst["tp"]:
        L.append(f"\nCLASSIFICATION  (of the {inst['tp']} localised regions)")
        L.append(f"  correct pattern: {class_correct}/{inst['tp']} = {acc:.1%}")

    L.append("\nPER-PATTERN")
    L.append(f"  {'':3}{'name':<16}{'tp':>4}{'fp':>4}{'fn':>4}{'prec':>7}{'rec':>7}")
    for k in KEYS:
        tp, fp, fn = per[k]
        pp, rr, _ = prf(tp, fp, fn)
        L.append(f"  {k:<3}{P.REGISTRY[k].name:<16}{tp:>4}{fp:>4}{fn:>4}"
                 f"{pp:>7.2f}{rr:>7.2f}")

    L.append("\nCONFUSION  (rows = true pattern, cols = predicted)")
    L.append("      " + "".join(f"{k:>4}" for k in KEYS))
    for k in KEYS:
        if sum(confusion[k].values()):
            L.append(f"  {k:<3} " + "".join(f"{confusion[k][j]:>4}" for j in KEYS))

    stats = {"image": {**img, "precision": ip, "recall": ir, "f1": if_},
             "instance": {**inst, "precision": np_, "recall": nr, "f1": nf},
             "classification_accuracy": acc,
             "per_pattern": {k: dict(zip(("tp", "fp", "fn"), v))
                             for k, v in per.items()},
             "confusion": confusion}
    return "\n".join(L), stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth", required=True, type=Path)
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--iou", type=float, default=0.3)
    a = ap.parse_args(argv)
    text, _ = evaluate(a.truth, a.results, a.iou)
    print(text)


if __name__ == "__main__":
    main()
