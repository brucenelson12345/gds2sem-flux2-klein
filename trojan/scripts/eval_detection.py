#!/usr/bin/env python3
"""
Score detector output (D/results.json) against injection ground truth
(ground_truth.json from inject_trojans.py).

Reports:
  * Image-level detection: does the image get flagged when it has >=1 trojan?
    (precision / recall / F1 over the trojan-vs-clean decision)
  * Instance-level detection: are individual trojan regions found, by
    IoU >= --iou, regardless of predicted class?
  * Classification: of the correctly-localised instances, how many got the
    right pattern letter? Plus a per-pattern table and a confusion matrix.

Greedy one-to-one matching of detections to ground-truth boxes per image.

Usage:
  python eval_detection.py --truth trojan_testset/ground_truth.json \
      --results D/results.json --iou 0.3
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import patterns as P  # noqa: E402

KEYS = P.ALL_KEYS


def iou(a, b):
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


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth", required=True, type=Path)
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--iou", type=float, default=0.3,
                    help="IoU for a detection to count as localised")
    args = ap.parse_args()

    truth = json.loads(args.truth.read_text())
    truth_imgs = truth["images"]
    res = json.loads(args.results.read_text())["images"]

    # image-level (flag vs clean)
    img_tp = img_fp = img_fn = img_tn = 0
    # instance-level
    inst_tp = inst_fp = inst_fn = 0
    class_correct = 0
    # per-pattern tp/fp/fn and confusion[true][pred]
    per = {k: [0, 0, 0] for k in KEYS}
    confusion = {k: {j: 0 for j in KEYS} for k in KEYS}

    for name, gt in truth_imgs.items():
        gts = list(gt["trojans"])
        dets = list(res.get(name, {}).get("detections", []))

        has = len(gts) > 0
        flagged = len(dets) > 0
        img_tp += has and flagged
        img_fn += has and not flagged
        img_fp += (not has) and flagged
        img_tn += (not has) and not flagged

        # greedy match by IoU
        used = set()
        for g in gts:
            best_j, best_i = -1, args.iou
            for j, d in enumerate(dets):
                if j in used:
                    continue
                v = iou(g["bbox"], d["bbox"])
                if v >= best_i:
                    best_i, best_j = v, j
            if best_j >= 0:
                used.add(best_j)
                inst_tp += 1
                per[g["pattern"]][0] += 1
                pred = dets[best_j]["pattern"]
                confusion[g["pattern"]][pred] += 1
                if pred == g["pattern"]:
                    class_correct += 1
                else:
                    per[g["pattern"]][2] += 0  # fn accounted below via class
            else:
                inst_fn += 1
                per[g["pattern"]][2] += 1
        for j, d in enumerate(dets):
            if j not in used:
                inst_fp += 1
                per[d["pattern"]][1] += 1

    print("=" * 60)
    print("IMAGE-LEVEL  (trojan present vs flagged)")
    p, r, f = prf(img_tp, img_fp, img_fn)
    print(f"  flagged-correct {img_tp}  missed {img_fn}  "
          f"false-alarm {img_fp}  clean-ok {img_tn}")
    print(f"  precision {p:.3f}  recall {r:.3f}  F1 {f:.3f}")

    print("\nINSTANCE-LEVEL  (regions localised @ IoU >= "
          f"{args.iou}, any class)")
    p, r, f = prf(inst_tp, inst_fp, inst_fn)
    print(f"  found {inst_tp}  missed {inst_fn}  spurious {inst_fp}")
    print(f"  precision {p:.3f}  recall {r:.3f}  F1 {f:.3f}")
    if inst_tp:
        print(f"\nCLASSIFICATION  (of the {inst_tp} localised regions)")
        print(f"  correct pattern: {class_correct}/{inst_tp} "
              f"= {class_correct / inst_tp:.1%}")

    print("\nPER-PATTERN")
    print(f"  {'':3}{'name':<16}{'tp':>4}{'fp':>4}{'fn':>4}"
          f"{'prec':>7}{'rec':>7}")
    for k in KEYS:
        tp, fp, fn = per[k]
        pp, rr, _ = prf(tp, fp, fn)
        print(f"  {k:<3}{P.REGISTRY[k].name:<16}{tp:>4}{fp:>4}{fn:>4}"
              f"{pp:>7.2f}{rr:>7.2f}")

    print("\nCONFUSION  (rows = true pattern, cols = predicted)")
    print("      " + "".join(f"{k:>4}" for k in KEYS))
    for k in KEYS:
        row = confusion[k]
        if sum(row.values()) == 0:
            continue
        print(f"  {k:<3} " + "".join(f"{row[j]:>4}" for j in KEYS))


if __name__ == "__main__":
    main()
