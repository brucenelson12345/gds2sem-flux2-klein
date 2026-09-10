#!/usr/bin/env python3
"""
Train a YOLO detector on an exported trojan dataset (from
export_yolo_dataset.py), for the detector's `yolo` backend.

This is optional: the golden-model backend needs no training. Use YOLO when
you want a single-image detector (no golden reference at inference), or to
push classification accuracy on subtle patterns beyond what the rules give.

Runs fully offline provided the base weights are already on disk — pass a
local --weights path (e.g. a pre-downloaded yolo11s.pt staged on the
transfer share). Ultralytics will NOT reach the network when given a local
checkpoint and with YOLO_OFFLINE=1 (set in the container).

Usage (inside the trojan container, GPU available):
  python train_yolo.py --data yolo_ds/dataset.yaml --weights /models/yolo11s.pt \
      --epochs 150 --imgsz 512 --batch 16 --project /data/runs --name troj_yolo
Result: /data/runs/troj_yolo/weights/best.pt  -> pass to detect_trojans
        --backend yolo --weights ...
"""
import argparse
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path, help="dataset.yaml")
    ap.add_argument("--weights", default="yolo11s.pt",
                    help="base checkpoint; use a LOCAL path when offline")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", type=Path, default=Path("runs"))
    ap.add_argument("--name", default="troj_yolo")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("needs `pip install ultralytics` (present in the trojan image)")

    model = YOLO(args.weights)
    model.train(
        data=str(args.data), epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, device=args.device,
        project=str(args.project), name=args.name,
        # SEM images are grayscale, structured, and small — mild augmentation
        # only; geometry-preserving so boxes stay valid.
        degrees=0.0, shear=0.0, perspective=0.0, mosaic=0.5,
        fliplr=0.5, flipud=0.5, hsv_h=0.0, hsv_s=0.0, hsv_v=0.2,
    )
    best = args.project / args.name / "weights" / "best.pt"
    print(f"\nbest weights -> {best}")
    print("use:  detect_trojans.py --backend yolo --weights "
          f"{best} --root INPUT --out D")


if __name__ == "__main__":
    main()
