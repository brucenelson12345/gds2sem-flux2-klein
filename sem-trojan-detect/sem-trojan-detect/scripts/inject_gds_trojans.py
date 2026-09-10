#!/usr/bin/env python3
"""
inject_gds_trojans — stamp trojan regions into a directory of GDS layouts.

Builds a labelled training/test set one level above the SEM: it edits the
*layout*, adding cells, deleting cells, bridging adjacent pairs and cutting
nets, then groups each modification together with its neighbouring cells and
labels that group as a trojan region.

Five patterns:

    Trojan A  inserted_cluster   2-3 new cells in the group's whitespace
    Trojan B  depopulated        1-2 existing cells deleted
    Trojan C  merged_pair        two adjacent cells bridged into one
    Trojan D  severed_net        one cell cut into two separated pieces
    Trojan E  rerouted_block     mixed: one added, one removed, one pair bridged

Writes the tampered layouts plus `gds_trojans.json`, which records for every
image the region boxes, their labels, and the individual edits — the ground
truth `screen_matcher.py --truth` scores against, and the source for a YOLO
training set.

The intended pipeline:

    clean layouts  ──gds2sem──►  SEM  =  B   (golden)
         │
         └─inject_gds_trojans──►  tampered layouts ──gds2sem──►  SEM  =  C
                                                                    │
                          screen_matcher.py  B vs C  ◄──────────────┘

Usage:
  python3 scripts/inject_gds_trojans.py --gds-dir gds_2_sem/A/train \\
      --out-dir A_trojan --rate 0.7 --max-per-image 2
  # only the connectivity patterns, three per layout:
  python3 scripts/inject_gds_trojans.py --gds-dir A/train --out-dir A_conn \\
      --patterns CD --max-per-image 3
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trojanlib.gds_trojans import main  # noqa: E402

if __name__ == "__main__":
    main()
