# Hardware-trojan screening (prototype)

Extends the GDS→SEM pipeline into a trojan-detection prototype: given a
manufactured chip's SEM (C) and the golden model you already hold (the GDS
layout A and the original known-good SEM B), it flags regions of C that
differ from golden, classifies each into one of ten trojan patterns (A–J),
and returns a JSON verdict plus annotated images with bounding boxes. It's
driven from LibreChat (Opus 5 backend) through an MCP server, and runs on
the same offline host as image generation — optionally on a separate GPU.

The approach follows the golden-model comparison the reference article
describes (detecting insertion / deletion / modification against a
trojan-free reference); the taxonomy below covers all three change classes.

## The ten patterns (A–J)

| key | name | class | what it is |
|---|---|---|---|
| A | extra_cell | addition | a standard-cell-sized block dropped into unused whitespace |
| B | bridge_short | bridge | a thin bar linking two adjacent lines (a short) |
| C | extra_via | addition | a small square contact/via blob |
| D | line_widen | modification | a run of one line made wider than golden |
| E | line_extend | modification | one line pushed longer than golden |
| F | filler_swap | addition | a dense block of thin bars (filler/capacitor cell) |
| G | line_cut | deletion | a notch/gap cut out of an existing line |
| H | dopant_patch | modification | an intensity-only patch (no shape change) — dopant-level |
| I | routing_jog | addition | an L-shaped routing connector in free space |
| J | parallel_route | addition | a thin redundant line beside an existing one |

Classes matter for triage: **additions** and **bridges** add material,
**deletions** remove it, **modifications** alter a feature in place. H
(dopant) is the hardest — geometry is unchanged, so it can only be seen by
comparing intensity against the original SEM (B); without B it is invisible.

## Files

```
trojan/
├── scripts/
│   ├── patterns.py            the 10 patterns + SEM-texture synthesis
│   ├── inject_trojans.py      build a LABELLED test set (tampered C + ground truth)
│   ├── detect_trojans.py      the detector (golden-model + optional YOLO backends)
│   ├── eval_detection.py      score detections vs ground truth
│   ├── export_yolo_dataset.py injected sets -> YOLO dataset
│   └── train_yolo.py          train the YOLO backend (optional)
│   ├── screen.py              CLI front-end: detect/demo/eval/remote, HTML report
├── mcp/server.py              MCP server exposing the detector to LibreChat
├── librechat/
│   ├── librechat.yaml         LibreChat config (MCP server + Opus 5 + image UI)
│   └── agent_instructions.md  system prompt for the screening agent
├── docker/
│   ├── Dockerfile             detection + MCP image (cu126, offline)
│   └── requirements.txt
├── run_detector_mcp.sh        launch the MCP service on the offline host
├── run_screen.sh              one-shot containerized CLI run (no LibreChat)
└── README.md                  (this file)
```

## How detection works (golden backend, the default)

No training, deterministic, fully offline. For each suspect image C:

1. Binarise C (Otsu) and build the golden material mask from A (upscaled to
   C's size — 465→512 nearest-neighbour) with B supplying the intensity
   baseline.
2. Difference against golden with a few px of tolerance:
   *added* = material in C not in golden, *removed* = golden material absent
   from C, *intensity* = shared material whose brightness in C departs from B
   (the dopant signal).
3. Label connected anomaly regions; for each, compute geometric/photometric
   features (addition/deletion/intensity fractions, aspect, size, bbox fill,
   how many golden features it touches, L-shape arms) and classify into A–J
   with a calibrated decision tree.
4. Draw class-coloured boxes, write `results.json` + `annotated/`.

The classifier thresholds were calibrated on the injected synthetic set;
here is the feature separation they key on (one region per pattern):

```
A extra_cell    add-frac 0.7  aspect 4.1  -> elongated standalone block
C extra_via     add-frac 0.5  aspect 1.0  long 13  -> small square
D line_widen    add-frac      aspect 2.9  short 7  area 140 -> thin edge strip
E line_extend   add-frac      aspect 1.2  short 16 area 320 -> chunky end block
F filler_swap   add-frac 0.8  aspect 1.1  area 1520 -> large square block
G line_cut      rem-frac 0.42            -> deletion
H dopant_patch  int-frac 0.45            -> intensity-only
I routing_jog   add-frac      fill 0.42  arms both -> L-shape
J parallel_route add-frac     aspect 13.9 short 7 -> very long thin line
```

On the synthetic evaluation (12 images, one of every pattern injected):
image-level detection precision/recall **1.00 / 1.00**, instance
localisation F1 **0.96**, and **11/11** localised regions classified
correctly. **These numbers are on synthetic data** where injected features
share the renderer's statistics; on real captures expect more golden-diff
noise (the generated/real SEM differs from golden along every edge), so
treat flags as "regions to review", tune `--tolerance` / `--min-area` /
`--intensity-delta` to your images, and lean on the YOLO backend for
production-grade classification.

### The YOLO / RF-DETR backend (optional, learned)

When you want a single-image detector (no golden reference at inference) or
higher classification accuracy on subtle patterns:

```bash
# 1. build several labelled sets with different seeds
python trojan/scripts/inject_trojans.py --gds-dir A/val --sem-dir C/val \
    --out-dir sets/s1 --rate 0.7 --seed 1
python trojan/scripts/inject_trojans.py ... --out-dir sets/s2 --seed 2
# 2. export to YOLO format (clean images included as negatives)
python trojan/scripts/export_yolo_dataset.py --inputs sets/s1 sets/s2 --out yolo_ds
# 3. train (base weights staged locally for offline)
python trojan/scripts/train_yolo.py --data yolo_ds/dataset.yaml \
    --weights /models/yolo11s.pt --epochs 150 --imgsz 512
# 4. detect with the trained model
python trojan/scripts/detect_trojans.py --root INPUT --out D \
    --backend yolo --weights runs/troj_yolo/weights/best.pt
```

RF-DETR drops in the same way — add a branch in `detect_yolo()`; the dataset
exporter already emits standard normalized boxes it can consume. Generate as
many labelled images as you like: the injector is the data engine, so the
learned detector's training set is unlimited even though you only have 74
real pairs.

## Running it

### Build + transfer (online → offline)

The detection image is built and shipped by the same script as the others:

```bash
./offline_prep/build_and_export_images.sh ./transfer   # now also builds gds2sem-trojan:v1
# offline:
docker load < transfer/images/gds2sem-trojan_v1.tar.gz
```

For the YOLO backend, also stage a base checkpoint (e.g. `yolo11s.pt`) under
your data root while online.

### Command line (no LibreChat)

`screen.py` is the one-command front-end; the individual scripts below it
remain available for finer control.

```bash
# screen a directory that has A/ B/ C/ subdirs (C = suspect), and write a
# self-contained report.html (summary + annotated images inline) next to
# results.json:
python trojan/scripts/screen.py detect --root INPUT_DIR --out D --report

# end-to-end dry run on your own data: inject trojans into clean generated
# SEMs, screen them, score vs ground truth, write the report:
python trojan/scripts/screen.py demo --root gds_2_sem --out demo_run

# re-score an existing run:
python trojan/scripts/screen.py eval --truth testset/ground_truth.json \
    --results D/results.json
```

Containerized one-shot (all paths must be under the /data mount):

```bash
DATA=/srv/trojan_data ./trojan/run_screen.sh detect \
    --root /data/incoming/lot42 --out /data/runs/lot42_D --report
```

And if the MCP service is already running (run_detector_mcp.sh), any
machine that can reach it can drive the same tools from the shell — this is
how a remote requester submits a job and gets results without LibreChat:

```bash
python trojan/scripts/screen.py remote --server http://HOST:8130/mcp patterns
python trojan/scripts/screen.py remote --server http://HOST:8130/mcp detect \
    --input-dir incoming/lot42          # path under the service's data root
python trojan/scripts/screen.py remote --server http://HOST:8130/mcp fetch \
    --image runs/lot42_D/annotated/img.png --save img.png
```

The remote mode needs only `pip install mcp` on the calling machine; the D
output stays on the service host under its data root, per the design.

Lower-level scripts, if you want the pieces individually:

```bash
python trojan/scripts/inject_trojans.py --gds-dir gds_2_sem/A/val \
    --sem-dir gds_2_sem/C/val --out-dir testset --round-robin --rate 0.6
python trojan/scripts/detect_trojans.py --root INPUT_DIR --out D
python trojan/scripts/eval_detection.py --truth testset/ground_truth.json \
    --results D/results.json
```

`D/results.json` lists, per image, either `no_trojan_detected` or every
detected pattern with its box and confidence; `D/annotated/` holds the
images with boxes drawn (green = addition, orange = bridge, blue =
modification, red = deletion).

### LibreChat (Opus 5 + MCP)

1. Launch the detector as an MCP service on the offline host (own GPU):

   ```bash
   DATA=/srv/trojan_data GPU=1 ./trojan/run_detector_mcp.sh
   ```

   `DATA` is the only tree the agent can read/write. A remote requester who
   "sends a directory of images" drops it under `DATA` (e.g.
   `DATA/incoming/lot42/{A,B,C}`); the `D` output is written back under
   `DATA/runs/` on that same host.

2. Point LibreChat at it with `trojan/librechat/librechat.yaml` (registers
   the `trojan-detector` MCP server and an Opus-5 endpoint), and create an
   agent with `trojan/librechat/agent_instructions.md` as its system prompt.

3. In chat: give the agent the input directory path. It calls
   `detect_trojans`, summarises the verdict, and calls `show_detection` to
   display each flagged image with its boxes inline. `list_trojan_patterns`
   explains the taxonomy; `inject_trojans` (demo only) builds test sets.

The MCP server exposes four tools — `list_trojan_patterns`,
`detect_trojans`, `show_detection`, `inject_trojans` — all sandboxed to the
data root, all offline.
