# sem-trojan-detect

Hardware-trojan screening for SEM images of manufactured chips.

Give it the golden model you already hold — the GDS layout (**A**) and the
original known-good SEM (**B**) — plus the SEM you captured from the chip
that came back (**C**). It flags every region of C that differs from golden,
classifies each into one of ten trojan patterns (A–J), and writes a **D**
output: a JSON verdict per image, annotated images with bounding boxes, and
a self-contained HTML report.

Runs fully offline. Drive it from the command line, or from LibreChat with
an Opus-5 agent through the bundled MCP server.

> This repo is **standalone**. It has no source dependency on the
> [gds2sem](#relationship-to-gds2sem) generator — when it needs SEM images
> rendered from GDS layouts it calls that tool over its HTTP API, so the two
> deploy and version independently.

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

Classes drive triage: **additions** and **bridges** add material,
**deletions** remove it, **modifications** alter a feature in place. H is
the hardest — geometry is unchanged, so it is only visible by comparing
intensity against B. Without B, dopant-class trojans cannot be detected.

## Layout

```
sem-trojan-detect/
├── trojanlib/                  the package (importable, self-contained)
│   ├── imagelib.py             image primitives (cv2 fast paths + numpy fallbacks)
│   ├── patterns.py             the 10 patterns + SEM-texture synthesis
│   ├── inject.py               build labelled test sets
│   ├── detect.py               the detector (golden + yolo backends)
│   ├── evaluate.py             score detections vs ground truth
│   ├── report.py               self-contained HTML report
│   ├── matcher.py              B vs C cell/connectivity matching + report
│   ├── gds_trojans.py          5 layout trojan patterns (A–E) + injector
│   ├── gds2sem_client.py       calls the gds2sem service over HTTP
│   └── llm_client.py           Claude via your Open WebUI instance
├── scripts/
│   ├── screen.py               the CLI (detect/demo/eval/inject/generate/llm/remote)
│   ├── screen_matcher.py       B vs C difference report
│   ├── doctor.py               environment diagnostic (stdlib only)
│   ├── inject_gds_trojans.py   stamp trojan regions into GDS layouts
│   ├── export_yolo_dataset.py  injected sets -> YOLO dataset
│   └── train_yolo.py           train the optional YOLO backend
├── mcp/server.py               MCP server (LibreChat / remote CLI)
├── librechat/                  librechat.yaml + .env.example + agent instructions
├── docker/                     Dockerfile + requirements (cu126, offline)
├── offline_prep/
│   ├── build_and_export_images.sh   online: build + save the image
│   ├── install_local.sh             offline: venv install, no docker
│   └── verify_setup.sh              offline: check everything is in place
├── run_screen.sh               one-shot containerized CLI run
└── run_detector_mcp.sh         run the MCP service (for LibreChat / remote)
```

## Offline setup

Build online (needs PyPI, no GPU), transfer, load:

```bash
# ONLINE
./offline_prep/build_and_export_images.sh ./transfer
# WITH_TORCH=0 ./offline_prep/build_and_export_images.sh ./transfer   # slimmer,
#   golden backend only, no YOLO (~4 GB instead of ~8 GB)

# OFFLINE
docker load < transfer/images/sem-trojan-detect_v1.tar.gz
./offline_prep/verify_setup.sh ./transfer
```

`verify_setup.sh` checks the image loads, its dependencies import, docker
can see the GPUs, the data root exists, and — as optional notes — whether
YOLO base weights are staged and whether the gds2sem service is reachable.

**No model weights are required.** The default golden-model backend is
deterministic image processing; it needs nothing but the image. Weights are
only involved if you opt into the YOLO backend, in which case stage a base
checkpoint (e.g. `yolo11s.pt`) into `transfer/models/` while online.

Host prerequisites: `docker`, and `nvidia-container-toolkit` only if you
want GPU for YOLO. The golden backend is CPU-only.

## Command line

```bash
# screen a directory containing A/ B/ C/ (C = suspect), write D + report
python3 scripts/screen.py detect --root /data/incoming/lot42 \
    --out /data/runs/lot42_D --report

# end-to-end dry run on your own data: inject -> screen -> score -> report
python3 scripts/screen.py demo --root /data/gds_2_sem --out /data/runs/demo

# pieces on their own
python3 scripts/screen.py inject --gds-dir A/val --sem-dir C/val \
    --out-dir testset --round-robin
python3 scripts/screen.py eval --truth testset/ground_truth.json \
    --results D/results.json
```

Containerized (all paths must be under the `/data` mount):

```bash
DATA=/srv/trojan_data ./run_screen.sh detect \
    --root /data/incoming/lot42 --out /data/runs/lot42_D --report
```

`D/results.json` gives, per image, either `no_trojan_detected` or every
detected pattern with box and confidence. `D/annotated/` holds the boxed
images (green = addition, orange = bridge, blue = modification, red =
deletion). `D/report.html` is a single file with the summary and every
flagged image embedded — openable on an air-gapped box, no server.

## Layout trojans → SEM → detection

A second, layout-level workflow that produces labelled training data and
exercises the matcher end to end. Where `screen.py detect` classifies single
features against the GDS golden model, this pair works on **regions**: a
group of neighbouring cells in which something was added, removed, bridged
or cut.

```
clean layouts ──gds2sem──► SEM = B (golden)
      │
      └─inject_gds_trojans──► tampered layouts ──gds2sem──► SEM = C (suspect)
                                                               │
                        screen_matcher.py  B vs C  ◄───────────┘
```

### 1. Inject trojans into the layouts

```bash
python3 scripts/inject_gds_trojans.py --gds-dir gds_2_sem/A/train \
    --out-dir A_trojan --rate 0.7 --max-per-image 2
```

Five patterns, cycled round-robin so a 70-image training set gets a balanced
spread (~13–16 regions each):

| label | pattern | what it does to the layout |
|---|---|---|
| **Trojan A** | inserted_cluster | 2–3 new cells placed in the group's whitespace |
| **Trojan B** | depopulated | 1–2 existing cells deleted from the group |
| **Trojan C** | merged_pair | two adjacent cells bridged into one |
| **Trojan D** | severed_net | one cell cut into two separated pieces |
| **Trojan E** | rerouted_block | mixed — one added, one removed, one pair bridged |

Each region is the modified cells **plus their untouched neighbours**, which
is the unit an analyst is actually asked to find. `gds_trojans.json` records
every region box, its label, and the individual edits — ground truth for
scoring, and the source for a YOLO training set.

`--patterns CD` restricts to a subset; `--group-size` sets how many cells
form a region; `--rate` how many layouts get tampered at all.

### 2. Compare the SEMs

```bash
python3 scripts/screen_matcher.py --root /data/incoming/lot42 \
    --out /data/runs/lot42_M
# score the detections against the layout ground truth:
python3 scripts/screen_matcher.py --b-dir B --c-dir C --out run \
    --truth A_trojan/gds_trojans.json
```

Each image is reduced to its **cells**, and the B and C cell sets are linked
into an overlap graph. Every component of that graph says what happened:

| overlay | meaning |
|---|---|
| **red** | a cell in B, **missing from C** — material removed |
| **green** | a cell in C, **absent from B** — material gained |
| **light blue** | connectivity changed — cells that should be **connected** (a net was severed) or **separated** (two were bridged) |
| **yellow box** | a trojan region: nearby anomalies grouped and labelled *Trojan A … Trojan E* |

The overlay puts **B on top of C** — C is the base, B blended over it at
`--alpha`, anomalies tinted, regions boxed and labelled.

Linking uses **containment**, not IoU: a merged cell is far larger than
either cell it swallowed, so their IoU is low while containment is near 1.
That is what makes merges and splits detectable at all rather than showing
up as a confusing mix of missing and gained.

Regions are labelled from the mix of changes inside them — only additions →
Trojan A, only removals → B, only merges → C, only splits → D, any mixture →
E — which is one-to-one with the five injector recipes, so the injector
doubles as ground truth without the detector ever seeing it.

> **Colour change.** Red now marks missing and green marks gained, matching
> gds2sem's `overlay_compare`. Earlier builds of this script had them the
> other way round; `--legacy-colors` restores that.

### The accuracy score

```
accuracy = matched / (matched + missing + gained + connectivity)
```

Scored on cells rather than pixels, so every changed cell costs the same
regardless of area — a hair-thin added route weighs as much as a large
block. Pixel IoU sits beside it as an area-weighted second opinion; the two
diverge exactly when differences are small in area but many in number, which
is what an insertion looks like.

With `--truth`, the report also scores detected regions against the layout
ground truth: recall, precision, label accuracy and an A–E confusion matrix.
On a 70-layout synthetic run (51 tampered, 19 clean) the current defaults
give **89% region recall, 100% precision, 86% label accuracy**, with zero
false alarms on the clean images. The residual label errors are almost all
Trojan E read as A or B — when only part of a mixed recipe lands close
enough to cluster, the region honestly only contains one kind of change.

### Output

| file | what it is |
|---|---|
| `match_report.html` | self-contained: **every B and C image** plus the B-over-C overlay, summary tiles, the accuracy score, the trojan-region table and a per-image table sorted worst-first |
| `match_results.json` | the same numbers plus every anomaly and region box |
| `overlays/*.png` | the composited overlays on their own |

Images embed as JPEG so a lot-sized report stays openable — 12 pairs is
about 2 MB. `--lossless` gives pixel-exact PNG; `--thumb-width` shrinks.

### Tuning

`--link-cover` (default 0.35) is the containment above which a B cell and a
C cell are linked. **Lower it** if a slightly shifted capture reports paired
false missing/gained cells — that pattern (a red and a green cell in the
same spot) means one real cell failed to link to itself. `--group-gap`
(default 56) is how close anomalies must be to join one region: raise it if
one trojan is being reported as several, lower it if separate trojans are
merging into one. `--tolerance` adds px of slack to the overlap test, and
`--min-area` drops specks.

## Troubleshooting installs

### What this actually depends on

**numpy and Pillow. That is the whole hard requirement.** Everything else is
optional and degrades cleanly:

| package | needed for | if absent |
|---|---|---|
| `opencv-python-headless` | faster components / morphology / blur | `trojanlib.imagelib` uses pure-numpy fallbacks — same results, slower |
| `mcp` | the MCP server and `screen.py remote` | local CLI still works |
| `torch` + `ultralytics` | the YOLO detection backend only | golden backend (the default) still works |

The core is exercised against **numpy 1.26 through 2.4**, so whatever version
your mirror carries is almost certainly fine. `docker/requirements.txt` uses
ranges rather than exact pins for exactly that reason.

### "Cannot install numpy 2.1.3, 1.26.4 is already installed" at build time

Ubuntu 24.04's system Python is *externally managed* and its numpy belongs to
dpkg, not pip — pip cannot uninstall it (there is no RECORD file), so any
`pip install numpy==X` into the system interpreter fails. Installing apt's
`python3-opencv` is what pulls that numpy in.

Fixed in the image by **installing everything into a virtualenv at
`/opt/venv`** and no longer installing `python3-opencv` from apt. pip owns its
own site-packages there, so nothing is fought over and
`--break-system-packages` is gone.

If you are building against an internal mirror:

```bash
PIP_INDEX=https://pypi.internal/simple ./offline_prep/build_and_export_images.sh ./transfer
# leaner builds, if a package is missing from your mirror:
WITH_TORCH=0 WITH_CV2=0 ./offline_prep/build_and_export_images.sh ./transfer
```

### "ModuleNotFoundError: No module named 'numpy'" when running screen.py

Almost always the *wrong interpreter* — packages installed for one Python,
script run under another. Ask the doctor, which is stdlib-only so it still
runs when numpy is missing:

```bash
python3 scripts/doctor.py                     # on the host
./run_screen.sh doctor                        # inside the image
docker run --rm sem-trojan-detect:v1 python /app/scripts/doctor.py
```

It prints which interpreter is running, whether it is a virtualenv, and where
each dependency was found — apt's `dist-packages` versus pip's
`site-packages`. Seeing both is the tell for the conflict above.

### Running on the offline host without docker

```bash
./offline_prep/install_local.sh --index-url https://pypi.internal/simple
# optional extras, each skipped with a warning if your mirror lacks it:
./offline_prep/install_local.sh --index-url https://pypi.internal/simple \
    --with-cv2 --with-mcp
```

This builds a virtualenv at `./.venv`, installs only numpy and Pillow by
default, and finishes by running the doctor. Then use that interpreter:

```bash
.venv/bin/python scripts/screen_matcher.py --root /data/lot42 --out /data/runs/M
```

`--extra-index-url`, `--trusted-host` and `--find-links` are passed through,
so a local wheelhouse works too:
`./offline_prep/install_local.sh --find-links /mnt/wheels --index-url ""`.

## LibreChat (Opus 5 + MCP)

1. Run the MCP service on the offline host:

   ```bash
   DATA=/srv/trojan_data GPU=1 ./run_detector_mcp.sh
   ```

   `DATA` is the only tree the agent can read or write. A remote requester
   drops a directory under it (e.g. `DATA/incoming/lot42/{A,B,C}`); the D
   output is written back under `DATA/runs/` on that same host.

2. Point LibreChat at it with `librechat/librechat.yaml` (registers the
   `trojan-detector` MCP server and an Opus-5 endpoint), and create an agent
   whose system prompt is `librechat/agent_instructions.md`.

3. In chat, give the agent the input directory. It calls `detect_trojans`,
   summarises the verdict, and calls `show_detection` to display each
   flagged image with its boxes inline.

Tools: `list_trojan_patterns`, `detect_trojans`, `match_sems`,
`inject_gds_trojans`, `show_detection`, `inject_trojans`, `generate_sem`,
`summarize_run` — all sandboxed to the data root.

Any machine that can reach the service can also drive it from a shell,
without LibreChat:

```bash
python3 scripts/screen.py remote --server http://HOST:8130/mcp patterns
python3 scripts/screen.py remote --server http://HOST:8130/mcp detect \
    --input-dir incoming/lot42
python3 scripts/screen.py remote --server http://HOST:8130/mcp fetch \
    --image runs/lot42_D/annotated/img.png --save img.png
```

Remote mode needs only `pip install mcp` on the calling machine.

## Claude access through Open WebUI

Claude models are reached through your existing **Open WebUI** instance,
which exposes an OpenAI-compatible API — nothing here talks to Anthropic
directly, so this works on an internal network. Both front-ends use it:
LibreChat as its model endpoint, and the CLI for optional analyst summaries.

Create a token in Open WebUI under **Settings → Account → API Keys** (the
instance must have API keys enabled). Tokens look like `sk-...`.

### Command line

Save the token once (written to `~/.config/sem-trojan-detect/config.json`,
chmod 600), or supply it per-run:

```bash
python3 scripts/screen.py llm login --url http://webui.internal:3000 --api-key sk-...
python3 scripts/screen.py llm test          # verify connectivity + list Claude models
python3 scripts/screen.py llm models        # everything this token can use
```

Resolution order is `--api-key/--url` → `OPENWEBUI_API_KEY`/`OPENWEBUI_URL`
→ the saved config, so CI can pass env vars and never touch the file.
Override the config path with `SEM_TROJAN_CONFIG`.

Then add `--summarize` to a screening run to get an analyst triage narrative
written by Claude from the detector's findings — printed, saved as
`summary.md`, and embedded in `report.html`:

```bash
python3 scripts/screen.py detect --root /data/incoming/lot42 \
    --out /data/runs/lot42_D --report --summarize
# or after the fact, on a finished run:
python3 scripts/screen.py llm summarize --results /data/runs/lot42_D/results.json
```

`--model` picks a specific model id; the default auto-selects a Claude model
from the instance (preferring Opus). **The summary is strictly optional** —
if Open WebUI is unreachable or the token is rejected, the screening still
completes and the report is still written; only the narrative is skipped,
with the reason printed.

The containers take the same settings as environment variables — the run
scripts forward `OPENWEBUI_URL` and `OPENWEBUI_API_KEY`, and
`SEM_TROJAN_CONFIG` defaults to `/data/.openwebui.json` so a token saved
into the data root persists:

```bash
OPENWEBUI_URL=http://webui.internal:3000 OPENWEBUI_API_KEY=sk-... \
  DATA=/srv/trojan_data ./run_screen.sh detect --root /data/incoming/lot42 \
      --out /data/runs/lot42_D --report --summarize
```

A key is never printed, logged, or written into a report — only a masked
form (`sk-abc…7890`) ever appears.

### LibreChat

`librechat/librechat.yaml` registers Open WebUI as a custom endpoint.
LibreChat appends `/chat/completions` and `/models` to the configured
baseURL, which lines up with Open WebUI's `/api/...`, so the baseURL is the
instance root plus `/api`:

```yaml
endpoints:
  custom:
    - name: "OpenWebUI"
      apiKey: "user_provided"          # each analyst pastes their own token
      baseURL: "${OPENWEBUI_URL}/api"
      models:
        default: ["claude-opus-5"]
        fetch: true
```

Set `OPENWEBUI_URL` in LibreChat's `.env` (see
`librechat/.env.example`). With `apiKey: "user_provided"` each user enters
their own Open WebUI token in the LibreChat UI, so usage stays attributable
per analyst and no shared secret sits in a file; switch it to
`"${OPENWEBUI_API_KEY}"` if you would rather use one service token.

## Relationship to gds2sem

[gds2sem](../gds2sem-flux2-klein) is a separate tool that renders SEM-style
images from GDS layouts with a FLUX.2 Klein LoRA. This repo calls it over
the ComfyUI HTTP API — never by importing its code — so neither repo
constrains the other's dependencies or release cycle.

You need it running only for these cases:

* `screen.py generate` — render SEM images from layouts, to build test sets,
  or to synthesise a golden SEM baseline for a region where you hold the
  layout but have no known-good capture (which is what makes dopant-class
  detection possible there).
* `screen.py demo` — auto-invoked if the demo needs generated SEMs and none
  exist.

```bash
# in the gds2sem repo, on the offline host:
COMFY_MODELS=$PWD/transfer/comfy_models GPU=1 ./inference/run_comfyui.sh

# here:
python3 scripts/screen.py generate --gds-dir /data/gds_2_sem/A/val \
    --out-dir /data/gds_2_sem/C/val --server http://localhost:8188
```

Containers pick the address up from `GDS2SEM_SERVER` (the run scripts
default to `http://host.docker.internal:8188`).

**Production screening does not need gds2sem at all** — there, C comes off a
real microscope and A/B are your existing golden model.

## How detection works

### Golden backend (default, no training)

1. Binarise C (Otsu); build the golden material mask from A, upscaled to C's
   size (465→512, nearest-neighbour so edges stay crisp), with B supplying
   the intensity baseline.
2. Difference against golden with a few px of tolerance: *added* = material
   in C not in golden, *removed* = golden material absent from C,
   *intensity* = shared material whose brightness in C departs from B.
3. Label connected anomaly regions; for each, compute geometric and
   photometric features (add/remove/intensity fractions, aspect, size, bbox
   fill, how many golden features it touches, L-shape arms).
4. Classify with a calibrated decision tree, draw class-coloured boxes, and
   write the results.

Tunable via `--tolerance`, `--intensity-delta`, `--merge`, `--min-area`,
`--min-conf`.

On the bundled synthetic evaluation (12 images, all ten patterns injected):
image-level precision/recall **1.00 / 1.00**, instance localisation F1
**0.97**, classification **94%** of localised regions. **These are synthetic
numbers** — injected features share the renderer's statistics. Real captures
carry more golden-diff noise along every edge, so treat flags as "regions to
review", expect to tune the thresholds to your images, and consider the
learned backend for production classification.

### YOLO backend (optional, learned)

For single-image detection with no golden reference at inference, or higher
accuracy on subtle patterns. The injector is an unlimited data engine, so
the training set is not limited by how many real pairs you have.

```bash
python3 scripts/screen.py inject --gds-dir A/val --sem-dir C/val \
    --out-dir sets/s1 --rate 0.7 --seed 1
python3 scripts/screen.py inject ... --out-dir sets/s2 --seed 2
python3 scripts/export_yolo_dataset.py --inputs sets/s1 sets/s2 --out yolo_ds
python3 scripts/train_yolo.py --data yolo_ds/dataset.yaml \
    --weights /models/yolo11s.pt --epochs 150 --imgsz 512
python3 scripts/screen.py detect --root INPUT --out D \
    --backend yolo --weights runs/troj_yolo/weights/best.pt
```

RF-DETR drops in the same way — add a branch in `trojanlib.detect`; the
exporter already emits standard normalised boxes.

## Using it as a library

```python
from trojanlib import screen_directory, inject_directory, evaluate, write_report

report = screen_directory("data/A", "data/B", "data/C", "out/D")
print(report["summary"])          # {'images': 12, 'flagged': 9, ...}

inject_directory("data/A", "data/C", "testset", rate=0.6, round_robin=True)
text, stats = evaluate("testset/ground_truth.json", "out/D/results.json")
write_report("out/D", text)
```

`trojanlib.imagelib` uses OpenCV when importable and falls back to pure
numpy otherwise, so the package runs anywhere numpy + Pillow do.
