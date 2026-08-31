# GDS → SEM style transfer with FLUX.2 Klein 4B

End-to-end workflow for LoRA-finetuning **FLUX.2 [klein] 4B** on 74 paired
GDS/SEM images (465×465) so that, at inference, you feed a simulated GDS
standard-cell layout and get back an SEM-style rendering that preserves every
rectangle. Training uses **ostris/ai-toolkit**; inference uses **ComfyUI**.
Both run in separate Docker images, built online and transferred to your
offline dual-RTX-6000-Ada machine.

## Why this approach works

FLUX.2 Klein is natively an *image-editing* model: it accepts reference
images through `ReferenceLatent` conditioning. So instead of a plain style
LoRA (text → image), you train an **edit LoRA** on a *paired* dataset —
ai-toolkit's `control_path` mechanism feeds the GDS image as the control
input and the SEM image as the training target, with an instruction-style
caption. The model learns "given this layout, re-render it as SEM," which is
exactly the structure-preserving mapping you want. You train on the
**undistilled base** model (`FLUX.2-klein-base-4B`) and can run inference on
either the base (20 steps, CFG ~4, best fidelity) or the **distilled**
model (4 steps, CFG 1, ~1s/image) — LoRAs trained on base transfer to
distilled.

Key facts verified against ai-toolkit source:

| Piece | Value |
|---|---|
| ai-toolkit arch | `flux2_klein_4b` |
| Transformer weights | local file `flux-2-klein-base-4b.safetensors` (undistilled base, ~7.8 GB) |
| Text encoder | local dir `qwen3-4b/` — full transformers-format Qwen3-4B (config + tokenizer + weight shards) |
| VAE | local file `ae.safetensors` (or `flux2-vae.safetensors`), BFL single-file format |
| Paired dataset | `datasets[].folder_path` = targets (SEM) + captions, `datasets[].control_path` = controls (GDS), matched by filename |
| Sample-time control | `--ctrl_img /path.png` appended to sample prompts |

**No Hugging Face anywhere.** Neither container touches the hub or an HF
cache: ComfyUI loads plain files from its models folders, and training runs
through `training/scripts/run_offline.py`, which points ai-toolkit at plain
local paths (ai-toolkit resolves the transformer and VAE from a local
`name_or_path` directory natively; the text-encoder location is hardcoded
to a hub id in its source, so the launcher patches that class attribute to
your local `qwen3-4b/` directory before starting). `HF_HUB_OFFLINE=1` /
`TRANSFORMERS_OFFLINE=1` are set as tripwires so any accidental hub call
fails immediately instead of hanging. The `transformers`/`huggingface_hub`
*libraries* stay installed inside the images (ai-toolkit imports them), but
they never use the network or a cache.

## Critical hardware constraint

Your driver is **560.35.05 → CUDA 12.6 max**. Upstream ai-toolkit's
Dockerfile uses cu130 wheels that require driver ≥ 580 and will *not* run.
Both Dockerfiles here pin `torch==2.13.0+cu126` (official cu126 wheels
exist for torch 2.13) and verify the build at image-build time. RTX 6000
Ada (sm_89) is fully supported by cu126. The base image is
`nvidia/cuda:12.6.3-runtime-ubuntu24.04`; the wheels bundle their own CUDA
libs, so the host only needs the driver + `nvidia-container-toolkit`.

ai-toolkit is single-GPU per job — with two 48 GB cards you can run two
training configs in parallel (`GPU=0` / `GPU=1`), or train on GPU 0 while
ComfyUI serves on GPU 1. A 4B LoRA at 512px in full bf16 (no quantization)
uses well under 48 GB, so the config disables quantization for best quality.

## Repository layout

```
gds2sem-flux2-klein/
├── README.md
├── offline_prep/
│   ├── build_and_export_images.sh   # online: build + save both docker images
│   └── verify_models.sh             # offline: check the model file layout
├── training/
│   ├── Dockerfile                   # ai-toolkit, torch cu126
│   ├── config/gds2sem_klein4b.yaml  # edit-LoRA config, wired to gds_2_sem/
│   ├── scripts/setup_dataset.py     # validate pairs, captions, patch config
│   ├── scripts/run_offline.py       # local-weights launcher (no HF)
│   └── run_training.sh              # docker run wrapper (offline)
├── inference/
│   ├── Dockerfile                   # ComfyUI, torch cu126
│   ├── run_comfyui.sh               # docker run wrapper (offline)
│   ├── workflows/
│   │   ├── gds2sem_klein4b_base_lora.json       # 20 steps, CFG 4
│   │   └── gds2sem_klein4b_distilled_lora.json  # 4 steps, CFG 1
│   └── scripts/batch_infer.py       # headless folder→folder conversion via API
├── eval/
│   ├── eval_structure.py            # IoU + SSIM checkpoint scoring
│   ├── overlay_compare.py           # visual C-over-A overlay + diff maps
│   ├── matchlib.py                  # shared image helpers (cv2 + numpy paths)
│   ├── match_gds_sem.py             # per-cell A→C matcher, green/red @ threshold
│   └── match_sem_sem.py             # B→C SEM matcher, green/red @ threshold
└── trojan/                          # hardware-trojan screening (see trojan/README.md)
    ├── scripts/                     # patterns, inject, detect, eval, YOLO export/train
    ├── mcp/server.py                # MCP server for LibreChat
    ├── librechat/                   # librechat.yaml + agent instructions
    ├── docker/                      # detection + MCP image (cu126, offline)
    └── run_detector_mcp.sh
```

## Step 1 — Build images (online) and lay out your model files

The only thing the online machine is needed for is building the two docker
images (GitHub + PyPI access, no Hugging Face, no GPU):

```bash
cd gds2sem-flux2-klein
./offline_prep/build_and_export_images.sh ./transfer   # docker images (~15-20 GB)
```

Arrange the model files you already have into this layout on the transfer
share — plain files, no HF cache structure:

```
transfer/
├── images/                              docker tars (from the build script)
├── models/flux2-klein/                  TRAINING weights
│   ├── flux-2-klein-base-4b.safetensors   undistilled base transformer
│   ├── ae.safetensors                     FLUX.2 VAE, BFL single-file format
│   │                                      (a file named flux2-vae.safetensors
│   │                                       is auto-detected too)
│   └── qwen3-4b/                          text encoder — must be the FULL
│       ├── config.json                    transformers-format directory:
│       ├── tokenizer.json ...             config, tokenizer files, and the
│       └── model*.safetensors (+index)    weight shards. The single-file
│                                          ComfyUI qwen_3_4b.safetensors
│                                          canNOT be used for training.
└── comfy_models/                        INFERENCE weights (as before)
    ├── diffusion_models/flux-2-klein-4b-fp8.safetensors
    ├── diffusion_models/flux-2-klein-base-4b-fp8.safetensors
    ├── text_encoders/qwen_3_4b.safetensors
    ├── vae/flux2-vae.safetensors
    └── loras/
```

Then, on the offline machine:

```bash
docker load < transfer/images/gds2sem-train_v1.tar.gz
docker load < transfer/images/gds2sem-comfy_v1.tar.gz
./offline_prep/verify_models.sh ./transfer     # confirms every file is in place
```

Offline host prerequisites (from your internal apt mirror): `docker`,
`nvidia-container-toolkit`. If your files carry different names, rename or
symlink them into the layout above — `verify_models.sh` tells you exactly
what's missing. The training launcher also runs its own preflight at start,
including a check that the VAE file really is the BFL-format FLUX.2 VAE
(it reads only the safetensors header, so the check is instant).

## Step 2 — Point training at gds_2_sem

Training consumes your existing dataset layout directly — no images are
copied, moved, or resized:

```
gds_2_sem/
├── A/            GDS images (controls)
│   ├── train/
│   └── val/
└── B/            SEM images (targets)
    ├── train/
    └── val/
```

The config sets `folder_path: B/train` (targets) and `control_path:
A/train` (controls); ai-toolkit pairs them by filename stem (`A/train/x.png`
↔ `B/train/x.png`, extensions may differ). The raw 465×465 sizes are fine:
targets are bucketed to 512 by the dataloader, and for FLUX.2 the GDS
control images are passed through raw and center-cropped to a multiple of 16
(465→464, a 1-px border) inside the pipeline.

Run the setup script once:

```bash
mkdir -p workspace && cp training/config/gds2sem_klein4b.yaml workspace/config.yaml
python3 training/scripts/setup_dataset.py --root /path/to/gds_2_sem \
    --config workspace/config.yaml
```

It validates that every A/train image has its B/train partner (and same for
val), writes one orientation-aware instruction caption (`.txt`, containing
the `g2s3m` trigger) next to each SEM image in B/train — additive only,
skip with `--no-captions` to use the config's `default_caption` instead —
and rewrites the config's sample `prompts:` block to reference your actual
`A/val` filenames, so train-time samples are generated from held-out GDS
inputs.

Only ~70 training pairs is workable for this task because the mapping is
narrow and highly regular (one style, one content family), but watch the
samples for overfitting — the A/val samples during training are your
early-warning signal.

## Step 3 — Train

```bash
DATASET=/path/to/gds_2_sem WORKSPACE=$PWD/workspace \
    MODELS=$PWD/transfer/models/flux2-klein GPU=0 ./training/run_training.sh
```

The dataset is mounted at `/workspace/gds_2_sem` inside the container
(read-write — `cache_latents_to_disk` writes a latent-cache folder inside
the dataset directories; delete those `_latent_cache` folders if you ever
change the images). The model directory is mounted read-only at
`/models/flux2-klein`.

The container runs `python /app/run_offline.py /workspace/config.yaml`:
after its preflight it patches ai-toolkit's text-encoder path to
`/models/flux2-klein/qwen3-4b` and hands off to the normal `run.py` — every
weight loads from the mounted directory, nothing is fetched or cached.
(Non-default locations can be set with `FLUX2_MODEL_DIR`, `FLUX2_TE_DIR`,
and `FLUX2_VAE_FILE`.) Checkpoints and sample grids appear in
`workspace/output/gds2sem_klein4b_v1/` every 250 steps.

What to expect and tune:

- **Checkpoint choice.** Quality typically peaks around steps 1500–2500,
  before the final step. Pick by looking at the val samples (they use
  held-out GDS inputs) and by scoring with `eval/eval_structure.py`.
- **Rank** (`network.linear`): 32 to start. Raise to 64 if SEM texture is
  weak; drop to 16 if rectangles start drifting or merging.
- **LR** 1e-4 with adamw8bit is the standard klein recipe; 8e-5 if samples
  get noisy/unstable.
- **Batch size** 2 fits comfortably; you can raise it, but with 70 images
  more steps at small batch generally beats fewer steps at large batch.
- Run a second config variant on GPU 1 in parallel (`GPU=1`, separate
  `WORKSPACE`) to A/B rank or LR in one wall-clock pass.

When done, copy the best
`workspace/output/gds2sem_klein4b_v1/gds2sem_klein4b_v1_000001750.safetensors`
(or whichever step wins) to `transfer/comfy_models/loras/gds2sem_klein4b_v1.safetensors`.

## Step 4 — Inference in ComfyUI

```bash
COMFY_MODELS=$PWD/transfer/comfy_models GPU=1 ./inference/run_comfyui.sh
```

Open `http://<host>:8188`, drag in
`inference/workflows/gds2sem_klein4b_base_lora.json`. The graph is the
official Comfy-Org Klein 4B image-edit graph flattened, plus a
`LoraLoaderModelOnly` node:

```
UNETLoader → LoraLoaderModelOnly ─────────────→ CFGGuider ─→ SamplerCustomAdvanced → VAEDecode → SaveImage
CLIPLoader → CLIPTextEncode(pos/neg) → ReferenceLatent ↗        ↑ ↑ ↑
LoadImage(GDS) → ImageScale 512 → VAEEncode ──↗   RandomNoise / KSamplerSelect(euler) / Flux2Scheduler / EmptyFlux2LatentImage
```

Load a GDS image in the `LoadImage` node and queue. Two variants:

- **base** (`flux-2-klein-base-4b-fp8`): 20 steps, CFG 4 — matches the
  training distribution, best structure fidelity. Start here.
- **distilled** (`flux-2-klein-4b-fp8`): 4 steps, CFG 1 — near-instant.
  Verify fidelity holds; if not, stay on base.

If the LoRA loads with a "no keys matched" style warning, your ComfyUI
checkout predates FLUX.2-klein LoRA support — rebuild the inference image
with a newer `COMFY_REF`.

For folder-scale conversion without touching the UI:

```bash
python3 inference/scripts/batch_infer.py --input-dir /path/to/gds_2_sem/A/val \
    --server http://localhost:8188 --variant base \
    --lora gds2sem_klein4b_v1.safetensors
```

Results land in `comfy_data/output/gds2sem/`.

## Step 5 — Evaluate checkpoints

```bash
python3 eval/eval_structure.py \
    --gds-dir /path/to/gds_2_sem/A/val \
    --gen-dir comfy_data/output/gds2sem \
    --ref-dir /path/to/gds_2_sem/B/val
```

Reports per-image IoU between the binarized GDS layout and the binarized
generation (rectangle-preservation score — this is the number that matters
for your use case) plus SSIM against the real SEM. Sweep your saved LoRA
steps through batch inference and keep the checkpoint with the best
IoU/visual balance.

## Step 6 — Visual overlay comparison (C over A)

Store the FLUX-generated images under `gds_2_sem/C/train` and
`gds_2_sem/C/val`, mirroring A and B (filenames matched by stem; ComfyUI
counter suffixes like `_00001_` are stripped automatically, and C's 512px
outputs are resized back to A's dimensions). Then:

```bash
python3 eval/overlay_compare.py --root /path/to/gds_2_sem --split val
# both splits, custom transparency:
python3 eval/overlay_compare.py --root /path/to/gds_2_sem --split both --alpha 0.55
```

For every pair it writes three images into `comparisons/<split>/`:

- `overlay_<name>.png` — C blended semi-transparently over A (`--alpha`
  sets C's opacity, default 0.55), so the original rectangles show through
  for a direct visual match check.
- `diff_<name>.png` — structure difference map: rectangle pixels present
  in both A and C stay **white**, additional pixels that C introduced are
  **green**, pixels from A that C failed to reproduce are **red**, and
  shared background stays black. A is binarized at mid-gray; C (a grayscale
  SEM rendering) is binarized with Otsu's threshold.
- `panel_<name>.png` — one strip of `[ A | C | overlay | diff ]` for quick
  flipping through the whole set.

It also prints per-image and mean IoU / extra% / missing% and writes them
to `comparisons/stats.csv` — the same numbers you'd use to rank
checkpoints, now attached to the visuals.

## Step 7 — Threshold matchers (pass/fail, green vs red)

Where `overlay_compare.py` shows you *what* differs, the two matchers judge
*whether it's close enough*, against an adjustable threshold (default 90%;
both accept `--threshold 0.9` or `--threshold 90`). Both take two
directories, pair files by stem (ComfyUI's `_00001_` suffix is ignored),
resample everything to the generated image's size — 465 GDS art is upscaled
to 512 with nearest-neighbour so rectangle edges stay crisp — and write
overlays plus a stats CSV into a new output directory.

### GDS cells vs generated SEM (A/val → C/val)

```bash
python3 eval/match_gds_sem.py --root /path/to/gds_2_sem --split val \
    --out matches/gds_vs_gen
# or explicitly, with a looser bar and per-cell score labels:
python3 eval/match_gds_sem.py --gds-dir gds_2_sem/A/val --sem-dir gds_2_sem/C/val \
    --out matches/gds_vs_gen --threshold 85 --annotate
```

Each solid GDS rectangle is isolated as a *cell* (8-connected region) and
drawn translucently over the SEM image it should correspond to:

- **green** — the SEM has material across ≥ threshold of that rectangle
- **red** — below threshold: the cell is missing, shifted, or deformed

`--metric coverage` (default) asks "how much of this rectangle is present in
the SEM"; `--metric iou` also penalises SEM material bleeding outside the
rectangle. `--tolerance` (default 1px) dilates the SEM material before
scoring so sub-pixel edge placement and resampling softness don't cost you.
`--min-area` drops specks. The caption bar reports matched/total cells and
turns red when the image as a whole falls under threshold; per-image and
total stats land in `cell_match_stats.csv`.

Note that rectangles which touch in the layout form one cell, so a red
verdict on a large merged cell means part of that group is wrong.

### Original SEM vs generated SEM (B/val → C/val)

```bash
python3 eval/match_sem_sem.py --root /path/to/gds_2_sem --split val \
    --out matches/real_vs_gen
python3 eval/match_sem_sem.py --root /path/to/gds_2_sem --split val --metric ssim
```

The two SEMs are blended as the backdrop and highlighted by agreement —
deliberately tolerant rather than pixel-exact:

- `--metric structure` (default) — both images binarised (Otsu) and compared
  with `--tolerance` px of slack (default 2), so a bar landing a pixel or
  two off still counts. **Green** = material in both, **red** = material in
  only one (added or dropped by the generator), background left untinted.
  Score = tolerant IoU.
- `--metric ssim` — per-pixel local SSIM (11px window, light pre-blur to
  damp sensor noise) for texture as well as placement. Green ≥ threshold,
  red below. Scored only where either image has material, since flat noisy
  background scores badly under SSIM and would otherwise swamp the map; add
  `--include-background` to override.

The caption bar shows PASS/FAIL against the threshold plus extra/missing
percentages, and `sem_match_stats.csv` collects the numbers.

Both scripts share `eval/matchlib.py`. OpenCV is used when present (fast
connected components, morphology, blur) and every function falls back to
pure numpy when it isn't — so they run either on the host or inside the
training container (`gds2sem-train:v1` already has OpenCV), with no new
installs on the offline machine.

## Step 8 — Hardware-trojan screening (prototype)

The `trojan/` subsystem turns the pipeline into a trojan detector: given a
manufactured chip's SEM (C) plus the golden model you hold (GDS layout A and
original SEM B), it flags regions of C that differ from golden, classifies
each into one of ten trojan patterns (A–J), and returns a JSON verdict with
annotated bounding-box images. It's driven from LibreChat (Opus 5) via an
MCP server and ships as a third offline container (`gds2sem-trojan:v1`,
built by the same `build_and_export_images.sh`).

Quick command-line loop:

```bash
# make a labelled test set from clean generated C images
python3 trojan/scripts/inject_trojans.py --gds-dir gds_2_sem/A/val \
    --sem-dir gds_2_sem/C/val --out-dir testset --round-robin --rate 0.6
# screen a directory containing A/ B/ C/ (C = suspect) -> D/
python3 trojan/scripts/screen.py detect --root INPUT_DIR --out D --report
# score against ground truth
python3 trojan/scripts/eval_detection.py --truth testset/ground_truth.json \
    --results D/results.json
```

The default detector is a golden-model comparison (deterministic, no
training); a YOLO/RF-DETR backend can be trained on injected sets for
single-image detection. Full details — the pattern taxonomy, detector
internals, evaluation numbers, the MCP tools, and the LibreChat setup — are
in **[trojan/README.md](trojan/README.md)**.

## Troubleshooting notes

- `CUDA error: no kernel image` or driver mismatch at container start →
  something pulled cu130/cu128 wheels; both Dockerfiles assert
  `+cu126` at build time, so rebuild and check that assert step.
- Any error mentioning huggingface.co or `HF_HUB_OFFLINE` → a weight file
  wasn't found locally and ai-toolkit fell through to its hub path. Run
  `offline_prep/verify_models.sh` and check the preflight output at the top
  of the training log — it prints exactly which transformer/VAE/text-encoder
  paths were resolved.
- Preflight rejects the text encoder → you pointed it at the single-file
  ComfyUI `qwen_3_4b.safetensors`. Training needs the full
  transformers-format Qwen3-4B directory (config.json + tokenizer files +
  weight shards) in `models/flux2-klein/qwen3-4b/`.
- Generated images ignore the GDS layout → confirm the workflow's
  `ReferenceLatent` path is intact and the caption includes the trigger
  word; also check LoRA strength (1.0) and that you're on the base model.
- Rectangles blur/merge on distilled inference but not base → distilled +
  edit LoRA interaction; use base for production, or try distilled with
  8 steps.
