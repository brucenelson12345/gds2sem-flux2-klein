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
| Transformer weights | `flux-2-klein-base-4b.safetensors` from `black-forest-labs/FLUX.2-klein-base-4B` (~7.8 GB) |
| Text encoder | `Qwen/Qwen3-4B` (repo id hardcoded in ai-toolkit; resolved from HF cache) |
| VAE | `ae.safetensors` from `ai-toolkit/flux2_vae` |
| Paired dataset | `datasets[].folder_path` = targets (SEM) + captions, `datasets[].control_path` = controls (GDS), matched by filename |
| Sample-time control | `--ctrl_img /path.png` appended to sample prompts |

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
│   └── download_models.sh           # online: fetch all model weights (~30 GB)
├── training/
│   ├── Dockerfile                   # ai-toolkit, torch cu126
│   ├── config/gds2sem_klein4b.yaml  # edit-LoRA config, wired to gds_2_sem/
│   ├── scripts/setup_dataset.py     # validate pairs, captions, patch config
│   └── run_training.sh              # docker run wrapper (offline)
├── inference/
│   ├── Dockerfile                   # ComfyUI, torch cu126
│   ├── run_comfyui.sh               # docker run wrapper (offline)
│   ├── workflows/
│   │   ├── gds2sem_klein4b_base_lora.json       # 20 steps, CFG 4
│   │   └── gds2sem_klein4b_distilled_lora.json  # 4 steps, CFG 1
│   └── scripts/batch_infer.py       # headless folder→folder conversion via API
└── eval/eval_structure.py           # IoU + SSIM checkpoint scoring
```

## Step 1 — Online machine: build images and download weights

```bash
cd gds2sem-flux2-klein
./offline_prep/build_and_export_images.sh ./transfer   # docker images (~15-20 GB)
./offline_prep/download_models.sh ./transfer           # weights (~30 GB)
```

`transfer/` then contains `images/` (two docker tars), `hf_cache/`
(training weights in HF-cache layout), and `comfy_models/`
(diffusion_models, text_encoders, vae, loras). Copy the whole folder to the
transfer share. Offline host prerequisites (from your internal apt mirror):
`docker`, `nvidia-container-toolkit`.

On the offline machine:

```bash
docker load < transfer/images/gds2sem-train_v1.tar.gz
docker load < transfer/images/gds2sem-comfy_v1.tar.gz
```

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
    HF_CACHE=$PWD/transfer/hf_cache GPU=0 ./training/run_training.sh
```

The dataset is mounted at `/workspace/gds_2_sem` inside the container
(read-write — `cache_latents_to_disk` writes a latent-cache folder inside
the dataset directories; delete those `_latent_cache` folders if you ever
change the images).

The container runs `python run.py /workspace/config.yaml` with
`HF_HUB_OFFLINE=1`, resolving all weights from the mounted cache.
Checkpoints and sample grids appear in
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

## Troubleshooting notes

- `CUDA error: no kernel image` or driver mismatch at container start →
  something pulled cu130/cu128 wheels; both Dockerfiles assert
  `+cu126` at build time, so rebuild and check that assert step.
- Training tries to reach huggingface.co offline → the HF cache mount is
  missing one of the three repos (klein base 4B, `ai-toolkit/flux2_vae`,
  `Qwen/Qwen3-4B`). `HF_HUB_OFFLINE=1` errors name the missing repo.
- Generated images ignore the GDS layout → confirm the workflow's
  `ReferenceLatent` path is intact and the caption includes the trigger
  word; also check LoRA strength (1.0) and that you're on the base model.
- Rectangles blur/merge on distilled inference but not base → distilled +
  edit LoRA interaction; use base for production, or try distilled with
  8 steps.
