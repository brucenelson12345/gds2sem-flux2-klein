#!/usr/bin/env python3
"""
Fully-local launcher for ai-toolkit FLUX.2 Klein training — no Hugging Face
hub, no HF cache. All weights are plain files in one directory.

Expected model directory (override with FLUX2_MODEL_DIR, default
/models/flux2-klein):

    flux2-klein/
      flux-2-klein-base-4b.safetensors   transformer (undistilled base)
      ae.safetensors                     FLUX.2 VAE, BFL single-file format
                                         (flux2-vae.safetensors also accepted)
      qwen3-4b/                          text encoder, full transformers
        config.json                      format directory: config, tokenizer
        tokenizer.json / tokenizer_config.json
        model*.safetensors (+ model.safetensors.index.json if sharded)

Why this wrapper exists: ai-toolkit resolves the transformer and VAE from
`model.name_or_path` (a local directory works out of the box), but the
Klein text-encoder location is a CLASS ATTRIBUTE hardcoded to the hub id
"Qwen/Qwen3-4B". This script patches that attribute (and the VAE fallback
path) to local paths before handing control to ai-toolkit's normal run.py.
Extensions are imported through the regular module system, so the patch
applies to the same class objects the trainer uses.

Usage (inside the training container):
    python run_offline.py /workspace/config.yaml
The config's model.name_or_path must point at the model directory
(e.g. "/models/flux2-klein").
"""
import json
import os
import runpy
import struct
import sys

AI_TOOLKIT = os.environ.get("AI_TOOLKIT_DIR", "/app/ai-toolkit")
MODEL_DIR = os.environ.get("FLUX2_MODEL_DIR", "/models/flux2-klein")
TE_DIR = os.environ.get("FLUX2_TE_DIR", os.path.join(MODEL_DIR, "qwen3-4b"))
VAE_FILE = os.environ.get("FLUX2_VAE_FILE", "")

TRANSFORMER = "flux-2-klein-base-4b.safetensors"


def die(msg: str):
    print(f"\nERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def safetensors_keys(path: str) -> set:
    """Read only the safetensors JSON header — cheap even for huge files."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    return set(header)


def find_vae() -> str:
    if VAE_FILE:
        if not os.path.isfile(VAE_FILE):
            die(f"FLUX2_VAE_FILE={VAE_FILE} does not exist")
        return VAE_FILE
    for name in ("ae.safetensors", "flux2-vae.safetensors"):
        p = os.path.join(MODEL_DIR, name)
        if os.path.isfile(p):
            return p
    die(f"no VAE found in {MODEL_DIR} (looked for ae.safetensors / "
        "flux2-vae.safetensors); set FLUX2_VAE_FILE to point at it")


def preflight() -> str:
    problems = []
    t = os.path.join(MODEL_DIR, TRANSFORMER)
    if not os.path.isfile(t):
        problems.append(f"missing transformer: {t}")

    vae = find_vae()
    keys = safetensors_keys(vae)
    if "decoder.up.0.block.0.conv1.bias" not in keys:
        problems.append(
            f"VAE {vae} is not in the BFL single-file format ai-toolkit "
            "expects (no 'decoder.up.0...' keys). Use the FLUX.2 "
            "ae.safetensors / flux2-vae.safetensors VAE, not a "
            "diffusers-folder or renamed non-FLUX.2 VAE.")

    if not os.path.isfile(os.path.join(TE_DIR, "config.json")):
        problems.append(
            f"text encoder dir {TE_DIR} has no config.json — training needs "
            "the FULL transformers-format Qwen3-4B directory (config.json, "
            "tokenizer files, model*.safetensors). The single-file ComfyUI "
            "qwen_3_4b.safetensors is NOT usable here.")
    else:
        cfg = json.load(open(os.path.join(TE_DIR, "config.json")))
        if "qwen3" not in str(cfg.get("model_type", "")).lower():
            problems.append(f"{TE_DIR}/config.json model_type="
                            f"{cfg.get('model_type')!r}, expected qwen3")
        if not any(f.startswith("tokenizer") for f in os.listdir(TE_DIR)):
            problems.append(f"no tokenizer files in {TE_DIR}")
        if not any(f.endswith(".safetensors") for f in os.listdir(TE_DIR)):
            problems.append(f"no model *.safetensors in {TE_DIR}")

    if problems:
        die("model preflight failed:\n  - " + "\n  - ".join(problems))
    print(f"preflight OK\n  transformer: {t}\n  vae:         {vae}"
          f"\n  text enc:    {TE_DIR}")
    return vae


def main():
    # Belt-and-braces: guarantee nothing silently reaches out to the hub.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("DISABLE_TELEMETRY", "YES")

    if not os.path.isdir(AI_TOOLKIT):
        die(f"ai-toolkit not found at {AI_TOOLKIT} (set AI_TOOLKIT_DIR)")
    vae = preflight()

    os.chdir(AI_TOOLKIT)
    sys.path.insert(0, AI_TOOLKIT)

    from extensions_built_in.diffusion_models.flux2.flux2_klein_model import (
        Flux2KleinModel, Flux2Klein4BModel, Flux2Klein9BModel)

    Flux2Klein4BModel.flux2_klein_te_path = TE_DIR
    Flux2Klein9BModel.flux2_klein_te_path = os.environ.get(
        "FLUX2_TE_DIR_9B", TE_DIR)
    # Fallback VAE location -> local file (used when the model dir itself
    # has no ae.safetensors and the config sets no vae_path).
    Flux2KleinModel.flux2_vae_path = vae

    print("launching ai-toolkit run.py ...")
    runpy.run_path(os.path.join(AI_TOOLKIT, "run.py"), run_name="__main__")


if __name__ == "__main__":
    main()
