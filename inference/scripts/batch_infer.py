#!/usr/bin/env python3
"""
Headless batch GDS -> SEM conversion through a running ComfyUI server.

Builds the same graph as workflows/gds2sem_klein4b_*.json in ComfyUI
API-prompt form and submits one job per input image.

Prereqs:
  * ComfyUI running (inference/run_comfyui.sh), reachable at --server
  * models in place (klein 4B fp8 base and/or distilled, qwen_3_4b TE,
    flux2-vae, your LoRA in models/loras)

Usage:
  python batch_infer.py --input-dir ./gds_images --server http://localhost:8188 \
      --lora gds2sem_klein4b_v1.safetensors --variant base
Outputs land in ComfyUI's output dir (comfy_data/output on the host),
prefixed gds2sem/<input filename>.
"""
import argparse
import json
import time
import urllib.request
import uuid
from pathlib import Path

PROMPT_TEXT = ("g2s3m convert this GDS standard cell layout into a scanning "
               "electron microscope image, preserving the exact position and "
               "size of every rectangle, grayscale SEM texture with realistic "
               "noise, edge roughness and slight blur")

VARIANTS = {
    "base":      {"unet": "flux-2-klein-base-4b-fp8.safetensors", "steps": 20, "cfg": 4},
    "distilled": {"unet": "flux-2-klein-4b-fp8.safetensors",      "steps": 4,  "cfg": 1},
}


def api(server, path, data=None, content_type="application/json"):
    req = urllib.request.Request(server + path)
    if data is not None:
        if content_type == "application/json":
            data = json.dumps(data).encode()
        req.add_header("Content-Type", content_type)
        req.data = data
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read() or b"{}")


def upload_image(server, path: Path) -> str:
    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{path.name}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(server + "/upload/image", data=body)
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())["name"]


def build_prompt(image_name, v, lora, lora_strength, seed, size, prompt_text, out_prefix):
    g = {
        "1":  {"class_type": "UNETLoader",
               "inputs": {"unet_name": v["unet"], "weight_dtype": "default"}},
        "2":  {"class_type": "LoraLoaderModelOnly",
               "inputs": {"model": ["1", 0], "lora_name": lora,
                          "strength_model": lora_strength}},
        "3":  {"class_type": "CLIPLoader",
               "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "flux2",
                          "device": "default"}},
        "4":  {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}},
        "5":  {"class_type": "CLIPTextEncode",
               "inputs": {"clip": ["3", 0], "text": prompt_text}},
        "6":  {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": ""}},
        "7":  {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "8":  {"class_type": "ImageScale",
               "inputs": {"image": ["7", 0], "upscale_method": "nearest-exact",
                          "width": size, "height": size, "crop": "disabled"}},
        "9":  {"class_type": "VAEEncode", "inputs": {"pixels": ["8", 0], "vae": ["4", 0]}},
        "10": {"class_type": "ReferenceLatent",
               "inputs": {"conditioning": ["5", 0], "latent": ["9", 0]}},
        "11": {"class_type": "ReferenceLatent",
               "inputs": {"conditioning": ["6", 0], "latent": ["9", 0]}},
        "12": {"class_type": "CFGGuider",
               "inputs": {"model": ["2", 0], "positive": ["10", 0],
                          "negative": ["11", 0], "cfg": v["cfg"]}},
        "13": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "14": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "15": {"class_type": "Flux2Scheduler",
               "inputs": {"steps": v["steps"], "width": size, "height": size}},
        "16": {"class_type": "EmptyFlux2LatentImage",
               "inputs": {"width": size, "height": size, "batch_size": 1}},
        "17": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["13", 0], "guider": ["12", 0],
                          "sampler": ["14", 0], "sigmas": ["15", 0],
                          "latent_image": ["16", 0]}},
        "18": {"class_type": "VAEDecode", "inputs": {"samples": ["17", 0], "vae": ["4", 0]}},
        "19": {"class_type": "SaveImage",
               "inputs": {"images": ["18", 0], "filename_prefix": out_prefix}},
    }
    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True, type=Path)
    ap.add_argument("--server", default="http://localhost:8188")
    ap.add_argument("--variant", choices=list(VARIANTS), default="base")
    ap.add_argument("--lora", default="gds2sem_klein4b_v1.safetensors")
    ap.add_argument("--lora-strength", type=float, default=1.0)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt", default=PROMPT_TEXT)
    args = ap.parse_args()

    v = VARIANTS[args.variant]
    images = sorted(p for p in args.input_dir.iterdir()
                    if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif"})
    if not images:
        raise SystemExit(f"no images in {args.input_dir}")
    print(f"{len(images)} images -> {args.server} ({args.variant}, "
          f"{v['steps']} steps, cfg {v['cfg']})")

    client_id = uuid.uuid4().hex
    pending = {}
    for p in images:
        name = upload_image(args.server, p)
        g = build_prompt(name, v, args.lora, args.lora_strength, args.seed,
                         args.size, args.prompt, f"gds2sem/{p.stem}")
        r = api(args.server, "/prompt", {"prompt": g, "client_id": client_id})
        pending[r["prompt_id"]] = p.name
        print(f"queued {p.name} -> {r['prompt_id']}")

    print("waiting for queue to drain...")
    while pending:
        time.sleep(2.0)
        hist_done = []
        for pid in list(pending):
            h = api(args.server, f"/history/{pid}")
            if pid in h:
                status = h[pid].get("status", {})
                ok = status.get("status_str", "success")
                print(f"done  {pending[pid]}  ({ok})")
                hist_done.append(pid)
        for pid in hist_done:
            pending.pop(pid)
    print("all jobs finished; results are in ComfyUI's output/gds2sem/")


if __name__ == "__main__":
    main()
