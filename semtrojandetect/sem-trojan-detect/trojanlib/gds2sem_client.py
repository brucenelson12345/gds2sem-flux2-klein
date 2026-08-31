"""
Client for the **gds2sem** generator (separate repository).

This repo does not import gds2sem code. When it needs SEM images rendered
from GDS layouts, it calls the ComfyUI service that gds2sem stands up, over
that service's HTTP API — so the two tools stay independently deployable and
versioned. All you need running is gds2sem's inference container:

    # in the gds2sem repo, on the offline host
    COMFY_MODELS=... GPU=1 ./inference/run_comfyui.sh

Then point this client at it (default http://localhost:8188).

Where it's used
  * `screen.py generate` — render a directory of GDS layouts into SEM
    images (e.g. to build a suspect/test set, or to synthesise a golden SEM
    baseline for a region where you hold the layout but no known-good
    capture — which is what enables dopant-class detection there).
  * `screen.py demo` — auto-invoked when the demo needs generated SEMs and
    none are present.

Production screening does NOT need this: there, C comes off a real
microscope and A/B are your existing golden model.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

DEFAULT_SERVER = "http://localhost:8188"
DEFAULT_PROMPT = (
    "g2s3m convert this GDS standard cell layout into a scanning electron "
    "microscope image, preserving the exact position and size of every "
    "rectangle, grayscale SEM texture with realistic noise, edge roughness "
    "and slight blur")

# Must match what the gds2sem inference container has in its models folders.
VARIANTS = {
    "base":      {"unet": "flux-2-klein-base-4b-fp8.safetensors", "steps": 20, "cfg": 4},
    "distilled": {"unet": "flux-2-klein-4b-fp8.safetensors",      "steps": 4,  "cfg": 1},
}
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class Gds2SemUnavailable(RuntimeError):
    """The generator service could not be reached."""


def _api(server, path, data=None):
    req = urllib.request.Request(server.rstrip("/") + path)
    if data is not None:
        req.add_header("Content-Type", "application/json")
        req.data = json.dumps(data).encode()
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read() or b"{}")


def ping(server=DEFAULT_SERVER) -> bool:
    """True if the gds2sem ComfyUI service is reachable."""
    try:
        _api(server, "/system_stats")
        return True
    except Exception:
        return False


def _upload(server, path: Path) -> str:
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; '
            f'filename="{path.name}"\r\n'
            f"Content-Type: application/octet-stream\r\n\r\n").encode()
    body += path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(server.rstrip("/") + "/upload/image", data=body)
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["name"]


def _graph(image_name, v, lora, lora_strength, seed, size, prompt, prefix):
    """The gds2sem edit graph in ComfyUI API form (FLUX.2 Klein +
    ReferenceLatent conditioning on the GDS image, plus the trained LoRA)."""
    return {
        "1":  {"class_type": "UNETLoader",
               "inputs": {"unet_name": v["unet"], "weight_dtype": "default"}},
        "2":  {"class_type": "LoraLoaderModelOnly",
               "inputs": {"model": ["1", 0], "lora_name": lora,
                          "strength_model": lora_strength}},
        "3":  {"class_type": "CLIPLoader",
               "inputs": {"clip_name": "qwen_3_4b.safetensors",
                          "type": "flux2", "device": "default"}},
        "4":  {"class_type": "VAELoader",
               "inputs": {"vae_name": "flux2-vae.safetensors"}},
        "5":  {"class_type": "CLIPTextEncode",
               "inputs": {"clip": ["3", 0], "text": prompt}},
        "6":  {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": ""}},
        "7":  {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "8":  {"class_type": "ImageScale",
               "inputs": {"image": ["7", 0], "upscale_method": "nearest-exact",
                          "width": size, "height": size, "crop": "disabled"}},
        "9":  {"class_type": "VAEEncode",
               "inputs": {"pixels": ["8", 0], "vae": ["4", 0]}},
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
        "18": {"class_type": "VAEDecode",
               "inputs": {"samples": ["17", 0], "vae": ["4", 0]}},
        "19": {"class_type": "SaveImage",
               "inputs": {"images": ["18", 0], "filename_prefix": prefix}},
    }


def generate_sem(gds_dir, out_dir, server=DEFAULT_SERVER, variant="base",
                 lora="gds2sem_klein4b_v1.safetensors", lora_strength=1.0,
                 size=512, seed=42, prompt=DEFAULT_PROMPT, timeout=1800,
                 quiet=False):
    """Render every GDS layout in gds_dir into an SEM image in out_dir.

    Submits one job per image to the gds2sem ComfyUI service, waits for the
    queue to drain, then copies the rendered images out of the service's
    output directory (via its /view endpoint) into out_dir, named after the
    source layout so A/B/C stay filename-matched.
    """
    gds_dir, out_dir = Path(gds_dir), Path(out_dir)
    if not ping(server):
        raise Gds2SemUnavailable(
            f"gds2sem ComfyUI service not reachable at {server}. Start it from "
            "the gds2sem repo (inference/run_comfyui.sh), or pass --server.")
    v = VARIANTS[variant]
    images = sorted(p for p in gds_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        raise SystemExit(f"no GDS images in {gds_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    client_id = uuid.uuid4().hex
    pending = {}
    for p in images:
        name = _upload(server, p)
        g = _graph(name, v, lora, lora_strength, seed, size, prompt,
                   f"trojan_gen/{p.stem}")
        r = _api(server, "/prompt", {"prompt": g, "client_id": client_id})
        pending[r["prompt_id"]] = p.stem
        if not quiet:
            print(f"queued {p.name}")

    if not quiet:
        print(f"waiting for {len(pending)} job(s) on {server} ...")
    written, deadline = [], time.time() + timeout
    while pending and time.time() < deadline:
        time.sleep(2.0)
        for pid in list(pending):
            hist = _api(server, f"/history/{pid}")
            if pid not in hist:
                continue
            stem = pending.pop(pid)
            for node in hist[pid].get("outputs", {}).values():
                for im in node.get("images", []):
                    q = (f"/view?filename={im['filename']}"
                         f"&subfolder={im.get('subfolder', '')}"
                         f"&type={im.get('type', 'output')}")
                    with urllib.request.urlopen(server.rstrip("/") + q,
                                                timeout=60) as resp:
                        data = resp.read()
                    dst = out_dir / f"{stem}.png"
                    dst.write_bytes(data)
                    written.append(dst)
                    if not quiet:
                        print(f"  {dst.name}")
    if pending:
        raise TimeoutError(f"{len(pending)} job(s) still running after "
                           f"{timeout}s; check the ComfyUI service")
    if not quiet:
        print(f"generated {len(written)} SEM image(s) -> {out_dir}")
    return written
