#!/usr/bin/env python3
"""
sem-trojan-detect — command-line screening (no LibreChat required).

Subcommands
  detect    Screen a directory containing A/ B/ C/ (C = suspect SEMs).
            Writes the D output (results.json + annotated/) and, with
            --report, a self-contained report.html openable in any browser.

  demo      End-to-end dry run: inject trojans into clean SEMs to build a
            suspect set, screen it, score against the injection ground
            truth, write the report. If C images are missing it calls the
            gds2sem generator to render them from the GDS layouts.

  eval      Score an existing results.json against a ground_truth.json.

  inject    Build a labelled test set only (no screening).

  generate  Render SEM images from GDS layouts by calling the gds2sem
            ComfyUI service (separate repo, must be running).

  llm       Open WebUI access for the Claude models: `login` saves your API
            token, `test` checks connectivity, `models` lists what the token
            can use, `summarize` writes an analyst narrative for a finished
            run. Add --summarize to detect/demo to do it inline.

  remote    Drive a running MCP service (run_detector_mcp.sh) from any
            machine that can reach it — the same tools LibreChat calls.

Examples
  python scripts/screen.py detect --root /data/incoming/lot42 \
      --out /data/runs/lot42_D --report
  python scripts/screen.py demo --root /data/gds_2_sem --out /data/runs/demo
  python scripts/screen.py generate --gds-dir /data/gds_2_sem/A/val \
      --out-dir /data/gds_2_sem/C/val --server http://localhost:8188
  python scripts/screen.py llm login --url http://webui:3000 --api-key sk-...
  python scripts/screen.py detect --root /data/lot42 --out /data/runs/D \
      --report --summarize
  python scripts/screen.py remote --server http://HOST:8130/mcp detect \
      --input-dir incoming/lot42
"""
import argparse
import base64
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trojanlib import (evaluate, inject_directory, screen_directory,  # noqa: E402
                       write_report)
from trojanlib.detect import DetectParams  # noqa: E402
from trojanlib import gds2sem_client as g2s  # noqa: E402
from trojanlib import llm_client as llm  # noqa: E402

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _params(a) -> DetectParams:
    return DetectParams(min_conf=a.min_conf) if not hasattr(a, "tolerance") \
        else DetectParams(a.tolerance, a.intensity_delta, a.merge,
                          a.min_area, a.min_conf)


def _summarize(report, out_dir, a, eval_text=""):
    """Ask Claude (via Open WebUI) for an analyst summary; never fatal."""
    try:
        text = llm.summarize_run(report, eval_text, model=a.model,
                                 api_key=a.api_key, url=a.url)
    except llm.LLMError as e:
        print(f"\n[summary skipped] {e}")
        return ""
    print("\n=== analyst summary (Claude via Open WebUI) ===\n" + text)
    (out_dir / "summary.md").write_text(text + "\n")
    return text


def cmd_detect(a):
    root = a.root
    a_dir, b_dir, c_dir = (root / "A", root / "B", root / "C") if root \
        else (a.a_dir, a.b_dir, a.c_dir)
    if not c_dir or not Path(c_dir).is_dir():
        sys.exit("need --root with a C/ subdir, or --c-dir")
    report = screen_directory(a_dir, b_dir, c_dir, a.out, a.backend,
                              a.weights, _params(a))
    text = ""
    if a.truth:
        text, _ = evaluate(a.truth, a.out / "results.json", a.iou)
        print("\n" + text)
    summary = _summarize(report, a.out, a, text) if a.summarize else ""
    if a.report:
        print(f"\nreport   -> {write_report(a.out, text, summary)}")


def cmd_generate(a):
    try:
        g2s.generate_sem(a.gds_dir, a.out_dir, server=a.server,
                         variant=a.variant, lora=a.lora, size=a.size,
                         seed=a.seed)
    except g2s.Gds2SemUnavailable as e:
        sys.exit(f"ERROR: {e}")


def _has_images(d: Path) -> bool:
    return d.is_dir() and any(p.suffix.lower() in IMG_EXTS for p in d.iterdir())


def cmd_demo(a):
    gds, ref, sem = (a.root / "A" / a.split, a.root / "B" / a.split,
                     a.root / "C" / a.split)
    if not _has_images(gds):
        sys.exit(f"demo needs GDS layouts in {gds}")

    if not _has_images(sem):
        print(f"no generated SEMs in {sem} — calling the gds2sem service "
              f"at {a.server} to render them")
        try:
            g2s.generate_sem(gds, sem, server=a.server, variant=a.variant,
                             lora=a.lora)
        except g2s.Gds2SemUnavailable as e:
            sys.exit(f"{e}\n(or supply C/ images yourself and re-run)")

    a.out.mkdir(parents=True, exist_ok=True)
    inp = a.out / "input"
    (inp / "A").mkdir(parents=True, exist_ok=True)
    for p in gds.iterdir():
        if p.suffix.lower() in IMG_EXTS:
            shutil.copy(p, inp / "A" / p.name)
    if _has_images(ref):
        (inp / "B").mkdir(exist_ok=True)
        for p in ref.iterdir():
            if p.suffix.lower() in IMG_EXTS:
                shutil.copy(p, inp / "B" / p.name)

    print("\n== injecting trojans into clean SEMs ==")
    inject_directory(gds, sem, inp / "C", rate=a.rate, max_per_image=3,
                     round_robin=True, seed=a.seed)
    print("\n== screening ==")
    D = a.out / "D"
    report = screen_directory(inp / "A", inp / "B", inp / "C", D)
    print("\n== scoring vs ground truth ==")
    text, _ = evaluate(inp / "C" / "ground_truth.json", D / "results.json")
    print(text)
    summary = _summarize(report, D, a, text) if a.summarize else ""
    print(f"\nreport   -> {write_report(D, text, summary)}")


def cmd_eval(a):
    text, _ = evaluate(a.truth, a.results, a.iou)
    print(text)


def cmd_inject(a):
    inject_directory(a.gds_dir, a.sem_dir, a.out_dir, a.rate, a.max_per_image,
                     a.round_robin, list(a.patterns), a.seed)


def cmd_remote(a):
    try:
        import asyncio
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client
    except ImportError:
        sys.exit("remote mode needs `pip install mcp` on this machine")

    async def go():
        async with streamablehttp_client(a.server) as (r, w, _):
            async with ClientSession(r, w) as s:
                await s.initialize()
                if a.action == "patterns":
                    res = await s.call_tool("list_trojan_patterns", {})
                elif a.action == "detect":
                    if not a.input_dir:
                        sys.exit("remote detect needs --input-dir")
                    args = {"input_dir": a.input_dir, "backend": a.backend,
                            "min_confidence": a.min_conf}
                    if a.output_subdir:
                        args["output_subdir"] = a.output_subdir
                    res = await s.call_tool("detect_trojans", args)
                elif a.action == "fetch":
                    if not a.image:
                        sys.exit("remote fetch needs --image")
                    res = await s.call_tool("show_detection", {"image_path": a.image})
                else:
                    sys.exit(f"unknown remote action {a.action}")
                for c in res.content:
                    if c.type == "text":
                        print(c.text)
                    elif c.type == "image":
                        out = Path(a.save or "detection.png")
                        out.write_bytes(base64.b64decode(c.data))
                        print(f"saved image -> {out}")
    asyncio.run(go())


def _add_llm_flags(p):
    """Open WebUI access, shared by the commands that can summarize."""
    p.add_argument("--summarize", action="store_true",
                   help="ask Claude (via Open WebUI) for an analyst summary")
    p.add_argument("--model", default=None,
                   help="model id on the Open WebUI instance "
                        "(default: auto-pick a Claude model)")
    p.add_argument("--api-key", default=None,
                   help="Open WebUI API key (else OPENWEBUI_API_KEY, else "
                        "the saved config)")
    p.add_argument("--url", default=None,
                   help="Open WebUI base URL, e.g. http://host:3000 "
                        "(else OPENWEBUI_URL, else saved config)")


def cmd_llm(a):
    if a.action == "login":
        if not a.api_key or not a.url:
            sys.exit("llm login needs --url and --api-key")
        path = llm.save_config(a.url, a.api_key, a.model or "")
        print(f"saved credentials -> {path} (key {llm.mask(a.api_key)})")
        ok, msg = llm.ping(a.api_key, a.url)
        print(msg if ok else f"WARNING: {msg}")
        return
    if a.action == "test":
        ok, msg = llm.ping(a.api_key, a.url)
        print(msg)
        sys.exit(0 if ok else 1)
    if a.action == "models":
        try:
            for m in llm.list_models(a.api_key, a.url):
                print(m)
        except llm.LLMError as e:
            sys.exit(f"ERROR: {e}")
        return
    if a.action == "summarize":
        if not a.results:
            sys.exit("llm summarize needs --results <results.json>")
        import json as _json
        report = _json.loads(Path(a.results).read_text())
        try:
            text = llm.summarize_run(report, model=a.model,
                                     api_key=a.api_key, url=a.url)
        except llm.LLMError as e:
            sys.exit(f"ERROR: {e}")
        print(text)
        out = Path(a.results).parent
        (out / "summary.md").write_text(text + "\n")
        print(f"\nsaved -> {out / 'summary.md'}")


def _add_detect_knobs(p):
    p.add_argument("--tolerance", type=int, default=2)
    p.add_argument("--intensity-delta", type=float, default=28)
    p.add_argument("--merge", type=int, default=2)
    p.add_argument("--min-area", type=int, default=24)
    p.add_argument("--min-conf", type=float, default=0.3)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("detect", help="screen a directory with A/ B/ C/")
    d.add_argument("--root", type=Path)
    d.add_argument("--a-dir", type=Path)
    d.add_argument("--b-dir", type=Path)
    d.add_argument("--c-dir", type=Path)
    d.add_argument("--out", required=True, type=Path)
    d.add_argument("--backend", choices=["golden", "yolo"], default="golden")
    d.add_argument("--weights", type=Path)
    d.add_argument("--truth", type=Path, help="optional ground truth to score against")
    d.add_argument("--iou", type=float, default=0.3)
    d.add_argument("--report", action="store_true")
    _add_detect_knobs(d)
    _add_llm_flags(d)
    d.set_defaults(fn=cmd_detect)

    m = sub.add_parser("demo", help="generate/inject -> detect -> eval -> report")
    m.add_argument("--root", required=True, type=Path,
                   help="gds_2_sem directory (A/B/C with --split subdirs)")
    m.add_argument("--split", default="val")
    m.add_argument("--out", required=True, type=Path)
    m.add_argument("--rate", type=float, default=0.75)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--server", default=g2s.DEFAULT_SERVER,
                   help="gds2sem ComfyUI service, used only if C/ is empty")
    m.add_argument("--variant", choices=["base", "distilled"], default="base")
    m.add_argument("--lora", default="gds2sem_klein4b_v1.safetensors")
    _add_llm_flags(m)
    m.set_defaults(fn=cmd_demo)

    g = sub.add_parser("generate", help="render SEM from GDS via gds2sem")
    g.add_argument("--gds-dir", required=True, type=Path)
    g.add_argument("--out-dir", required=True, type=Path)
    g.add_argument("--server", default=g2s.DEFAULT_SERVER)
    g.add_argument("--variant", choices=["base", "distilled"], default="base")
    g.add_argument("--lora", default="gds2sem_klein4b_v1.safetensors")
    g.add_argument("--size", type=int, default=512)
    g.add_argument("--seed", type=int, default=42)
    g.set_defaults(fn=cmd_generate)

    e = sub.add_parser("eval", help="score results.json vs ground_truth.json")
    e.add_argument("--truth", required=True, type=Path)
    e.add_argument("--results", required=True, type=Path)
    e.add_argument("--iou", type=float, default=0.3)
    e.set_defaults(fn=cmd_eval)

    j = sub.add_parser("inject", help="build a labelled test set")
    j.add_argument("--gds-dir", required=True, type=Path)
    j.add_argument("--sem-dir", required=True, type=Path)
    j.add_argument("--out-dir", required=True, type=Path)
    j.add_argument("--rate", type=float, default=0.6)
    j.add_argument("--max-per-image", type=int, default=2)
    j.add_argument("--round-robin", action="store_true")
    j.add_argument("--patterns", default="ABCDEFGHIJ")
    j.add_argument("--seed", type=int, default=0)
    j.set_defaults(fn=cmd_inject)

    l = sub.add_parser("llm", help="Open WebUI / Claude access")
    l.add_argument("action", choices=["login", "test", "models", "summarize"])
    l.add_argument("--url", default=None, help="e.g. http://host:3000")
    l.add_argument("--api-key", default=None)
    l.add_argument("--model", default=None)
    l.add_argument("--results", type=Path,
                   help="summarize: path to a results.json")
    l.set_defaults(fn=cmd_llm)

    r = sub.add_parser("remote", help="call a running MCP service")
    r.add_argument("--server", required=True, help="e.g. http://HOST:8130/mcp")
    r.add_argument("action", choices=["patterns", "detect", "fetch"])
    r.add_argument("--input-dir")
    r.add_argument("--output-subdir", default="")
    r.add_argument("--backend", choices=["golden", "yolo"], default="golden")
    r.add_argument("--min-conf", type=float, default=0.3)
    r.add_argument("--image")
    r.add_argument("--save")
    r.set_defaults(fn=cmd_remote)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
