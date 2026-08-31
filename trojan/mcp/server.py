#!/usr/bin/env python3
"""
MCP server for the hardware-trojan detection prototype.

Exposes the detector to an MCP client (LibreChat with an Opus-backed agent)
as a small set of tools. The heavy lifting lives in trojan/scripts; this is
a thin, well-typed wrapper so the agent can:

  * list the trojan taxonomy (A-J)
  * run detection on an input directory that contains A/ B/ C subdirs
  * pull individual annotated result images back for display in the chat UI
  * optionally build a labelled test set by injecting trojans

Transport: stdio by default (what LibreChat launches), or streamable-http
when MCP_HTTP=1 (bind host/port with MCP_HOST / MCP_PORT) so it can run as a
long-lived service, e.g. inside the detection container on its own GPU.

Everything runs locally; no network calls.
"""
import base64
import json
import os
import subprocess
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
import patterns as P  # noqa: E402

# Root under which relative input/output paths are resolved and — crucially —
# the only tree the tools are allowed to read/write. Set to the transfer
# share / data mount on the offline host.
DATA_ROOT = Path(os.environ.get("TROJAN_DATA_ROOT", "/data")).resolve()
# Where D/ output directories are written when the caller doesn't name one.
OUT_ROOT = Path(os.environ.get("TROJAN_OUT_ROOT", str(DATA_ROOT / "runs"))).resolve()
PYTHON = os.environ.get("PYTHON", sys.executable)

mcp = FastMCP("trojan-detector")


def _safe(path: str, must_exist=True) -> Path:
    """Resolve a caller path inside DATA_ROOT; reject traversal outside it."""
    p = (DATA_ROOT / path).resolve() if not os.path.isabs(path) else Path(path).resolve()
    if DATA_ROOT not in p.parents and p != DATA_ROOT:
        raise ValueError(f"path {p} is outside the allowed data root {DATA_ROOT}")
    if must_exist and not p.exists():
        raise FileNotFoundError(f"path does not exist: {p}")
    return p


def _run(argv: list[str]) -> str:
    r = subprocess.run([PYTHON, *argv], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{argv[0]} failed:\n{r.stderr[-2000:]}")
    return r.stdout


@mcp.tool()
def list_trojan_patterns() -> str:
    """List the ten hardware-trojan patterns (A-J) the detector knows,
    with each one's human name and change-class (addition/bridge/
    modification/deletion). Returns JSON."""
    return json.dumps(P.catalog(), indent=2)


@mcp.tool()
def detect_trojans(input_dir: str, output_subdir: str = "",
                   backend: str = "golden", weights: str = "",
                   min_confidence: float = 0.3) -> str:
    """Run trojan detection on an input directory.

    input_dir must contain subdirectories:
      A/  GDS layouts (golden intent)
      B/  original 'known-good' SEM images (golden capture; optional but
          needed to catch dopant-class / intensity-only trojans)
      C/  the newly-captured suspect SEM images to screen
    Images are matched across A/B/C by filename.

    Writes a D-style output directory (results.json + annotated/ images) and
    returns a JSON summary: per-image status ('no_trojan_detected' or the
    list of detected patterns with boxes), overall counts, and the output
    path on the host. backend='golden' (default) needs no model; backend=
    'yolo' needs a trained model path in `weights`.

    Use show_detection() afterwards to display any flagged image in the chat.
    """
    in_path = _safe(input_dir)
    if not (in_path / "C").is_dir():
        raise ValueError(f"{in_path} has no C/ subdirectory (suspect SEM images)")
    out = (_safe(output_subdir, must_exist=False) if output_subdir
           else OUT_ROOT / (in_path.name + "_D"))
    out.mkdir(parents=True, exist_ok=True)

    argv = [str(SCRIPTS / "detect_trojans.py"), "--root", str(in_path),
            "--out", str(out), "--backend", backend,
            "--min-conf", str(min_confidence)]
    if backend == "yolo" and weights:
        argv += ["--weights", str(_safe(weights))]
    _run(argv)

    report = json.loads((out / "results.json").read_text())
    s = report["summary"]
    flagged = {n: v["detections"] for n, v in report["images"].items()
               if v["status"] == "trojan_detected"}
    return json.dumps({
        "output_dir": str(out),
        "results_json": str(out / "results.json"),
        "annotated_dir": str(out / "annotated"),
        "summary": s,
        "flagged_images": flagged,
    }, indent=2)


@mcp.tool()
def show_detection(image_path: str) -> ImageContent:
    """Return one annotated result image (PNG) for inline display in the
    chat. Pass either the annotated_dir path from detect_trojans plus the
    filename, or a full path to an image under the data root."""
    p = _safe(image_path)
    data = base64.b64encode(p.read_bytes()).decode()
    return ImageContent(type="image", data=data, mimeType="image/png")


@mcp.tool()
def inject_trojans(gds_dir: str, sem_dir: str, output_dir: str,
                   rate: float = 0.6, max_per_image: int = 2,
                   round_robin: bool = False, seed: int = 0) -> str:
    """Build a LABELLED test set by injecting synthetic trojans (A-J) into
    clean generated SEM images. gds_dir places trojans plausibly; sem_dir is
    the clean C images to tamper. Writes tampered images + ground_truth.json
    to output_dir. For evaluation/demo use, not production screening."""
    argv = [str(SCRIPTS / "inject_trojans.py"),
            "--gds-dir", str(_safe(gds_dir)), "--sem-dir", str(_safe(sem_dir)),
            "--out-dir", str(_safe(output_dir, must_exist=False)),
            "--rate", str(rate), "--max-per-image", str(max_per_image),
            "--seed", str(seed)]
    if round_robin:
        argv.append("--round-robin")
    log = _run(argv)
    gt = _safe(output_dir) / "ground_truth.json"
    return json.dumps({"output_dir": str(_safe(output_dir)),
                       "ground_truth": str(gt),
                       "summary": json.loads(gt.read_text())["summary"],
                       "log": log.strip()}, indent=2)


if __name__ == "__main__":
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if os.environ.get("MCP_HTTP") == "1":
        mcp.settings.host = os.environ.get("MCP_HOST", "0.0.0.0")
        mcp.settings.port = int(os.environ.get("MCP_PORT", "8130"))
        mcp.run(transport="streamable-http")
    else:
        mcp.run()  # stdio
