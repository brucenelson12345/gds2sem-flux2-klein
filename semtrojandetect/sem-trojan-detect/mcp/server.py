#!/usr/bin/env python3
"""
MCP server for sem-trojan-detect.

Exposes the detector to an MCP client (LibreChat with an Opus-backed agent,
or scripts/screen.py's `remote` mode) as a small set of typed tools:

  list_trojan_patterns   the A-J taxonomy
  detect_trojans         screen an input dir with A/ B/ C -> D output
  show_detection         return one annotated image for inline display
  inject_trojans         build a labelled test set (demo/eval only)
  generate_sem           render SEM from GDS via the gds2sem service
  match_sems             B vs C cell-level difference report
  summarize_run          Claude-written triage summary of a run

Transport: stdio by default (what LibreChat launches), or streamable-http
when MCP_HTTP=1 (MCP_HOST / MCP_PORT) to run as a long-lived service on its
own GPU.

Everything is sandboxed to TROJAN_DATA_ROOT; the only outbound call this
process can make is to the gds2sem ComfyUI service, and only from
generate_sem.
"""
import base64
import json
import os
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trojanlib import (catalog, evaluate, inject_directory,  # noqa: E402
                       screen_directory, write_report)
from trojanlib.detect import DetectParams  # noqa: E402
from trojanlib.matcher import (MatchParams, match_directories,  # noqa: E402
                               write_match_report)
from trojanlib import gds2sem_client as g2s  # noqa: E402
from trojanlib import llm_client as llm  # noqa: E402

DATA_ROOT = Path(os.environ.get("TROJAN_DATA_ROOT", "/data")).resolve()
OUT_ROOT = Path(os.environ.get("TROJAN_OUT_ROOT", str(DATA_ROOT / "runs"))).resolve()
GDS2SEM_SERVER = os.environ.get("GDS2SEM_SERVER", g2s.DEFAULT_SERVER)

mcp = FastMCP("trojan-detector")


def _safe(path: str, must_exist=True) -> Path:
    """Resolve a caller path inside DATA_ROOT; reject traversal outside it."""
    p = (Path(path) if os.path.isabs(path) else DATA_ROOT / path).resolve()
    if DATA_ROOT not in p.parents and p != DATA_ROOT:
        raise ValueError(f"path {p} is outside the allowed data root {DATA_ROOT}")
    if must_exist and not p.exists():
        raise FileNotFoundError(f"path does not exist: {p}")
    return p


@mcp.tool()
def list_trojan_patterns() -> str:
    """List the ten hardware-trojan patterns (A-J) the detector knows, with
    each one's human name and change-class (addition / bridge /
    modification / deletion). Returns JSON."""
    return json.dumps(catalog(), indent=2)


@mcp.tool()
def detect_trojans(input_dir: str, output_subdir: str = "",
                   backend: str = "golden", weights: str = "",
                   min_confidence: float = 0.3, make_report: bool = True) -> str:
    """Run trojan detection on an input directory.

    input_dir must contain subdirectories:
      A/  GDS layouts (golden intent)
      B/  original known-good SEM images (optional, but required to catch
          dopant-class / intensity-only trojans)
      C/  the newly-captured suspect SEM images to screen
    Images are matched across A/B/C by filename.

    Writes a D-style output directory (results.json, annotated/ images, and
    a self-contained report.html) and returns a JSON summary: per-image
    status ('no_trojan_detected' or the detected patterns with boxes),
    overall counts, and the output paths on the host.

    Use show_detection() afterwards to display a flagged image in the chat.
    """
    in_path = _safe(input_dir)
    if not (in_path / "C").is_dir():
        raise ValueError(f"{in_path} has no C/ subdirectory (suspect SEM images)")
    out = (_safe(output_subdir, must_exist=False) if output_subdir
           else OUT_ROOT / (in_path.name + "_D"))

    report = screen_directory(
        in_path / "A", in_path / "B", in_path / "C", out,
        backend=backend, weights=(_safe(weights) if weights else None),
        params=DetectParams(min_conf=min_confidence), quiet=True)

    if make_report:
        write_report(out)
    s = report["summary"]
    flagged = {n: v["detections"] for n, v in report["images"].items()
               if v["status"] == "trojan_detected"}
    return json.dumps({
        "output_dir": str(out),
        "results_json": str(out / "results.json"),
        "annotated_dir": str(out / "annotated"),
        "report_html": str(out / "report.html") if make_report else None,
        "summary": s,
        "flagged_images": flagged,
    }, indent=2)


@mcp.tool()
def show_detection(image_path: str) -> ImageContent:
    """Return one annotated result image (PNG) for inline display in the
    chat. Pass the annotated_dir path from detect_trojans plus the
    filename, or any image path under the data root."""
    p = _safe(image_path)
    return ImageContent(type="image", mimeType="image/png",
                        data=base64.b64encode(p.read_bytes()).decode())


@mcp.tool()
def inject_trojans(gds_dir: str, sem_dir: str, output_dir: str,
                   rate: float = 0.6, max_per_image: int = 2,
                   round_robin: bool = False, seed: int = 0) -> str:
    """Build a LABELLED test set by injecting synthetic trojans (A-J) into
    clean SEM images. gds_dir places them plausibly; sem_dir holds the clean
    images to tamper. Writes tampered images + ground_truth.json to
    output_dir. For evaluation and demos only — never on real screening
    data, since it modifies images."""
    meta = inject_directory(_safe(gds_dir), _safe(sem_dir),
                            _safe(output_dir, must_exist=False),
                            rate, max_per_image, round_robin, None, seed,
                            quiet=True)
    return json.dumps({"output_dir": str(_safe(output_dir)),
                       "ground_truth": str(_safe(output_dir) / "ground_truth.json"),
                       "summary": meta["summary"]}, indent=2)


@mcp.tool()
def generate_sem(gds_dir: str, output_dir: str, variant: str = "base",
                 lora: str = "gds2sem_klein4b_v1.safetensors") -> str:
    """Render SEM images from GDS layouts by calling the gds2sem generator
    service (a separate tool; its ComfyUI container must be running).

    Use this to build a suspect/test set from layouts, or to synthesise a
    golden SEM baseline for a region where the layout exists but no
    known-good capture does. Real screening does not need it — there, C
    comes from an actual microscope."""
    try:
        written = g2s.generate_sem(_safe(gds_dir),
                                   _safe(output_dir, must_exist=False),
                                   server=GDS2SEM_SERVER, variant=variant,
                                   lora=lora, quiet=True)
    except g2s.Gds2SemUnavailable as e:
        return json.dumps({"error": str(e), "server": GDS2SEM_SERVER})
    return json.dumps({"output_dir": str(_safe(output_dir)),
                       "generated": len(written),
                       "server": GDS2SEM_SERVER}, indent=2)


@mcp.tool()
def match_sems(input_dir: str, output_subdir: str = "",
               match_iou: float = 0.25, tolerance: int = 2) -> str:
    """Compare the golden SEM images (B/) against the suspect ones (C/)
    cell by cell, and write a visual match report.

    Unlike detect_trojans, this makes no attempt to classify anything — it
    just answers "which cells changed": cells present in B but missing from
    C, and cells present in C but absent from B. Useful as a fast
    first-pass sanity check, and as the evidence view an analyst reads
    alongside a detection run.

    Writes match_report.html (every B and C image plus a B-over-C overlay,
    green for missing, red for gained), match_results.json and
    overlays/*.png. Returns the summary including the cell accuracy score.
    """
    in_path = _safe(input_dir)
    for sub in ("B", "C"):
        if not (in_path / sub).is_dir():
            raise ValueError(f"{in_path} has no {sub}/ subdirectory")
    out = (_safe(output_subdir, must_exist=False) if output_subdir
           else OUT_ROOT / (in_path.name + "_M"))
    report = match_directories(in_path / "B", in_path / "C", out,
                               MatchParams(tolerance=tolerance,
                                           match_iou=match_iou),
                               quiet=True)
    path = write_match_report(out, report)
    return json.dumps({"output_dir": str(out),
                       "report_html": str(path),
                       "results_json": str(out / "match_results.json"),
                       "summary": report["summary"]}, indent=2)


@mcp.tool()
def summarize_run(results_json: str, model: str = "") -> str:
    """Write an analyst triage summary of a finished screening run.

    Pass the results_json path returned by detect_trojans. The summary is
    produced by a Claude model on the configured Open WebUI instance
    (OPENWEBUI_URL / OPENWEBUI_API_KEY in this container's environment) and
    is also saved as summary.md beside the results file.

    A LibreChat agent can usually just read the detect_trojans output and
    summarise it itself; this tool is for clients that cannot — e.g. the
    command-line `screen.py remote` mode."""
    path = _safe(results_json)
    report = json.loads(path.read_text())
    try:
        text = llm.summarize_run(report, model=model or None)
    except llm.LLMError as e:
        return json.dumps({"error": str(e)})
    (path.parent / "summary.md").write_text(text + "\n")
    return text


if __name__ == "__main__":
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if os.environ.get("MCP_HTTP") == "1":
        mcp.settings.host = os.environ.get("MCP_HOST", "0.0.0.0")
        mcp.settings.port = int(os.environ.get("MCP_PORT", "8130"))
        mcp.run(transport="streamable-http")
    else:
        mcp.run()
