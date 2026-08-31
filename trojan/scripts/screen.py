#!/usr/bin/env python3
"""
Command-line trojan screening — no LibreChat needed.

One entry point for the whole flow, runnable on the offline host directly
(or inside the gds2sem-trojan container via trojan/run_screen.sh):

  detect   Screen a directory containing A/ B/ C/ (C = suspect SEMs).
           Writes the D output (results.json + annotated/) and, with
           --report, a self-contained report.html you can open in any
           browser — summary counts, per-image verdicts, and the annotated
           images inline.

  demo     End-to-end dry run on your own data: injects trojans into clean
           generated SEMs to build a suspect set, screens it, scores the
           result against the injection ground truth, and writes the report.
           Proves the loop works before you point it at real captures.

  eval     Score an existing D/results.json against a ground_truth.json.

  remote   Drive a running MCP service (trojan/run_detector_mcp.sh) from
           any machine that can reach it — same tools LibreChat would call:
             remote --server http://HOST:8130/mcp patterns
             remote --server ... detect --input-dir incoming/lot42
             remote --server ... fetch --image PATH --save out.png
           Needs `pip install mcp` on the calling machine only.

Examples:
  python screen.py detect --root /data/incoming/lot42 --out /data/runs/lot42_D --report
  python screen.py demo   --root gds_2_sem --out demo_run
  python screen.py eval   --truth testset/ground_truth.json --results D/results.json
"""
import argparse
import base64
import html
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run_script(name: str, argv: list[str], capture=False) -> str:
    r = subprocess.run([sys.executable, str(HERE / name), *argv],
                       capture_output=capture, text=True)
    if r.returncode != 0:
        if capture:
            sys.stderr.write(r.stderr or "")
        sys.exit(f"{name} failed (exit {r.returncode})")
    return r.stdout if capture else ""


# --------------------------------------------------------------------------
# HTML report
# --------------------------------------------------------------------------
def _thumb(path: Path, width=560) -> str:
    """Inline <img> as base64 (downscaled thumbnail keeps the report small)."""
    try:
        from PIL import Image
        import io
        im = Image.open(path)
        if im.width > width:
            im = im.resize((width, int(im.height * width / im.width)))
        buf = io.BytesIO()
        im.save(buf, "PNG")
        data = buf.getvalue()
    except Exception:
        data = path.read_bytes()
    return ("<img src='data:image/png;base64,"
            + base64.b64encode(data).decode() + "' style='max-width:100%'>")


def write_report(out_dir: Path, eval_text: str = "") -> Path:
    report = json.loads((out_dir / "results.json").read_text())
    s = report["summary"]
    ann = out_dir / "annotated"
    cat = {c["pattern"]: c for c in report["catalog"]}

    rows = []
    for name, rec in sorted(report["images"].items(),
                            key=lambda kv: -kv[1]["count"]):
        flagged = rec["status"] == "trojan_detected"
        dets = "".join(
            f"<li><b>{d['pattern']}</b> — {html.escape(d['name'])} "
            f"<i>({html.escape(d['class'])})</i>, bbox {d['bbox']}, "
            f"conf {d['confidence']:.0%}</li>" for d in rec["detections"])
        img_html = ""
        if flagged and (ann / name).exists():
            img_html = _thumb(ann / name)
        badge = ("<span class='bad'>TROJAN DETECTED</span>" if flagged
                 else "<span class='ok'>no trojan detected</span>")
        rows.append(
            f"<div class='card'><h3>{html.escape(name)} {badge}</h3>"
            + (f"<ul>{dets}</ul>" if dets else "")
            + img_html + "</div>")

    per = "".join(
        f"<tr><td><b>{k}</b></td><td>{html.escape(cat[k]['name'])}</td>"
        f"<td>{html.escape(cat[k]['class'])}</td>"
        f"<td style='text-align:right'>{v}</td></tr>"
        for k, v in report["summary"]["per_pattern"].items())

    ev = (f"<h2>Evaluation vs ground truth</h2><pre>{html.escape(eval_text)}"
          "</pre>" if eval_text else "")

    page = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>Trojan screening report</title><style>
 body{{font-family:system-ui,sans-serif;margin:24px auto;max-width:900px;
      background:#111;color:#ddd}}
 h1,h2,h3{{color:#fff}} .ok{{color:#3fbf6f}} .bad{{color:#ff5252;font-weight:700}}
 table{{border-collapse:collapse}} td,th{{border:1px solid #444;padding:4px 10px}}
 .card{{border:1px solid #333;border-radius:8px;padding:12px 16px;margin:14px 0;
       background:#181818}}
 pre{{background:#181818;padding:10px;overflow-x:auto}}
</style></head><body>
<h1>Trojan screening report</h1>
<p>generated {html.escape(report['generated'])} · backend
 <b>{html.escape(report['backend'])}</b></p>
<p><b>{s['images']}</b> images screened —
 <span class='bad'>{s['flagged']} flagged</span>,
 <span class='ok'>{s['clean']} clean</span>,
 {s['detections']} detections total.</p>
<h2>Detections per pattern</h2>
<table><tr><th>key</th><th>name</th><th>class</th><th>count</th></tr>{per}</table>
{ev}
<h2>Per-image results</h2>
{''.join(rows)}
</body></html>"""
    p = out_dir / "report.html"
    p.write_text(page)
    return p


# --------------------------------------------------------------------------
# subcommands
# --------------------------------------------------------------------------
def cmd_detect(a):
    argv = ["--root", str(a.root), "--out", str(a.out),
            "--backend", a.backend, "--min-conf", str(a.min_conf)]
    if a.weights:
        argv += ["--weights", str(a.weights)]
    run_script("detect_trojans.py", argv)
    ev = ""
    if a.truth:
        ev = run_script("eval_detection.py",
                        ["--truth", str(a.truth),
                         "--results", str(a.out / "results.json")], capture=True)
        print(ev)
    if a.report:
        print(f"report   -> {write_report(a.out, ev)}")


def cmd_demo(a):
    gds = a.root / "A" / a.split
    sem = a.root / "C" / a.split
    ref = a.root / "B" / a.split
    for d in (gds, sem):
        if not d.is_dir():
            sys.exit(f"missing {d} (demo needs A/{a.split} and C/{a.split})")
    a.out.mkdir(parents=True, exist_ok=True)
    inp = a.out / "input"
    (inp / "A").mkdir(parents=True, exist_ok=True)
    for p in gds.iterdir():
        (inp / "A" / p.name).write_bytes(p.read_bytes())
    if ref.is_dir():
        (inp / "B").mkdir(exist_ok=True)
        for p in ref.iterdir():
            (inp / "B" / p.name).write_bytes(p.read_bytes())

    print("== injecting trojans into clean generated SEMs ==")
    run_script("inject_trojans.py",
               ["--gds-dir", str(gds), "--sem-dir", str(sem),
                "--out-dir", str(inp / "C"), "--round-robin",
                "--rate", str(a.rate), "--max-per-image", "3",
                "--seed", str(a.seed)])
    print("\n== screening ==")
    D = a.out / "D"
    run_script("detect_trojans.py", ["--root", str(inp), "--out", str(D)])
    print("\n== scoring vs ground truth ==")
    ev = run_script("eval_detection.py",
                    ["--truth", str(inp / "C" / "ground_truth.json"),
                     "--results", str(D / "results.json")], capture=True)
    print(ev)
    print(f"report   -> {write_report(D, ev)}")


def cmd_eval(a):
    print(run_script("eval_detection.py",
                     ["--truth", str(a.truth), "--results", str(a.results),
                      "--iou", str(a.iou)], capture=True))


def cmd_remote(a):
    try:
        import asyncio
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client
    except ImportError:
        sys.exit("remote mode needs `pip install mcp` on this machine")

    async def go():
        async with streamablehttp_client(a.server) as (r, w, _):
            async with ClientSession(r, w) as sess:
                await sess.initialize()
                if a.action == "patterns":
                    res = await sess.call_tool("list_trojan_patterns", {})
                elif a.action == "detect":
                    if not a.input_dir:
                        sys.exit("remote detect needs --input-dir")
                    args = {"input_dir": a.input_dir,
                            "backend": a.backend,
                            "min_confidence": a.min_conf}
                    if a.output_subdir:
                        args["output_subdir"] = a.output_subdir
                    res = await sess.call_tool("detect_trojans", args)
                elif a.action == "fetch":
                    if not a.image:
                        sys.exit("remote fetch needs --image")
                    res = await sess.call_tool("show_detection",
                                               {"image_path": a.image})
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


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("detect", help="screen a directory with A/ B/ C/")
    d.add_argument("--root", required=True, type=Path)
    d.add_argument("--out", required=True, type=Path)
    d.add_argument("--backend", choices=["golden", "yolo"], default="golden")
    d.add_argument("--weights", type=Path)
    d.add_argument("--min-conf", type=float, default=0.3)
    d.add_argument("--truth", type=Path,
                   help="optional ground_truth.json to score against")
    d.add_argument("--report", action="store_true",
                   help="write self-contained report.html into --out")
    d.set_defaults(fn=cmd_detect)

    m = sub.add_parser("demo", help="inject -> detect -> eval -> report")
    m.add_argument("--root", required=True, type=Path,
                   help="gds_2_sem directory (A/B/C with --split subdirs)")
    m.add_argument("--split", default="val")
    m.add_argument("--out", required=True, type=Path)
    m.add_argument("--rate", type=float, default=0.75)
    m.add_argument("--seed", type=int, default=0)
    m.set_defaults(fn=cmd_demo)

    e = sub.add_parser("eval", help="score results.json vs ground_truth.json")
    e.add_argument("--truth", required=True, type=Path)
    e.add_argument("--results", required=True, type=Path)
    e.add_argument("--iou", type=float, default=0.3)
    e.set_defaults(fn=cmd_eval)

    r = sub.add_parser("remote",
                       help="call a running MCP service instead of local scripts")
    r.add_argument("--server", required=True,
                   help="e.g. http://HOST:8130/mcp")
    r.add_argument("action", choices=["patterns", "detect", "fetch"])
    r.add_argument("--input-dir", help="detect: path under the service's data root")
    r.add_argument("--output-subdir", default="")
    r.add_argument("--backend", choices=["golden", "yolo"], default="golden")
    r.add_argument("--min-conf", type=float, default=0.3)
    r.add_argument("--image", help="fetch: annotated image path on the service")
    r.add_argument("--save", help="fetch: local filename to save to")
    r.set_defaults(fn=cmd_remote)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
