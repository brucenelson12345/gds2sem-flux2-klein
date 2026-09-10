"""
Client for Claude models served through an **Open WebUI** instance.

Open WebUI exposes an OpenAI-compatible API:
    GET  {base}/api/models              list the models you can use
    POST {base}/api/chat/completions    chat completion
    Authorization: Bearer <api key>     (keys look like sk-...)

Create a key in Open WebUI under Settings -> Account -> API Keys (the
instance must have API keys enabled). Nothing here talks to Anthropic
directly — every call goes to your Open WebUI instance, so this works on an
internal network with no outside access.

Credential resolution order (first hit wins):
    1. explicit arguments / --api-key --url
    2. environment: OPENWEBUI_API_KEY, OPENWEBUI_URL, OPENWEBUI_MODEL
    3. config file: $SEM_TROJAN_CONFIG, else ~/.config/sem-trojan-detect/config.json
       (written by `screen.py llm login`, chmod 600)

The key is never printed or written to reports; only a masked form is shown.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_URL = "http://localhost:3000"
CONFIG_PATH = Path(os.environ.get(
    "SEM_TROJAN_CONFIG",
    Path.home() / ".config" / "sem-trojan-detect" / "config.json"))


class LLMError(RuntimeError):
    """Anything that went wrong talking to Open WebUI."""


# --------------------------------------------------------------------------
# credentials
# --------------------------------------------------------------------------
def mask(key: str) -> str:
    if not key:
        return "(none)"
    return f"{key[:6]}…{key[-4:]}" if len(key) > 12 else "…" * len(key)


def load_config() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text())
    except Exception:
        return {}


def save_config(url: str, api_key: str, model: str = "") -> Path:
    """Persist credentials with owner-only permissions."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    cfg.update({"url": url.rstrip("/"), "api_key": api_key})
    if model:
        cfg["model"] = model
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    try:
        CONFIG_PATH.chmod(0o600)
    except OSError:
        pass
    return CONFIG_PATH


def resolve(api_key=None, url=None, model=None):
    """Return (url, api_key, model, source) from flags > env > config file."""
    cfg = load_config()
    if api_key:
        source = "flag"
    elif os.environ.get("OPENWEBUI_API_KEY"):
        api_key, source = os.environ["OPENWEBUI_API_KEY"], "env"
    elif cfg.get("api_key"):
        api_key, source = cfg["api_key"], f"config {CONFIG_PATH}"
    else:
        source = "none"

    url = (url or os.environ.get("OPENWEBUI_URL") or cfg.get("url")
           or DEFAULT_URL).rstrip("/")
    model = (model or os.environ.get("OPENWEBUI_MODEL") or cfg.get("model")
             or "auto")
    return url, api_key, model, source


# --------------------------------------------------------------------------
# transport
# --------------------------------------------------------------------------
def _request(url: str, path: str, api_key: str, payload=None, timeout=180):
    if not api_key:
        raise LLMError(
            "no Open WebUI API key. Set OPENWEBUI_API_KEY, pass --api-key, or "
            "run:  screen.py llm login --url http://HOST:3000 --api-key sk-...")
    req = urllib.request.Request(url.rstrip("/") + path)
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Accept", "application/json")
    if payload is not None:
        req.add_header("Content-Type", "application/json")
        req.data = json.dumps(payload).encode()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        body = (e.read() or b"")[:400].decode(errors="replace")
        if e.code in (401, 403):
            raise LLMError(
                f"Open WebUI rejected the API key ({e.code}). Check the key is "
                f"current and that API keys are enabled on {url}.") from None
        if e.code == 404:
            raise LLMError(
                f"{url}{path} returned 404 — is the base URL right? It should "
                f"be the Open WebUI root, e.g. http://host:3000 (no /api).") from None
        raise LLMError(f"Open WebUI HTTP {e.code} on {path}: {body}") from None
    except urllib.error.URLError as e:
        raise LLMError(f"cannot reach Open WebUI at {url}: {e.reason}") from None


def list_models(api_key=None, url=None) -> list[str]:
    """Model ids available to this key."""
    url, key, _, _ = resolve(api_key, url)
    data = _request(url, "/api/models", key, timeout=30)
    items = data.get("data", data if isinstance(data, list) else [])
    out = []
    for m in items:
        mid = m.get("id") or m.get("name") if isinstance(m, dict) else str(m)
        if mid:
            out.append(mid)
    return sorted(out)


def pick_model(models: list[str], prefer="claude") -> str | None:
    """Choose a Claude model when --model auto. Prefers opus, then sonnet."""
    c = [m for m in models if prefer in m.lower()]
    if not c:
        return models[0] if models else None
    for tier in ("opus", "sonnet", "haiku"):
        hit = [m for m in c if tier in m.lower()]
        if hit:
            return sorted(hit)[-1]
    return sorted(c)[-1]


def chat(messages, model=None, api_key=None, url=None, temperature=0.2,
         max_tokens=1400) -> str:
    """One chat completion; returns the assistant text."""
    url, key, cfg_model, _ = resolve(api_key, url, model)
    model = model or cfg_model
    if not model or model == "auto":
        model = pick_model(list_models(key, url))
        if not model:
            raise LLMError("no models available to this Open WebUI key")
    data = _request(url, "/api/chat/completions", key, {
        "model": model, "messages": messages,
        "temperature": temperature, "max_tokens": max_tokens, "stream": False})
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        raise LLMError(f"unexpected response shape from Open WebUI: "
                       f"{json.dumps(data)[:400]}") from None


# --------------------------------------------------------------------------
# the screening-specific prompt
# --------------------------------------------------------------------------
SYSTEM = """You are a hardware-assurance analyst reviewing an automated \
hardware-trojan screen of SEM images taken from manufactured chips. A \
detector compared each suspect SEM against a golden model (the GDS layout \
and the original known-good SEM) and produced the findings below.

Write a concise triage summary for an engineer:
- Lead with the overall verdict: how many images screened, flagged, clean.
- Group the findings by change class (additions, bridges, modifications, \
deletions) and say what each implies physically (e.g. a bridge is a short \
between two nets; a deletion is missing material).
- Call out the images that most need human eyes first, and why.
- Note anything that looks like a likely false positive (e.g. a single tiny \
region on an otherwise clean die).
Be precise and brief. Use plain prose with short lists. These are potential \
trojans requiring human review, not confirmed findings — say so once, do not \
belabour it. Never claim a clean result guarantees the absence of a trojan."""


def _compact(report: dict, max_images=40) -> str:
    s = report.get("summary", {})
    cat = {c["pattern"]: c for c in report.get("catalog", [])}
    lines = [f"backend: {report.get('backend')}",
             f"images screened: {s.get('images')}, flagged: {s.get('flagged')}, "
             f"clean: {s.get('clean')}, total detections: {s.get('detections')}",
             "",
             "pattern legend (key: name [class] - description):"]
    for k, c in cat.items():
        lines.append(f"  {k}: {c['name']} [{c['class']}] - {c['description']}")
    counts = s.get("per_pattern", {})
    lines += ["", "detections per pattern: "
              + ", ".join(f"{k}={v}" for k, v in counts.items() if v), "",
              "flagged images:"]
    n = 0
    for name, rec in report.get("images", {}).items():
        if rec.get("status") != "trojan_detected":
            continue
        n += 1
        if n > max_images:
            lines.append(f"  ... and {s.get('flagged', 0) - max_images} more")
            break
        dets = "; ".join(f"{d['pattern']}({d['name']}) box={d['bbox']} "
                         f"conf={d['confidence']}" for d in rec["detections"])
        lines.append(f"  {name}: {dets}")
    if n == 0:
        lines.append("  (none — every image was clean)")
    return "\n".join(lines)


def summarize_run(report: dict, eval_text: str = "", model=None,
                  api_key=None, url=None) -> str:
    """Ask Claude (via Open WebUI) for an analyst summary of a screening run."""
    user = _compact(report)
    if eval_text:
        user += ("\n\nThis run was scored against known ground truth "
                 "(a synthetic evaluation):\n" + eval_text)
    return chat([{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": user}],
                model=model, api_key=api_key, url=url)


def ping(api_key=None, url=None) -> tuple[bool, str]:
    """(reachable, message) — for `llm test` and setup verification."""
    url, key, model, src = resolve(api_key, url)
    try:
        models = list_models(key, url)
    except LLMError as e:
        return False, str(e)
    claude = [m for m in models if "claude" in m.lower()]
    chosen = pick_model(models)
    return True, (f"Open WebUI at {url} OK (key {mask(key)} from {src})\n"
                  f"  {len(models)} model(s) available"
                  + (f", {len(claude)} Claude: {', '.join(claude[:6])}"
                     if claude else "")
                  + f"\n  auto-selected model: {chosen}")
