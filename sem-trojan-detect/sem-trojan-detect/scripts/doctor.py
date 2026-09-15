#!/usr/bin/env python3
"""
doctor — diagnose a sem-trojan-detect environment.

Deliberately uses the standard library ONLY, so it still runs (and can tell
you why) on a machine where numpy is missing or the wrong interpreter is
being used. Run it anywhere: on the host, or inside the container.

    python3 scripts/doctor.py
    docker run --rm sem-trojan-detect:v1 python /app/scripts/doctor.py

It reports which interpreter is running, whether that interpreter is a
virtualenv, where each dependency was found (apt's dist-packages vs pip's
site-packages — mixing the two is the usual cause of both the build conflict
and ModuleNotFoundError), and what to do next.
"""
import importlib.util
import os
import site
import sys
from pathlib import Path

CORE = ["numpy", "PIL"]
OPTIONAL = {
    "cv2": "faster components/morphology/blur (pure-numpy fallback exists)",
    "mcp": "MCP server and `screen.py remote`",
    "torch": "YOLO backend only",
    "ultralytics": "YOLO backend only",
}
OK, BAD, WARN = "  ok  ", " FAIL ", " note "


def origin(name):
    """Where a module would be imported from, without importing it."""
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    return getattr(spec, "origin", None) if spec else None


def kind_of(path):
    if not path:
        return ""
    p = str(path)
    if "/dist-packages" in p:
        return "apt (dist-packages)"
    if "/site-packages" in p:
        return "pip (site-packages)"
    return ""


def version_of(name):
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "present")
    except Exception as e:                                   # noqa: BLE001
        return f"import failed: {type(e).__name__}: {e}"


def main():
    problems, notes = [], []
    print("=" * 68)
    print("sem-trojan-detect · environment doctor")
    print("=" * 68)

    # ---- interpreter -------------------------------------------------------
    in_venv = sys.prefix != sys.base_prefix
    in_docker = Path("/.dockerenv").exists()
    print(f"\ninterpreter   {sys.executable}")
    print(f"version       {sys.version.split()[0]}")
    print(f"virtualenv    {'yes — ' + sys.prefix if in_venv else 'NO (system Python)'}")
    print(f"in container  {'yes' if in_docker else 'no'}")
    if os.environ.get("VIRTUAL_ENV") and not in_venv:
        problems.append(
            f"VIRTUAL_ENV is set to {os.environ['VIRTUAL_ENV']} but this "
            f"interpreter is not that venv — you are running the wrong python. "
            f"Use {os.environ['VIRTUAL_ENV']}/bin/python.")

    # ---- dependencies ------------------------------------------------------
    print("\ndependencies")
    seen_kinds = set()
    for name in CORE:
        path = origin(name)
        if path is None:
            print(f"{BAD} {name:<12} NOT FOUND")
            problems.append(f"{name} is missing — install it into THIS "
                            f"interpreter: {sys.executable} -m pip install {name}")
        else:
            k = kind_of(path)
            seen_kinds.add(k)
            v = version_of(name)
            if v.startswith("import failed"):
                print(f"{BAD} {name:<12} {v}")
                problems.append(f"{name} is present at {path} but does not "
                                f"import — {v}")
            else:
                print(f"{OK} {name:<12} {v:<12} {k}")
    for name, why in OPTIONAL.items():
        path = origin(name)
        if path is None:
            print(f"{WARN} {name:<12} absent — optional: {why}")
        else:
            k = kind_of(path)
            seen_kinds.add(k)
            v = version_of(name)
            mark = BAD if v.startswith("import failed") else OK
            print(f"{mark} {name:<12} {v:<12} {k}"
                  + ("   (optional — safe to remove)"
                     if mark is BAD else ""))

    if {"apt (dist-packages)", "pip (site-packages)"} <= seen_kinds:
        notes.append(
            "Packages are coming from BOTH apt (dist-packages) and pip "
            "(site-packages). That mix is what produces 'cannot uninstall "
            "numpy 1.26.4' at build time. Prefer a virtualenv so pip owns "
            "everything: python3 -m venv .venv && .venv/bin/pip install -r "
            "docker/requirements.txt")

    # ---- the package itself ------------------------------------------------
    print("\npackage")
    repo = Path(__file__).resolve().parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        import trojanlib
        print(f"{OK} trojanlib    {trojanlib.__version__}  <- {Path(trojanlib.__file__).parent}")
        from trojanlib import imagelib
        print(f"{OK} imagelib     OpenCV path: "
              f"{'active' if imagelib.cv2 is not None else 'pure-numpy fallback'}")
    except Exception as e:                                   # noqa: BLE001
        print(f"{BAD} trojanlib    import failed: {type(e).__name__}: {e}")
        problems.append(f"trojanlib does not import: {e}")

    # ---- verdict -----------------------------------------------------------
    print("\n" + "-" * 68)
    if not problems:
        print("No blocking problems found — screening should run.")
    else:
        print(f"{len(problems)} problem(s) to fix:\n")
        for i, p in enumerate(problems, 1):
            print(f"  {i}. {p}\n")
    for n in notes:
        print(f"  note: {n}\n")

    if problems:
        print("Quickest fix on an offline host with a PyPI mirror:")
        print("  ./offline_prep/install_local.sh --index-url "
              "https://your-mirror/simple")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
