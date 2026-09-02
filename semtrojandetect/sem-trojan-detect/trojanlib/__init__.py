"""
sem-trojan-detect — hardware-trojan screening for SEM images of chips.

Compares a suspect SEM capture (C) against the golden model you hold — the
GDS layout (A) and the original known-good SEM (B) — flags regions that
differ, and classifies each into one of ten trojan patterns (A-J).

Standalone: this package vendors everything it needs (see imagelib) and has
no source dependency on the gds2sem generator. It talks to that tool, when
it needs generated SEM images, over its ComfyUI HTTP API — see
`trojanlib.gds2sem_client`.

Claude models (for the optional analyst summary) are reached through an
Open WebUI instance — see `trojanlib.llm_client`.

Public API:
    from trojanlib import screen_directory, inject_directory, evaluate
"""
from .detect import detect_image, screen_directory          # noqa: F401
from .llm_client import LLMError, summarize_run              # noqa: F401
from .evaluate import evaluate                              # noqa: F401
from .inject import inject_directory                        # noqa: F401
from .matcher import (MatchParams, match_directories,       # noqa: F401
                      write_match_report)
from .patterns import ALL_KEYS, REGISTRY, catalog           # noqa: F401
from .report import write_report                            # noqa: F401

__version__ = "1.0.0"
