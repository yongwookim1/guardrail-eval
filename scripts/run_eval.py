#!/usr/bin/env python
"""Thin shim around guardrail_eval.cli so `python scripts/run_eval.py ...` works."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guardrail_eval.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
