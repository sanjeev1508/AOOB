#!/usr/bin/env python3
"""Fail if ground-truth CSV filename leaks into investigation code paths."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = "Full_alarms_with_result.csv"
ALLOWED_PREFIXES = (
    ROOT / "scripts",
    ROOT / "data" / "convert.py",
)
SCAN_DIRS = (ROOT / "aoob_agent", ROOT / "cf_viz")


def _allowed(path: Path) -> bool:
    for prefix in ALLOWED_PREFIXES:
        try:
            path.relative_to(prefix if prefix.is_dir() else prefix.parent)
            if prefix.is_file() and path.resolve() != prefix.resolve():
                continue
            return True
        except ValueError:
            continue
    return False


def main() -> int:
    violations: list[str] = []
    for base in SCAN_DIRS:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix in {".pyc"}:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if FORBIDDEN in text and not _allowed(path):
                violations.append(str(path.relative_to(ROOT)))
    if violations:
        print("Ground-truth leak detected:", file=sys.stderr)
        for v in violations:
            print(f"  {v}", file=sys.stderr)
        return 1
    print("OK: no ground-truth filename in aoob_agent/ or cf_viz/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
