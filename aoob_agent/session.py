"""Per-investigation case session. Never reuse across Order ids."""

from __future__ import annotations

import threading
from typing import Optional

from aoob_agent.case_compiler import CaseFile, PathStep

_LOCK = threading.Lock()
_SESSION: Optional["CaseSession"] = None


class CaseSession:
    """Window cursor over a compiled origin → alarm path."""

    def __init__(self, case: CaseFile):
        self.case = case
        self.current_index = 0
        self.opened: set[int] = set()
        self.inspected: set[str] = set()
        self.verdict: Optional[dict] = None
        self.size_unknown_retries = 0

    @property
    def n_steps(self) -> int:
        return max(1, len(self.case.path))

    def current_step(self) -> Optional[PathStep]:
        if not self.case.path:
            return None
        idx = min(max(0, self.current_index), len(self.case.path) - 1)
        return self.case.path[idx]

    def mark_opened(self) -> None:
        self.opened.add(self.current_index)

    def move(self, *, direction: str = "", step: int = 0) -> PathStep:
        if not self.case.path:
            raise ValueError("Case path is empty.")
        n = len(self.case.path)
        if step:
            self.current_index = min(max(1, int(step)), n) - 1
        else:
            d = (direction or "next").strip().lower()
            if d in {"next", "forward", "toward_alarm", "callee"}:
                self.current_index = min(self.current_index + 1, n - 1)
            elif d in {"prev", "previous", "back", "caller"}:
                self.current_index = max(self.current_index - 1, 0)
            else:
                raise ValueError(
                    f"Unknown move {direction!r}. Use next, prev, or step=N on the case path."
                )
        return self.current_step()  # type: ignore[return-value]

    def can_inspect(self, name: str) -> bool:
        n = (name or "").strip()
        return bool(n) and n in set(self.case.helpers)


def begin_session(case: CaseFile) -> CaseSession:
    global _SESSION
    with _LOCK:
        _SESSION = CaseSession(case)
        return _SESSION


def get_session() -> CaseSession:
    with _LOCK:
        if _SESSION is None:
            raise RuntimeError("No active case session. compile_case / begin_session first.")
        return _SESSION


def clear_session() -> None:
    global _SESSION
    with _LOCK:
        _SESSION = None
