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
        # Path-step indices whose shown window omitted part of the function
        # (via pack_path_windows' own bulk-preview line cap — get_window /
        # move_window / inspect always return the complete function and
        # never truncate), and indices the LLM explicitly re-opened via
        # get_window/move_window after that automatic pass. submit_verdict
        # uses these to stop a high-confidence true/false verdict on a step
        # whose truncated window was never actually re-checked.
        self.truncated_steps: set[int] = set()
        self.explicit_review_steps: set[int] = set()

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

    def alarm_window_opened(self) -> bool:
        if not self.case.path:
            return False
        for i, step in enumerate(self.case.path):
            if step.role in {"alarm", "origin_and_alarm"} and i in self.opened:
                return True
        # Single-step path: opening it is enough.
        if len(self.case.path) == 1:
            return 0 in self.opened
        return False

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

    def mark_truncated(self, truncated: bool) -> None:
        if truncated:
            self.truncated_steps.add(self.current_index)

    def mark_explicit_review(self) -> None:
        self.explicit_review_steps.add(self.current_index)

    def alarm_step_index(self) -> Optional[int]:
        if not self.case.path:
            return None
        for i, step in enumerate(self.case.path):
            if step.role in {"alarm", "origin_and_alarm"}:
                return i
        return 0

    def alarm_step_explicitly_reviewed(self) -> bool:
        """True only if the LLM itself called get_window/move_window while
        positioned on the alarm-role step — Python's automatic upfront
        pack_path_windows() pass (which marks every step "opened" for
        bookkeeping/preview purposes) does not count. submit_verdict uses
        this so a verdict can't go through on Python's silent pre-fetch
        alone; the agent must actually navigate to the alarm snippet."""
        idx = self.alarm_step_index()
        return idx is not None and idx in self.explicit_review_steps

    def alarm_window_fully_reviewed(self) -> bool:
        """True unless the alarm step's window was truncated and never
        explicitly re-opened by the LLM via get_window/move_window."""
        idx = self.alarm_step_index()
        if idx is None or idx not in self.truncated_steps:
            return True
        return idx in self.explicit_review_steps


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
