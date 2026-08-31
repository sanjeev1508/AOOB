"""Per-investigation case session. Never reuse across Order ids."""

from __future__ import annotations

import hashlib
import threading
from typing import Optional

from aoob_agent.case_compiler import CaseFile, PathStep

_LOCK = threading.Lock()
_SESSION: Optional["CaseSession"] = None

# Process-level dedup registry: fingerprint -> (order_id, verdict dict)
_VERDICT_REGISTRY: dict[str, tuple[int, dict]] = {}
_REGISTRY_LOCK = threading.Lock()


def case_fingerprint(case: CaseFile) -> str:
    """Structural fingerprint for sibling alarm deduplication."""
    key = "|".join(
        [
            case.indexed_object.name or "",
            str(case.array_size),
            case.index_expression or "",
            case.operand_symbol or "",
            ",".join(sorted(case.guards)),
            ",".join(sorted(case.write_helpers)),
            ",".join(sorted(case.helpers)),
            ",".join(sorted(case.gaps)),
        ]
    )
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def register_verdict(fingerprint: str, order_id: int, verdict: dict) -> None:
    with _REGISTRY_LOCK:
        _VERDICT_REGISTRY[fingerprint] = (order_id, dict(verdict))


def lookup_verdict(fingerprint: str) -> Optional[tuple[int, dict]]:
    with _REGISTRY_LOCK:
        hit = _VERDICT_REGISTRY.get(fingerprint)
        if hit is None:
            return None
        return hit[0], dict(hit[1])


def clear_verdict_registry() -> None:
    with _REGISTRY_LOCK:
        _VERDICT_REGISTRY.clear()


class CaseSession:
    """Window cursor over a compiled origin → alarm path."""

    def __init__(self, case: CaseFile, *, naive_full_file_tokens: int = 1):
        self.case = case
        self.fingerprint = case_fingerprint(case)
        self.current_index = 0
        self.opened: set[int] = set()
        self.micro_windows_opened: set[int] = set()
        self.inspected: set[str] = set()
        self.call_graph_queries: set[str] = set()
        self.data_flow_queries: set[str] = set()
        self.human_tag_match: str = ""
        self.simulation: Optional[dict] = None
        self.verdict: Optional[dict] = None
        self.confidence_so_far: str = "low"
        self.simulation_run: bool = False
        self.gaps_addressed: bool = not bool(case.gaps)
        self.counterargument_pending: bool = False
        self.counterargument_done: bool = False
        self.force_review: bool = False
        self.force_review_reason: str = ""
        self.tokens_used: int = 0
        self.naive_full_file_tokens: int = max(1, naive_full_file_tokens)
        self.derived_from: Optional[int] = None
        self.prefilter_route: str = ""
        self.lite_investigation: bool = False

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

    def mark_micro_opened(self) -> None:
        self.micro_windows_opened.add(self.current_index)

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
        return bool(n) and (n in set(self.case.helpers) or n in self.call_graph_queries)

    def can_inspect_declaration(self, symbol_name: str) -> bool:
        n = (symbol_name or "").strip()
        if not n:
            return False
        allowed = {self.case.indexed_object.name, self.case.operand_symbol}
        for g in self.case.guards:
            if n in g:
                allowed.add(n)
                break
        return n in allowed or bool(n)

    def gaps_block_false(self) -> tuple[bool, str]:
        """Evidence gates — structural checks, not classification tables."""
        if not self.case.gaps:
            return False, ""
        sim = self.simulation or {}
        cap = self.case.array_size
        dmax = sim.get("derived_max")
        if (
            "missing_df" in self.case.gaps
            and isinstance(cap, int)
            and isinstance(dmax, int)
            and dmax >= cap
        ):
            return True, (
                "missing_df gap with mechanical derived_max >= declared capacity — "
                "index origin is untraced; submit review or true (low), not false."
            )
        obj = self.case.indexed_object.name or ""
        op_root = (self.case.operand_symbol or "").split(".")[-1]
        alarm_fn = self.case.alarm_function or ""
        for g in self.case.guards:
            if alarm_fn and alarm_fn not in g:
                continue
            if obj and obj in g and "[" in g and op_root and op_root in g:
                return True, (
                    "Alarm-line guard dereferences array[index] — do not dismiss as false "
                    "without tracing every write to the index operand."
                )
        return False, ""


def begin_session(case: CaseFile, *, naive_full_file_tokens: int = 1) -> CaseSession:
    global _SESSION
    with _LOCK:
        _SESSION = CaseSession(case, naive_full_file_tokens=naive_full_file_tokens)
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
