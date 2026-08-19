"""Structured report schema produced by the agent."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class AffectedSymbol(BaseModel):
    name: str = Field(description="Symbol name as used in source / data-flow.")
    kind: Literal["variable", "array", "pointer", "struct", "field", "other", "unknown"] = (
        Field(description="Agent-inferred symbol kind.")
    )
    evidence: str = Field(
        description="Short evidence from tools (data-flow row, source text, etc.)."
    )


class ManipulationStep(BaseModel):
    order: int = Field(description="1-based step index in the sequence.")
    function: str
    access: str = Field(description="read / write / other as observed.")
    variable: str
    location: str = Field(default="", description="Source location if known.")
    note: str = Field(default="", description="Agent note for this step.")


class BoundsEvidence(BaseModel):
    """Agent-authored account of declaration / index-origin evidence.

    Filled by the LLM from tool results — never computed/forced in Python.
    """

    symbol: str = Field(description="Array / pointer / indexed symbol examined.")
    declared_size: Optional[int] = Field(
        default=None,
        description="Literal size from the compiled case file when available.",
    )
    index_expression: str = Field(
        default="",
        description="Index expression text at the alarm site (e.g. idx_a + idx_b).",
    )
    index_inferred_range: str = Field(
        default="",
        description=(
            "Agent-inferred runtime range for the traced index expression "
            "from tool evidence."
        ),
    )
    index_safe_range: str = Field(
        default="",
        description="Safe admissible range derived from target capacity/bounds.",
    )
    target_capacity: Optional[int | str] = Field(
        default=None,
        description="Capacity of indexed target (integer literal or 'Dynamic').",
    )
    guards_found: list[str] = Field(
        default_factory=list,
        description="Guarding C statements relevant to bounds safety.",
    )
    index_origin_summary: str = Field(
        default="",
        description="Agent's own account of where the index/pointer value came from.",
    )
    evidence_completeness: Literal[
        "fully traced", "partially traced", "not traced"
    ] = Field(
        default="not traced",
        description="Agent's own characterization of how complete the backward trace was.",
    )


class AlarmInvestigationReport(BaseModel):
    alarm_order_id: int
    alarm_type: str = ""
    alarm_category: str = ""
    location: str = ""
    # Astrée diagnostic text copied from tools (intervals etc.) — not a human label.
    astree_message: str = Field(
        default="",
        description=(
            "Astrée Message from the alarm export (e.g. index interval vs declared "
            "range). Copied from tool evidence; not a human true/false verdict."
        ),
    )
    affected_symbols: list[AffectedSymbol] = Field(default_factory=list)
    function_name: Optional[str] = None
    function_snippet: str = Field(
        default="",
        description="Relevant C snippet (may be truncated by the agent).",
    )
    manipulation_sequence: list[ManipulationStep] = Field(default_factory=list)
    bounds_evidence: Optional[BoundsEvidence] = Field(
        default=None,
        description="Optional agent-filled bounds / index-origin evidence summary.",
    )
    summary: str = Field(
        description="Concise agent-written investigation summary for this alarm."
    )
    # Agent-authored triage — never read from CSV Classification/Comment columns.
    classification: Literal["true", "false", "review"] = Field(
        description=(
            "Agent's own verdict: true = real out-of-bounds bug; "
            "false = false positive (tools show the access is bounded); "
            "review = cannot tell reachable OOB from analyzer FP. "
            "Incomplete evidence is review, not false. "
            "Decided from tool evidence only."
        ),
    )
    comment: str = Field(
        default="",
        description=(
            "Agent's short rationale / note supporting classification "
            "(free text; not taken from the CSV Comment column)."
        ),
    )
    confidence: Literal["low", "medium", "high"] = "medium"
    tools_used: list[str] = Field(
        default_factory=list,
        description=(
            "Which tools the agent relied on "
            "(get_window / move_window / inspect / submit_verdict)."
        ),
    )
