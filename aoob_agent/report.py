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
    schema_version: int = Field(
        default=2,
        description="Report schema version for backward-compatible consumers.",
    )
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
    classification: Literal["true", "false", "true (low)", "undecided", "review"] = Field(
        description=(
            "Agent's own verdict: true = real out-of-bounds bug; "
            "true (low) = low-reachability / potential unguarded OOB bug; "
            "false = false positive (tools show the access is bounded); "
            "undecided = inter-procedural / dynamic state context needed; "
            "review = incomplete evidence or unparseable state."
        ),
    )
    human_tag_pattern: Optional[str] = Field(
        default="",
        description=(
            "Pattern tag matching human ground-truth commentary reasoning: "
            "'ALG TOOL: A1968 DELTA:' (tool over-approximation on multi-core lookup), "
            "'e' (enclosing loop invariant / socket bound), or "
            "'h' (high-risk unguarded array access)."
        ),
    )
    token_savings_percent: float = Field(
        default=0.0,
        description=(
            "Per-case token savings: 1 - (tokens_used / naive_full_file_tokens), "
            "computed from actual investigation token usage."
        ),
    )
    tokens_used: int = Field(
        default=0,
        description="Estimated or provider-reported tokens consumed this investigation.",
    )
    naive_full_file_tokens: int = Field(
        default=0,
        description="Token estimate for loading the entire source translation unit.",
    )
    micro_window_used: bool = Field(
        default=False,
        description="Whether token-trimmed micro-windowing context was used.",
    )
    reason_for_review: str = Field(
        default="",
        description="Mandatory explanation when classification=review (especially safety-ceiling cases).",
    )
    derived_from: Optional[int] = Field(
        default=None,
        description="Sibling Order id when verdict was propagated from a structurally identical case.",
    )
    safety_ceiling_hit: bool = Field(
        default=False,
        description="True when investigation hit the tool-call safety ceiling without high confidence.",
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
            "(get_micro_window / query_call_graph / query_data_flow / get_window / move_window / inspect / submit_verdict)."
        ),
    )

