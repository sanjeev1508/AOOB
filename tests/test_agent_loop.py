"""Unit tests for agentic loop state transitions (no LLM required)."""

from __future__ import annotations

import unittest

from aoob_agent.agent import (
    SAFETY_CEILING,
    _should_force_review,
    _simulation_was_run,
)
from langchain_core.messages import AIMessage, ToolMessage


class TestAgentLoop(unittest.TestCase):
    def test_simulation_detected_from_tool_messages(self):
        msgs = [
            AIMessage(content="", tool_calls=[{"name": "run_simulation", "args": {}, "id": "1"}]),
            ToolMessage(content="{}", name="run_simulation", tool_call_id="1"),
        ]
        self.assertTrue(_simulation_was_run(msgs))

    def test_no_force_review_under_ceiling_with_high_confidence(self):
        msgs = [ToolMessage(content="{}", name="get_micro_window", tool_call_id="1")] * (
            SAFETY_CEILING - 1
        )
        state = {"confidence_so_far": "high", "simulation_run": True}
        force, _ = _should_force_review(msgs, state)
        self.assertFalse(force)

    def test_force_review_at_ceiling_without_high_confidence(self):
        msgs = [ToolMessage(content="{}", name="get_micro_window", tool_call_id="1")] * SAFETY_CEILING
        state = {"confidence_so_far": "medium", "simulation_run": True}
        force, reason = _should_force_review(msgs, state)
        self.assertTrue(force)
        self.assertIn("Safety ceiling", reason)

    def test_force_review_at_ceiling_without_simulation(self):
        msgs = [ToolMessage(content="{}", name="get_micro_window", tool_call_id="1")] * SAFETY_CEILING
        state = {"confidence_so_far": "high", "simulation_run": False}
        force, _ = _should_force_review(msgs, state)
        self.assertTrue(force)


if __name__ == "__main__":
    unittest.main()
