"""Unit tests for submit_verdict enforcement gates (no LLM)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from aoob_agent.case_compiler import compile_case
from aoob_agent.data_store import DataStore
from aoob_agent.session import begin_session, clear_session
from aoob_agent.tools import bind_store, submit_verdict

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


class TestSubmitVerdictGates(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.store = DataStore.load(DATA)

    def setUp(self) -> None:
        clear_session()
        bind_store(self.store)

    def tearDown(self) -> None:
        clear_session()

    def _invoke(self, case_order: int, **kwargs: str) -> dict:
        case = compile_case(self.store, case_order)
        begin_session(case, naive_full_file_tokens=1000)
        bind_store(self.store)
        raw = submit_verdict.invoke(kwargs)
        return json.loads(raw)

    def test_rejects_decisive_verdict_without_simulation(self):
        """Order 7256 has gaps — false without run_simulation must be rejected."""
        out = self._invoke(
            7256,
            classification="false",
            comment="test",
            confidence="high",
            confidence_so_far="high",
        )
        self.assertFalse(out.get("accepted"))
        self.assertIn("run_simulation", str(out.get("error", "")).lower())

    def test_rejects_false_when_gaps_block_structurally(self):
        from aoob_agent.tools import get_micro_window, query_data_flow, run_simulation

        case = compile_case(self.store, 7256)
        begin_session(case, naive_full_file_tokens=1000)
        get_micro_window.invoke({"window_size": 15, "confidence_so_far": "low"})
        query_data_flow.invoke({"confidence_so_far": "medium"})
        run_simulation.invoke({"confidence_so_far": "high"})
        out = json.loads(
            submit_verdict.invoke(
                {
                    "classification": "false",
                    "comment": "test",
                    "confidence": "high",
                    "confidence_so_far": "high",
                }
            )
        )
        self.assertFalse(out.get("accepted"))
        err = str(out.get("error", "")).lower()
        self.assertTrue("missing_df" in err or "derived_max" in err or "alarm-line" in err)

    def test_accepts_review_with_reason_without_simulation(self):
        out = self._invoke(
            7256,
            classification="review",
            comment="untraced origin",
            confidence="low",
            confidence_so_far="low",
            reason_for_review="missing_df gap — index origin not traced",
        )
        self.assertTrue(out.get("accepted"))
        self.assertEqual(out.get("verdict", {}).get("classification"), "review")


if __name__ == "__main__":
    unittest.main()
