import json
import unittest
from pathlib import Path


UI_DIR = Path(__file__).resolve().parent


class EvaluationWorkspaceContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = (UI_DIR / "index.html").read_text(encoding="utf-8")
        cls.js = (UI_DIR / "app.js").read_text(encoding="utf-8")
        cls.css = (UI_DIR / "styles.css").read_text(encoding="utf-8")
        cls.fixture = json.loads(
            (UI_DIR / "fixtures" / "evaluation-realtime.json").read_text(encoding="utf-8")
        )

    def test_unified_workspace_retains_all_agent_lab_controls(self):
        self.assertIn('data-tab="evaluation"', self.html)
        self.assertNotIn('data-tab="agent-lab"', self.html)
        for element_id in (
            "question-matrix-workspace",
            "agent-lab",
            "run-agent-test",
            "agent-results",
            "agent-matrix-workspace",
            "run-agent-matrix",
            "refresh-agent-matrix",
            "replay-agent-matrix-fixture",
        ):
            self.assertIn(f'id="{element_id}"', self.html)
        self.assertIn("function setupEvaluationWorkspace()", self.js)
        self.assertIn("workspace.insertBefore(agentMatrix, questionMatrix)", self.js)
        self.assertIn("workspace.appendChild(agentLab)", self.js)

    def test_completed_agent_matrix_is_the_primary_evaluation_surface(self):
        self.assertIn("Current Agent Evaluation Matrix", self.html)
        self.assertIn('id="agent-matrix-workspace" class="panel agent-matrix-panel evaluation-section evaluation-disclosure" open', self.html)
        self.assertLess(
            self.html.index('href="#agent-matrix-workspace"'),
            self.html.index('href="#question-matrix-workspace"'),
        )
        self.assertIn("terminalRows === rows.length", self.js)
        self.assertIn('if (rows.length) $("agent-matrix-limit").value = rows.length;', self.js)
        self.assertIn("Question-pipeline diagnostics", self.html)
        self.assertIn("Unscored stages are hidden by default", self.html)
        self.assertIn('class="evaluation-disclosure agent-matrix-controls"', self.html)

    def test_question_matrix_hides_unscored_stage_columns_by_default(self):
        self.assertIn("function scoredMatrixStageKeys(items = [])", self.js)
        self.assertIn("function applyMatrixDefaultColumnVisibility(items = [])", self.js)
        self.assertIn('["pass", "fail", "provisional"].includes', self.js)
        self.assertIn("state.matrixVisibleColumnsCustomized = true", self.js)
        self.assertIn("Scored columns only", self.html)

    def test_top_matrix_progress_bars_are_preserved(self):
        self.assertIn("function renderMatrixSummary(totals = {}, totalRows = 0)", self.js)
        self.assertIn('class="matrix-bar"', self.js)
        self.assertIn("renderMatrixSummary(totals, rows.length)", self.js)

    def test_mobile_matrix_has_collapsible_sections_and_live_position(self):
        for element_id in (
            "matrix-run-overview",
            "matrix-action-status",
            "matrix-summary",
            "matrix-table",
            "matrix-detail",
        ):
            self.assertIn(f'id="{element_id}"', self.html)
        self.assertIn("function renderMatrixRunOverview", self.js)
        self.assertIn("Stage ${stageIndex + 1} of ${MATRIX_STAGES.length}", self.js)
        self.assertIn('role="progressbar"', self.js)
        self.assertIn(".evaluation-disclosure", self.css)
        self.assertIn(".matrix-table-shell", self.css)
        self.assertIn(".matrix-summary .matrix-stat", self.css)
        self.assertIn("scroll-snap-type: inline proximity", self.css)

    def test_fixture_drives_every_backend_layer_from_provisional_to_final(self):
        layers = {
            "tool_selection",
            "candidate_recall",
            "document_retention",
            "hop_dependencies",
            "evidence_sufficiency",
            "grounded_answer",
            "latency_token_cost",
        }
        events = self.fixture["events"]
        for backend in ("langgraph", "llamaindex"):
            for layer in layers:
                statuses = [
                    event["cell"]["status"]
                    for event in events
                    if event["backend"] == backend and event["layer"] == layer
                ]
                self.assertEqual(["provisional", "pass"], statuses)
        self.assertIn("applyAgentMatrixLiveCell(event)", self.js)
        self.assertIn(".matrix-cell.provisional", self.css)

    def test_live_job_snapshots_reuse_the_same_provisional_cell_contract(self):
        self.assertIn("mergeAgentMatrixJobSnapshot(job)", self.js)
        self.assertIn('status: "provisional"', self.js)
        self.assertIn('label: "LIVE"', self.js)

    def test_expanded_agent_stage_details_have_isolated_responsive_layout(self):
        self.assertIn('class="agent-matrix-stage-details"', self.js)
        self.assertIn(".agent-matrix-stage-details > div", self.css)
        self.assertIn("grid-template-columns: minmax(150px, 190px) minmax(0, 1fr)", self.css)
        self.assertIn("overflow-wrap: normal", self.css)
        self.assertIn("word-break: normal", self.css)


if __name__ == "__main__":
    unittest.main()
