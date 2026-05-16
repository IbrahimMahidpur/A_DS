"""
Tests for EvaluationAgent (LLM-as-judge).
Run with: pytest tests/test_evaluation_agent.py -v

LLM calls are mocked — tests run fully offline.
"""
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_message_bus():
    from multimodal_ds.core.message_bus import reset_bus
    reset_bus()
    yield
    reset_bus()


@pytest.fixture
def good_judge_response() -> str:
    """LLM response for a high-quality task output."""
    return json.dumps({
        "statistical_validity":  {"score": 9, "reasoning": "Logistic regression appropriate for binary target."},
        "hallucination_risk":    {"score": 9, "reasoning": "Output references only columns present in data."},
        "data_leakage":          {"score": 10, "reasoning": "Target column excluded from feature matrix."},
        "output_completeness":   {"score": 8, "reasoning": "CSV and PNG outputs produced as expected."},
        "recommendation":        "Proceed to model evaluation with cross-validation.",
    })


@pytest.fixture
def flagged_judge_response() -> str:
    """LLM response triggering EVAL_FLAGGED (low scores)."""
    return json.dumps({
        "statistical_validity":  {"score": 2, "reasoning": "Correlation claimed on 8 rows — insufficient sample."},
        "hallucination_risk":    {"score": 3, "reasoning": "Output references 'revenue' column not in dataset."},
        "data_leakage":          {"score": 5, "reasoning": "Minor concern: target seen during scaling."},
        "output_completeness":   {"score": 7, "reasoning": "Files created but no summary printed."},
        "recommendation":        "Discard this analysis — resample with more data.",
    })


@pytest.fixture
def sample_task_result() -> dict:
    return {
        "step": 1,
        "name": "EDA",
        "success": True,
        "output_preview": "Shape: (8, 3)\nCorrelation matrix computed.\nFiles saved.",
        "files_created": ["summary_statistics.csv", "correlation_heatmap.png"],
        "error": "",
    }


@pytest.fixture
def failed_task_result() -> dict:
    return {
        "step": 2,
        "name": "Feature Engineering",
        "success": False,
        "output_preview": "",
        "files_created": [],
        "error": "KeyError: 'age_poly'",
    }


@pytest.fixture
def tmp_eval_agent(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "multimodal_ds.agents.evaluation_agent.OLLAMA_BASE_URL",
        "http://localhost:99999",
    )
    from multimodal_ds.agents.evaluation_agent import EvaluationAgent
    return EvaluationAgent(session_id="test_session", working_dir=str(tmp_path))


DATA_CONTEXT = "Dataset: 8 rows × 3 columns. Columns: age (int), income (int), churn (int). Target: churn."
STAT_REPORT  = {
    "normality":          {"age": {"is_normal": True}, "income": {"is_normal": True}},
    "multicollinearity":  {"multicollinearity_detected": False},
    "correlation":        {"n_strong": 0},
    "stationarity":       {},
}


# ── DimensionScore ─────────────────────────────────────────────────────────

class TestDimensionScore:
    def test_to_dict_structure(self):
        from multimodal_ds.agents.evaluation_agent import DimensionScore
        ds = DimensionScore(name="statistical_validity", score=8, reasoning="Good.", flagged=False)
        d  = ds.to_dict()
        assert d["name"]      == "statistical_validity"
        assert d["score"]     == 8
        assert d["flagged"]   is False
        assert "reasoning"    in d

    def test_flagged_true_when_below_threshold(self):
        from multimodal_ds.agents.evaluation_agent import DimensionScore, FLAG_DIMENSION_THRESHOLD
        ds = DimensionScore(name="hallucination_risk", score=FLAG_DIMENSION_THRESHOLD - 1, reasoning="Bad.", flagged=True)
        assert ds.flagged is True


# ── TaskEvaluation ────────────────────────────────────────────────────────

class TestTaskEvaluation:
    def test_dimension_accessors(self):
        from multimodal_ds.agents.evaluation_agent import TaskEvaluation, DimensionScore
        eval_result = TaskEvaluation(
            task_name="EDA",
            overall_score=8,
            flagged=False,
            dimensions=[
                DimensionScore("statistical_validity", 9, "Good", False),
                DimensionScore("hallucination_risk",   8, "Ok",   False),
                DimensionScore("data_leakage",         9, "Clean",False),
                DimensionScore("output_completeness",  7, "Fine", False),
            ],
        )
        assert eval_result.statistical_validity == 9
        assert eval_result.hallucination_risk   == 8
        assert eval_result.data_leakage         == 9
        assert eval_result.output_completeness  == 7

    def test_to_dict_contains_all_fields(self):
        from multimodal_ds.agents.evaluation_agent import TaskEvaluation
        te = TaskEvaluation(task_name="t", overall_score=7, flagged=False)
        d  = te.to_dict()
        assert "task_name"      in d
        assert "overall_score"  in d
        assert "flagged"        in d
        assert "flag_reasons"   in d
        assert "dimensions"     in d
        assert "recommendation" in d


# ── EvalReport ────────────────────────────────────────────────────────────

class TestEvalReport:
    def test_save_creates_json(self, tmp_path):
        from multimodal_ds.agents.evaluation_agent import EvalReport
        report = EvalReport(
            session_id="s1", task_count=2, flagged_count=0,
            pass_count=2, overall_session_score=8.5, session_verdict="PASS",
        )
        path = report.save(tmp_path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["session_verdict"] == "PASS"
        assert data["task_count"]      == 2

    def test_to_dict_structure(self):
        from multimodal_ds.agents.evaluation_agent import EvalReport
        report = EvalReport(
            session_id="s1", task_count=3, flagged_count=1,
            pass_count=2, overall_session_score=6.5, session_verdict="WARN",
        )
        d = report.to_dict()
        assert d["session_id"]            == "s1"
        assert d["flagged_count"]         == 1
        assert d["overall_session_score"] == 6.5
        assert d["session_verdict"]       == "WARN"


# ── Parse Judge Response ──────────────────────────────────────────────────

class TestParseJudgeResponse:
    def test_parses_clean_json(self, tmp_eval_agent, good_judge_response):
        result = tmp_eval_agent._parse_judge_response(good_judge_response)
        assert result["statistical_validity"]["score"] == 9
        assert result["hallucination_risk"]["score"]   == 9

    def test_parses_markdown_fenced_json(self, tmp_eval_agent, good_judge_response):
        fenced = f"```json\n{good_judge_response}\n```"
        result = tmp_eval_agent._parse_judge_response(fenced)
        assert result["data_leakage"]["score"] == 10

    def test_parses_json_with_preamble(self, tmp_eval_agent, good_judge_response):
        """LLM sometimes outputs text before the JSON object."""
        mixed = f"Here is my evaluation:\n{good_judge_response}\nEnd."
        result = tmp_eval_agent._parse_judge_response(mixed)
        assert "statistical_validity" in result

    def test_raises_on_missing_keys(self, tmp_eval_agent):
        incomplete = json.dumps({"statistical_validity": {"score": 5, "reasoning": "ok"}})
        with pytest.raises((ValueError, KeyError)):
            tmp_eval_agent._parse_judge_response(incomplete)

    def test_raises_on_unparseable_content(self, tmp_eval_agent):
        with pytest.raises((ValueError, json.JSONDecodeError)):
            tmp_eval_agent._parse_judge_response("This is not JSON at all.")


# ── Build Task Evaluation ─────────────────────────────────────────────────

class TestBuildTaskEvaluation:
    def test_high_scores_not_flagged(self, tmp_eval_agent, good_judge_response):
        raw = json.loads(good_judge_response)
        result = tmp_eval_agent._build_task_evaluation("EDA", raw, ["file.csv"])
        assert result.flagged is False
        assert result.overall_score >= 6
        assert len(result.flag_reasons) == 0

    def test_low_scores_flagged(self, tmp_eval_agent, flagged_judge_response):
        raw = json.loads(flagged_judge_response)
        result = tmp_eval_agent._build_task_evaluation("Bad Task", raw, [])
        assert result.flagged is True
        assert len(result.flag_reasons) >= 1

    def test_scores_clamped_to_0_10(self, tmp_eval_agent):
        raw = {
            "statistical_validity": {"score": 15,  "reasoning": "Off scale."},
            "hallucination_risk":   {"score": -5,  "reasoning": "Below zero."},
            "data_leakage":         {"score": 10,  "reasoning": "Fine."},
            "output_completeness":  {"score": 8,   "reasoning": "Good."},
            "recommendation":       "Fix the scores.",
        }
        result = tmp_eval_agent._build_task_evaluation("Test", raw, [])
        for dim in result.dimensions:
            assert 0 <= dim.score <= 10

    def test_weighted_overall_within_bounds(self, tmp_eval_agent, good_judge_response):
        raw = json.loads(good_judge_response)
        result = tmp_eval_agent._build_task_evaluation("EDA", raw, ["f.csv"])
        assert 0 <= result.overall_score <= 10

    def test_fallback_flag_false_when_fallback_above_threshold(self, tmp_eval_agent):
        """Fallback scores are 5/10 — above the 4-point flag threshold."""
        raw = tmp_eval_agent._fallback_scores(["output.csv"])
        result = tmp_eval_agent._build_task_evaluation("Fallback", raw, ["output.csv"])
        assert result.flagged is False
        assert result.llm_available is False

    def test_fallback_completeness_low_when_no_files(self, tmp_eval_agent):
        raw = tmp_eval_agent._fallback_scores([])
        completeness_score = raw["output_completeness"]["score"]
        assert completeness_score < 5


# ── Evaluate Task (mocked LLM) ────────────────────────────────────────────

class TestEvaluateTask:
    def test_evaluate_task_with_good_output(
        self, tmp_eval_agent, sample_task_result, good_judge_response
    ):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {"content": good_judge_response}
        }
        with patch("httpx.post", return_value=mock_resp):
            result = tmp_eval_agent.evaluate_task(
                task_result=sample_task_result,
                data_context=DATA_CONTEXT,
                stat_report=STAT_REPORT,
            )
        assert result.task_name     == "EDA"
        assert result.overall_score >= 6
        assert result.flagged       is False
        assert len(result.dimensions) == 4

    def test_evaluate_task_with_flagged_output(
        self, tmp_eval_agent, sample_task_result, flagged_judge_response
    ):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {"content": flagged_judge_response}
        }
        with patch("httpx.post", return_value=mock_resp):
            result = tmp_eval_agent.evaluate_task(
                task_result=sample_task_result,
                data_context=DATA_CONTEXT,
            )
        assert result.flagged is True
        assert len(result.flag_reasons) >= 1

    def test_evaluate_task_llm_unavailable_uses_fallback(
        self, tmp_eval_agent, sample_task_result
    ):
        """When LLM is unreachable, fallback scores applied, no crash."""
        with patch("httpx.post", side_effect=Exception("Connection refused")):
            result = tmp_eval_agent.evaluate_task(
                task_result=sample_task_result,
                data_context=DATA_CONTEXT,
            )
        assert result.llm_available is False
        assert result.overall_score == 5   # All fallback = 5, weighted = 5
        assert isinstance(result.dimensions, list)
        assert len(result.dimensions) == 4

    def test_evaluate_task_http_error_uses_fallback(
        self, tmp_eval_agent, sample_task_result
    ):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        with patch("httpx.post", return_value=mock_resp):
            result = tmp_eval_agent.evaluate_task(
                task_result=sample_task_result,
                data_context=DATA_CONTEXT,
            )
        assert result.llm_available is False


# ── Evaluate Task Results (session level) ─────────────────────────────────

class TestEvaluateTaskResults:
    def test_skips_failed_tasks(
        self, tmp_eval_agent, failed_task_result, good_judge_response
    ):
        """Failed tasks are not sent to the judge."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"message": {"content": good_judge_response}}

        with patch("httpx.post", return_value=mock_resp):
            report = tmp_eval_agent.evaluate_task_results(
                task_results=[failed_task_result],   # Only a failed task
                data_context=DATA_CONTEXT,
            )
        assert report.task_count == 0   # No successful tasks to eval

    def test_session_verdict_pass(
        self, tmp_eval_agent, sample_task_result, good_judge_response
    ):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"message": {"content": good_judge_response}}

        with patch("httpx.post", return_value=mock_resp):
            report = tmp_eval_agent.evaluate_task_results(
                task_results=[sample_task_result],
                data_context=DATA_CONTEXT,
                stat_report=STAT_REPORT,
            )
        assert report.session_verdict in ("PASS", "WARN")
        assert report.task_count == 1

    def test_session_verdict_fail(
        self, tmp_eval_agent, flagged_judge_response
    ):
        """All tasks flagged → FAIL verdict."""
        tasks = [
            {"step": i, "name": f"task_{i}", "success": True,
             "output_preview": "minimal", "files_created": [], "error": ""}
            for i in range(3)
        ]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"message": {"content": flagged_judge_response}}

        with patch("httpx.post", return_value=mock_resp):
            report = tmp_eval_agent.evaluate_task_results(
                task_results=tasks,
                data_context=DATA_CONTEXT,
            )
        assert report.flagged_count == 3
        assert report.session_verdict == "FAIL"

    def test_empty_task_list_returns_pass(self, tmp_eval_agent):
        report = tmp_eval_agent.evaluate_task_results(
            task_results=[],
            data_context=DATA_CONTEXT,
        )
        assert report.task_count     == 0
        assert report.session_verdict == "PASS"

    def test_eval_report_saved_to_disk(
        self, tmp_eval_agent, sample_task_result, good_judge_response
    ):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"message": {"content": good_judge_response}}

        with patch("httpx.post", return_value=mock_resp):
            tmp_eval_agent.evaluate_task_results(
                task_results=[sample_task_result],
                data_context=DATA_CONTEXT,
            )
        report_file = Path(tmp_eval_agent.working_dir) / "eval_report.json"
        assert report_file.exists()
        data = json.loads(report_file.read_text())
        assert "session_verdict" in data


# ── Bus Integration ───────────────────────────────────────────────────────

class TestEvalAgentBusIntegration:
    def test_publishes_eval_request(
        self, tmp_path, sample_task_result, good_judge_response, monkeypatch
    ):
        monkeypatch.setattr(
            "multimodal_ds.agents.evaluation_agent.OLLAMA_BASE_URL",
            "http://localhost:99999",
        )
        from multimodal_ds.core.message_bus import get_bus, reset_bus, MessageType
        from multimodal_ds.agents.evaluation_agent import EvaluationAgent

        reset_bus()
        bus = get_bus()
        received = []
        bus.subscribe(MessageType.EVAL_REQUEST, received.append)

        agent = EvaluationAgent(session_id="bus_test", working_dir=str(tmp_path))

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"message": {"content": good_judge_response}}

        with patch("httpx.post", return_value=mock_resp):
            agent.evaluate_task_results(
                task_results=[sample_task_result],
                data_context=DATA_CONTEXT,
            )

        assert len(received) == 1
        assert received[0].payload["session_id"] == "bus_test"

    def test_publishes_eval_flagged_on_bad_output(
        self, tmp_path, sample_task_result, flagged_judge_response, monkeypatch
    ):
        monkeypatch.setattr(
            "multimodal_ds.agents.evaluation_agent.OLLAMA_BASE_URL",
            "http://localhost:99999",
        )
        from multimodal_ds.core.message_bus import get_bus, reset_bus, MessageType
        from multimodal_ds.agents.evaluation_agent import EvaluationAgent

        reset_bus()
        bus = get_bus()
        flagged_msgs = []
        bus.subscribe(MessageType.EVAL_FLAGGED, flagged_msgs.append)

        agent = EvaluationAgent(session_id="flag_test", working_dir=str(tmp_path))

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"message": {"content": flagged_judge_response}}

        with patch("httpx.post", return_value=mock_resp):
            agent.evaluate_task_results(
                task_results=[sample_task_result],
                data_context=DATA_CONTEXT,
            )

        assert len(flagged_msgs) >= 1
        assert flagged_msgs[0].priority.value == 3   # Priority.HIGH

    def test_publishes_eval_complete_on_good_output(
        self, tmp_path, sample_task_result, good_judge_response, monkeypatch
    ):
        monkeypatch.setattr(
            "multimodal_ds.agents.evaluation_agent.OLLAMA_BASE_URL",
            "http://localhost:99999",
        )
        from multimodal_ds.core.message_bus import get_bus, reset_bus, MessageType
        from multimodal_ds.agents.evaluation_agent import EvaluationAgent

        reset_bus()
        bus = get_bus()
        complete_msgs = []
        bus.subscribe(MessageType.EVAL_COMPLETE, complete_msgs.append)

        agent = EvaluationAgent(session_id="complete_test", working_dir=str(tmp_path))

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"message": {"content": good_judge_response}}

        with patch("httpx.post", return_value=mock_resp):
            agent.evaluate_task_results(
                task_results=[sample_task_result],
                data_context=DATA_CONTEXT,
            )

        # Receives both per-task EVAL_COMPLETE and session-level EVAL_COMPLETE
        assert len(complete_msgs) >= 1