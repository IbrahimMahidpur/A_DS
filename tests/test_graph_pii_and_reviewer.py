import json
import tempfile
from pathlib import Path
import pytest

# Import the graph module
import multimodal_ds.graph as graph

# Stub Presidio components to avoid external dependency
class DummyAnalyzerEngine:
    def analyze(self, text, language="en"):
        # No PII detected
        return []

class DummyAnonymizerEngine:
    def anonymize(self, text, analyzer_results):
        class Result:
            def __init__(self, text):
                self.text = text
        return Result(text)

# Replace Presidio classes with dummies in the graph module
graph.presidio_analyzer = type('module', (), {'AnalyzerEngine': DummyAnalyzerEngine})
graph.presidio_anonymizer = type('module', (), {'AnonymizerEngine': DummyAnonymizerEngine})

# Dummy CodeExecutionAgent used to simulate execution results
class DummyCodeExecutionAgent:
    def __init__(self, session_id: str = "default"):
        self.session_id = session_id

    def execute(self, task_description: str, data_context: str = "", file_paths=None, max_retries: int = 2):
        # Simulate a successful execution that creates a CSV artifact
        return {
            "success": True,
            "code": "# dummy code",
            "output": "Execution succeeded",
            "full_output": "Full execution output with stats: Mean=5, Std=2",
            "files_created": ["result.csv"],
            "error": "",
            "retries_used": 0,
        }

graph.CodeExecutionAgent = DummyCodeExecutionAgent

# Dummy EvaluationAgent to capture reviewer inputs
class DummyEvalReport:
    def __init__(self, task_results):
        self.task_results = task_results

    def to_dict(self):
        # Return a mapping that includes the output previews for inspection
        return {"task_results": [tr.get("output_preview", "") for tr in self.task_results]}

class DummyEvaluationAgent:
    def __init__(self, session_id: str = "default"):
        self.session_id = session_id

    def evaluate_task_results(self, task_results, data_context="", stat_report=None):
        return DummyEvalReport(task_results)

graph.EvaluationAgent = DummyEvaluationAgent

# Helper to create a temporary OUTPUT_DIR for the logger
@pytest.fixture(autouse=True)
def temp_output_dir(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setattr(graph, "OUTPUT_DIR", Path(tmpdir))
        yield

def test_executor_node_generates_files_created(tmp_output_dir):
    # Minimal state required for _executor_node
    state = {
        "analysis_tasks": [{"name": "dummy_task", "description": "do something"}],
        "current_step": 0,
        "session_id": "testsession",
        "uploaded_files": [],
        "tabular_summaries": [],
        "full_code_outputs": [],
        "errors": [],
        "visualizations": [],
        "saved_artifacts": [],
        "current_step_files": [],
        "current_step_success": False,
    }

    result = graph._executor_node(state)
    # Verify that a step was advanced and a file was recorded
    assert result["current_step"] == 1
    assert result["current_step_files"] == ["result.csv"]
    assert result["files_created"] == ["result.csv"]

def test_reviewer_node_output_preview_contains_statistics():
    # Prepare state as if executor produced an output with statistics
    state = {
        "analysis_tasks": [{"name": "stats_task", "description": "compute stats"}],
        "full_code_outputs": ["Mean: 5, Std: 2, Min: 1, Max: 10"],
        "errors": [],
        "visualizations": [],
        "saved_artifacts": [],
        "_last_files_created": ["stats.csv"],
        "session_id": "testsession",
        "tabular_summaries": [],
    }

    result = graph._reviewer_node(state)
    eval_report = result.get("eval_report")
    assert eval_report is not None
    # The dummy eval report returns a dict with task_results list
    report_dict = eval_report.to_dict()
    # Ensure the expected statistics string appears in the output preview
    assert any("Mean: 5" in preview for preview in report_dict["task_results"])
