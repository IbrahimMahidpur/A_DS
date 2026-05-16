import os
import tempfile
import json
from pathlib import Path

# Import the graph module functions
from multimodal_ds.graph import _executor_node, _reviewer_node

# Stub CodeExecutionAgent to avoid LLM calls
class DummyCodeExecutionAgent:
    def __init__(self, session_id: str = "default"):
        self.session_id = session_id

    def execute(self, task_description: str, data_context: str = "", file_paths=None, max_retries: int = 2):
        # Return a successful execution with a dummy file created
        return {
            "success": True,
            "code": "# dummy code",
            "output": "Execution successful",
            "full_output": "Full execution output",
            "files_created": ["dummy_output.txt"],
            "error": "",
            "retries_used": 0,
        }

# Stub EvaluationAgent to capture task results
class DummyEvalReport:
    def __init__(self, task_results):
        self.task_results = task_results

    def to_dict(self):
        # Return a simple dict containing the output previews for inspection
        return {"task_outputs": [tr.get("output_preview", "") for tr in self.task_results]}

class DummyEvaluationAgent:
    def __init__(self, session_id: str = "default"):
        self.session_id = session_id

    def evaluate_task_results(self, task_results, data_context="", stat_report=None):
        # Return a dummy report that just echoes the task results
        return DummyEvalReport(task_results)

# Monkey‑patch the agents used inside the graph module
import multimodal_ds.graph as graph
graph.CodeExecutionAgent = DummyCodeExecutionAgent
graph.EvaluationAgent = DummyEvaluationAgent

def test_executor_node_returns_files_created():
    # Minimal state required for _executor_node
    state = {
        "analysis_tasks": [{"name": "test_task", "description": "test description"}],
        "current_step": 0,
        "session_id": "test_session",
        "uploaded_files": [],
        "tabular_summaries": [],
        "full_code_outputs": [],
        "errors": [],
        "visualizations": [],
        "saved_artifacts": [],
        "current_step_files": [],
        "current_step_success": False,
    }

    result = _executor_node(state)
    # The executor should have advanced the step and recorded created files
    assert result["current_step"] == 1
    assert result["current_step_files"]
    assert "dummy_output.txt" in result["current_step_files"]

def test_reviewer_node_includes_output_preview():
    # Prepare state as if the executor has run
    state = {
        "analysis_tasks": [{"name": "test_task", "description": "test description"}],
        "full_code_outputs": ["Mean: 5, Std: 2"],
        "errors": [],
        "visualizations": [],
        "saved_artifacts": [],
        "_last_files_created": ["dummy_output.txt"],
        "session_id": "test_session",
        "tabular_summaries": [],
    }

    result = _reviewer_node(state)
    eval_report = result.get("eval_report")
    assert eval_report is not None
    # The dummy evaluation report should contain the output preview we supplied
    report_dict = eval_report.to_dict()
    assert "Mean: 5" in report_dict["task_outputs"][0]
