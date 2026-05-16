import csv
import tempfile
from pathlib import Path
from unittest.mock import patch

from multimodal_ds.graph import make_initial_state, build_graph


class DummyResponse:
    def __init__(self, content: str):
        self.status_code = 200
        self._content = content

    def json(self):
        return {"message": {"content": self._content}}


def dummy_post(url, json=None, timeout=None):
    """Return dummy Ollama responses for planner calls.
    The first call generates hypotheses, the second generates a task plan.
    """
    # Identify call by the system prompt content
    system = json.get("messages", [{}])[0].get("content", "") if json else ""
    # Simple heuristic: if the system prompt contains "hypotheses" return hypotheses JSON
    if "hypotheses" in system.lower():
        # Return a single hypothesis JSON array
        content = "[{\"id\": \"h1\", \"statement\": \"Churn depends on age and income\", \"analysis_method\": \"logistic regression\", \"expected_outcome\": \"model predicts churn\"}]"
        return DummyResponse(content)
    else:
        # Return a simple task plan JSON array – one EDA task
        content = "[{\"step\": 1, \"name\": \"EDA\", \"type\": \"eda\", \"description\": \"Explore the dataset\", \"tools\": [\"pandas\"], \"expected_output\": \"summary\", \"depends_on\": []}]"
        return DummyResponse(content)


def test_end_to_end_smoke():
    # Create a small CSV with 5 rows and 3 columns
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "data.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["age", "income", "churn"])
            for i in range(5):
                writer.writerow([20 + i, 30000 + i * 1000, i % 2])

        # Initialise state
        state = make_initial_state(
            user_query="predict churn",
            uploaded_files=[str(csv_path)],
            session_id="test_session",
        )

        graph = build_graph()
        # Patch httpx.post used by the planner agent
        with patch("httpx.post", side_effect=dummy_post):
            final_state = graph.invoke(state)

        # Assertions
        assert final_state.get("tabular_summaries"), "Tabular summaries should be populated"
        assert final_state.get("statistical_report"), "Statistical report should be populated"
        assert final_state.get("analysis_tasks"), "Analysis tasks (planner output) should be non‑empty"
        # No blocked files for clean data
        assert not final_state.get("blocked_files"), "There should be no blocked files"
