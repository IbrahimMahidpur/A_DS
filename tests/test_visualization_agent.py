"""
Tests for VisualizationAgent.
Run with: pytest tests/test_visualization_agent.py -v
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    """8-row churn dataset matching the project sample data."""
    return pd.DataFrame({
        "age":    [25, 32, 41, 28, 55, 38, 29, 47],
        "income": [45000, 72000, 88000, 51000, 120000, 67000, 48000, 95000],
        "churn":  [0, 0, 1, 0, 1, 0, 0, 1],
    })


@pytest.fixture(scope="session")
def larger_df() -> pd.DataFrame:
    """100-row synthetic dataset for testing edge cases."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "age":     rng.integers(20, 70, 100),
        "income":  rng.integers(30000, 150000, 100),
        "tenure":  rng.integers(1, 120, 100),
        "score":   rng.uniform(0, 1, 100),
        "churn":   rng.integers(0, 2, 100),
    })


@pytest.fixture()
def tmp_agent(tmp_path, monkeypatch):
    """Agent with temp working dir — no Ollama needed (narrative uses fallback)."""
    monkeypatch.setattr(
        "multimodal_ds.agents.visualization_agent.OLLAMA_BASE_URL",
        "http://localhost:99999",  # Unreachable — forces fallback narrative
    )
    from multimodal_ds.agents.visualization_agent import VisualizationAgent
    from multimodal_ds.core.message_bus import reset_bus
    reset_bus()
    return VisualizationAgent(session_id="test_session", working_dir=str(tmp_path))


# ── ChartManifest ──────────────────────────────────────────────────────────

class TestChartManifest:
    def test_add_and_to_dict(self, tmp_path):
        from multimodal_ds.agents.visualization_agent import ChartManifest
        m = ChartManifest("sess1")
        m.add("correlation_heatmap", "corr.html", "Correlation Heatmap", "Strong correlation found.", (10, 5))
        d = m.to_dict()
        assert d["chart_count"] == 1
        assert d["charts"][0]["chart_type"] == "correlation_heatmap"
        assert d["charts"][0]["data_shape"] == [10, 5]

    def test_save_creates_json(self, tmp_path):
        from multimodal_ds.agents.visualization_agent import ChartManifest
        m = ChartManifest("sess2")
        m.add("distributions", "dist.html", "Distributions", "Skewed right.", (8, 3))
        saved = m.save(tmp_path)
        assert saved.exists()
        data = json.loads(saved.read_text())
        assert data["session_id"] == "sess2"
        assert data["chart_count"] == 1


# ── Chart Generation ───────────────────────────────────────────────────────

class TestChartGeneration:
    def test_generate_returns_manifest(self, tmp_agent, sample_df):
        manifest = tmp_agent.generate(df=sample_df, target_col="churn")
        assert hasattr(manifest, "charts")
        assert isinstance(manifest.charts, list)

    def test_charts_written_to_disk(self, tmp_agent, sample_df):
        manifest = tmp_agent.generate(df=sample_df, target_col="churn")
        session_dir = Path(tmp_agent.working_dir)
        # At least one chart file should exist
        html_files = list(session_dir.glob("*.html"))
        assert len(html_files) >= 1

    def test_manifest_json_saved(self, tmp_agent, sample_df):
        tmp_agent.generate(df=sample_df, target_col="churn")
        manifest_file = Path(tmp_agent.working_dir) / "chart_manifest.json"
        assert manifest_file.exists()

    def test_chart_types_produced(self, tmp_agent, sample_df):
        manifest = tmp_agent.generate(df=sample_df, target_col="churn")
        produced_types = {c["chart_type"] for c in manifest.charts}
        # With 3 numeric cols and a binary target, these should always appear
        assert "data_quality"       in produced_types
        assert "correlation_heatmap" in produced_types
        assert "distributions"       in produced_types

    def test_target_analysis_present_with_binary_target(self, tmp_agent, sample_df):
        manifest = tmp_agent.generate(df=sample_df, target_col="churn")
        types = {c["chart_type"] for c in manifest.charts}
        assert "target_analysis" in types

    def test_no_target_col_still_generates(self, tmp_agent, sample_df):
        manifest = tmp_agent.generate(df=sample_df, target_col=None)
        assert len(manifest.charts) >= 2  # At least data quality + distributions

    def test_larger_dataset_scatter_matrix(self, tmp_agent, larger_df):
        manifest = tmp_agent.generate(df=larger_df, target_col="churn")
        types = {c["chart_type"] for c in manifest.charts}
        assert "scatter_matrix" in types

    def test_roc_curve_generated(self, tmp_agent, larger_df):
        """ROC curve requires sklearn — skip if not installed."""
        pytest.importorskip("sklearn")
        manifest = tmp_agent.generate(df=larger_df, target_col="churn")
        types = {c["chart_type"] for c in manifest.charts}
        assert "roc_curve" in types

    def test_each_chart_has_narrative(self, tmp_agent, sample_df):
        manifest = tmp_agent.generate(df=sample_df, target_col="churn")
        for chart in manifest.charts:
            assert "narrative" in chart
            assert len(chart["narrative"]) > 10  # Non-empty narrative


# ── Individual Chart Methods ───────────────────────────────────────────────

class TestIndividualCharts:
    def test_missing_value_chart_all_present(self, tmp_agent, sample_df):
        from multimodal_ds.agents.visualization_agent import ChartManifest
        m = ChartManifest("t")
        tmp_agent._chart_missing_values(sample_df, m)
        assert len(m.charts) == 1
        assert m.charts[0]["chart_type"] == "data_quality"

    def test_missing_value_chart_with_nulls(self, tmp_agent):
        from multimodal_ds.agents.visualization_agent import ChartManifest
        df = pd.DataFrame({"a": [1, None, 3], "b": [None, None, 1]})
        m = ChartManifest("t")
        tmp_agent._chart_missing_values(df, m)
        assert len(m.charts) == 1

    def test_correlation_heatmap_single_col_skipped(self, tmp_agent):
        from multimodal_ds.agents.visualization_agent import ChartManifest
        df = pd.DataFrame({"only_col": [1, 2, 3]})
        m = ChartManifest("t")
        tmp_agent._chart_correlation_heatmap(df, ["only_col"], m)
        assert len(m.charts) == 0  # Not enough columns

    def test_feature_importance_from_csv(self, tmp_agent, tmp_path):
        """If a feature_importance.csv exists in session dir, it should be picked up."""
        fi_csv = Path(tmp_agent.working_dir) / "feature_importance.csv"
        fi_csv.write_text("feature,importance\nage,0.6\nincome,0.4\n")

        result = tmp_agent._find_feature_importance()
        assert "age" in result
        assert abs(result["age"] - 0.6) < 1e-6


# ── Message Bus Integration ────────────────────────────────────────────────

class TestVizAgentBusIntegration:
    def test_publishes_viz_request_on_generate(self, tmp_path, sample_df, monkeypatch):
        monkeypatch.setattr(
            "multimodal_ds.agents.visualization_agent.OLLAMA_BASE_URL",
            "http://localhost:99999",
        )
        from multimodal_ds.core.message_bus import get_bus, reset_bus, MessageType
        from multimodal_ds.agents.visualization_agent import VisualizationAgent

        reset_bus()
        bus = get_bus()
        received = []
        bus.subscribe(MessageType.VIZ_REQUEST, received.append)

        agent = VisualizationAgent(session_id="bus_test", working_dir=str(tmp_path))
        agent.generate(df=sample_df, target_col="churn")

        assert len(received) == 1
        assert received[0].payload["session_id"] == "bus_test"

    def test_publishes_viz_complete_on_generate(self, tmp_path, sample_df, monkeypatch):
        monkeypatch.setattr(
            "multimodal_ds.agents.visualization_agent.OLLAMA_BASE_URL",
            "http://localhost:99999",
        )
        from multimodal_ds.core.message_bus import get_bus, reset_bus, MessageType
        from multimodal_ds.agents.visualization_agent import VisualizationAgent

        reset_bus()
        bus = get_bus()
        completed = []
        bus.subscribe(MessageType.VIZ_COMPLETE, completed.append)

        agent = VisualizationAgent(session_id="bus_test2", working_dir=str(tmp_path))
        manifest = agent.generate(df=sample_df, target_col="churn")

        assert len(completed) == 1
        assert completed[0].payload["chart_count"] == len(manifest.charts)
        assert completed[0].payload["chart_count"] > 0


# ── Edge Cases ─────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_dataframe_does_not_crash(self, tmp_agent):
        df = pd.DataFrame({"a": [], "b": []})
        manifest = tmp_agent.generate(df=df)
        assert isinstance(manifest.charts, list)  # May be empty — no crash

    def test_single_row_dataframe(self, tmp_agent):
        df = pd.DataFrame({"age": [30], "income": [50000], "churn": [0]})
        manifest = tmp_agent.generate(df=df, target_col="churn")
        assert isinstance(manifest.charts, list)

    def test_all_null_column_handled(self, tmp_agent):
        df = pd.DataFrame({
            "age":    [25, 32, 41],
            "income": [None, None, None],
            "churn":  [0, 1, 0],
        })
        manifest = tmp_agent.generate(df=df, target_col="churn")
        assert len(manifest.charts) >= 1

    def test_plotly_not_available_returns_empty_manifest(self, tmp_path, monkeypatch):
        """Graceful degradation when plotly isn't installed."""
        import multimodal_ds.agents.visualization_agent as viz_module
        monkeypatch.setattr(viz_module, "_PLOTLY_AVAILABLE", False)

        from multimodal_ds.agents.visualization_agent import VisualizationAgent
        from multimodal_ds.core.message_bus import reset_bus
        reset_bus()

        agent = VisualizationAgent(session_id="no_plotly", working_dir=str(tmp_path))
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        manifest = agent.generate(df=df)
        assert len(manifest.charts) == 0
