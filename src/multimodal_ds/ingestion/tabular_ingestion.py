"""
Tabular Data Ingestion — pandas profiling + FLAML AutoML suggestions.
Handles CSV, Excel, Parquet, JSON tabular files.

PII integration (Conversation 5):
  After profiling, PIIGuard scans:
    1. Column names — pattern match (e.g. 'ssn', 'credit_card')
    2. Column values — presidio/regex scan on sampled string columns
  On hit: status → BLOCKED, structured_data cleared, pii_report stored in metadata.
  The router also runs a top-level gate, but this module gates early so
  profiling stats never get stored for PII-containing data either.
"""
import logging
import time
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

from multimodal_ds.core.schema import DataType, ProcessingStatus, Provenance, UnifiedDocument

logger = logging.getLogger(__name__)

SUPPORTED_TABULAR = {".csv", ".xlsx", ".xls", ".parquet", ".json", ".tsv"}


def ingest_tabular(file_path: str) -> UnifiedDocument:
    """
    Ingest tabular data with:
    - Schema detection
    - Statistical profiling
    - PII scan (column names + sampled values)
    - FLAML AutoML task suggestion
    - Missing value / outlier summary
    """
    path = Path(file_path)
    doc = UnifiedDocument(
        data_type=DataType.TABULAR,
        status=ProcessingStatus.PROCESSING,
        provenance=Provenance(
            source_path=str(path),
            processor="tabular_ingestion",
            raw_size_bytes=path.stat().st_size if path.exists() else 0,
        )
    )

    t0 = time.time()
    try:
        df = _load_dataframe(file_path)
        if df is None:
            raise ValueError(f"Could not load dataframe from {file_path}")

        doc.structured_data = df

        # Schema info
        doc.schema_info = {
            "shape": list(df.shape),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "numeric_cols": list(df.select_dtypes(include=np.number).columns),
            "categorical_cols": list(df.select_dtypes(include=["object", "category"]).columns),
            "datetime_cols": list(df.select_dtypes(include=["datetime64"]).columns),
        }

        # Data profile
        doc.data_profile = _compute_profile(df)

        # Text summary for LLM consumption
        doc.text_content = _generate_text_summary(df, doc.schema_info, doc.data_profile)

        doc.metadata["automl_suggestion"] = _suggest_automl_task(df)
        doc.metadata["file_format"] = path.suffix.lower()

        # ── PII scan (tabular-specific: column names + values) ─────────────
        # Note: router.py also runs a gate, but scanning here means we block
        # before profiling stats are ever written downstream.
        doc = _run_tabular_pii_scan(doc, df)

        # Only mark DONE if PII scan didn't block
        if doc.status != ProcessingStatus.BLOCKED:
            doc.status = ProcessingStatus.DONE

    except Exception as e:
        logger.error(f"[Tabular] Ingestion failed for {file_path}: {e}")
        doc.status = ProcessingStatus.FAILED
        doc.metadata["error"] = str(e)

    doc.provenance.processing_time_s = round(time.time() - t0, 2)
    logger.info(
        f"[Tabular] Ingested {path.name} — "
        f"status={doc.status.value} in {doc.provenance.processing_time_s}s"
    )
    return doc


def _run_tabular_pii_scan(doc: UnifiedDocument, df: pd.DataFrame) -> UnifiedDocument:
    """
    Run PIIGuard on the dataframe.
    On BLOCKED: clears structured_data and text_content, sets status.
    Returns doc (modified in place for status/metadata, returned for clarity).
    """
    try:
        from multimodal_ds.config import ENABLE_PII
        if not ENABLE_PII:
            return doc
        from multimodal_ds.core.pii_guard import get_pii_guard
    except ImportError:
        return doc

    guard = get_pii_guard()
    try:
        pii_report = guard.scan_dataframe(
            df,
            source=Path(doc.provenance.source_path).name,
        )
        doc.metadata["pii_report"] = pii_report.to_dict()

        if pii_report.blocked:
            logger.warning(
                f"[Tabular] PII BLOCKED — "
                f"entities: {pii_report.entity_types_found}, "
                f"surfaces: {pii_report.blocked_surfaces}"
            )
            doc.status = ProcessingStatus.BLOCKED
            doc.structured_data = None
            doc.text_content = (
                f"[BLOCKED: PII detected — entity types: "
                f"{', '.join(pii_report.entity_types_found)}]"
            )
            # Clear profiling data too — don't leak stats about PII columns
            doc.data_profile = {}
            doc.schema_info = {}

    except Exception as e:
        logger.error(f"[Tabular] PII scan failed: {e} — blocking as fail-safe")
        doc.status = ProcessingStatus.BLOCKED
        doc.metadata["pii_report"] = {"blocked": True, "error": str(e)}
        doc.structured_data = None
        doc.text_content = "[BLOCKED: PII scan error — fail-safe block applied]"

    return doc


def _load_dataframe(file_path: str) -> Optional[pd.DataFrame]:
    """Load file into pandas DataFrame based on extension."""
    ext = Path(file_path).suffix.lower()
    loaders = {
        ".csv":     lambda: pd.read_csv(file_path),
        ".tsv":     lambda: pd.read_csv(file_path, sep="\t"),
        ".xlsx":    lambda: pd.read_excel(file_path),
        ".xls":     lambda: pd.read_excel(file_path),
        ".parquet": lambda: pd.read_parquet(file_path),
        ".json":    lambda: pd.read_json(file_path),
    }
    loader = loaders.get(ext)
    return loader() if loader else None


def _compute_profile(df: pd.DataFrame) -> dict:
    """Compute statistical profile of the dataframe, ensuring JSON/Msgpack serializability."""
    import numpy as np

    def _sanitize(val):
        """Recursively convert numpy types to Python types."""
        if isinstance(val, dict):
            return {k: _sanitize(v) for k, v in val.items()}
        if isinstance(val, list):
            return [_sanitize(v) for v in val]
        if hasattr(val, "item"): # numpy types
            return val.item()
        if isinstance(val, (np.integer, np.floating)):
            return float(val) if isinstance(val, np.floating) else int(val)
        return val

    profile: dict = {}

    # Optimization for large datasets
    row_limit = 100_000
    is_large = len(df) > row_limit

    numeric_df = df.select_dtypes(include=np.number)
    
    # Limit number of columns in the profile to avoid overwhelming the LLM
    max_cols = 50
    if len(numeric_df.columns) > max_cols:
        logger.info(f"[Tabular] Truncating profile from {len(numeric_df.columns)} to {max_cols} numeric columns")
        numeric_df = numeric_df.iloc[:, :max_cols]

    if not numeric_df.empty:
        # describe() is fast even on large data, but we sample for safety if massive
        desc = numeric_df.describe()
        profile["numeric_stats"] = _sanitize(desc.to_dict())

    profile["missing_values"] = _sanitize(df.isnull().sum().to_dict())
    profile["missing_pct"] = _sanitize((df.isnull().mean() * 100).round(2).to_dict())
    
    # Expensive operations on large datasets
    if is_large:
        logger.info(f"[Tabular] Large dataset ({len(df)} rows) — skipping expensive duplicate check and deep memory usage")
        profile["duplicate_rows"] = "skipped (large dataset)"
        profile["memory_mb"] = round(float(df.memory_usage(deep=False).sum() / 1e6), 2)
    else:
        profile["duplicate_rows"] = int(df.duplicated().sum())
        profile["memory_mb"] = round(float(df.memory_usage(deep=True).sum() / 1e6), 2)

    # Cardinality for categoricals
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    profile["cardinality"] = {col: int(df[col].nunique()) for col in cat_cols}

    # Outlier detection (IQR method) — sample rows if massive to save time
    outlier_counts = {}
    outlier_df = numeric_df
    if len(numeric_df) > 10_000:
        outlier_df = numeric_df.sample(10_000, random_state=42)

    for col in outlier_df.columns:
        try:
            q1, q3 = outlier_df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            outliers = ((outlier_df[col] < q1 - 1.5 * iqr) | (outlier_df[col] > q3 + 1.5 * iqr)).sum()
            if outliers > 0:
                # If we sampled, scale it up
                if len(numeric_df) > 10_000:
                    outliers = int(outliers * (len(numeric_df) / 10_000))
                outlier_counts[col] = int(outliers)
        except Exception:
            continue
    profile["outlier_counts"] = outlier_counts

    return profile


def _generate_text_summary(df: pd.DataFrame, schema: dict, profile: dict) -> str:
    """Generate a natural language summary for LLM consumption."""
    rows, cols = schema["shape"]
    missing_total = sum(profile["missing_values"].values())
    top_missing = sorted(profile["missing_pct"].items(), key=lambda x: -x[1])[:3]

    lines = [
        f"Dataset: {rows} rows × {cols} columns",
        f"Numeric columns ({len(schema['numeric_cols'])}): {', '.join(schema['numeric_cols'][:5])}{'...' if len(schema['numeric_cols']) > 5 else ''}",
        f"Categorical columns ({len(schema['categorical_cols'])}): {', '.join(schema['categorical_cols'][:5])}",
        f"Missing values: {missing_total} total",
    ]
    if top_missing:
        lines.append("Columns with most missing: " + ", ".join(f"{c}={p:.1f}%" for c, p in top_missing))
    if profile.get("outlier_counts"):
        lines.append("Outliers detected in: " + ", ".join(profile["outlier_counts"].keys()))
    if profile.get("duplicate_rows"):
        lines.append(f"Duplicate rows: {profile['duplicate_rows']}")

    # Add describe stats
    if "numeric_stats" in profile:
        lines.append("\nNumeric Summary (first 20 columns, mean ± std):")
        for i, (col, stats) in enumerate(profile["numeric_stats"].items()):
            if i >= 20:
                lines.append("  ... [remaining columns truncated] ...")
                break
            mean = stats.get("mean", 0)
            std = stats.get("std", 0)
            lines.append(
                f"  {col}: mean={mean:.3g}, std={std:.3g}, "
                f"min={stats.get('min', 0):.3g}, max={stats.get('max', 0):.3g}"
            )

    return "\n".join(lines)


def _suggest_automl_task(df: pd.DataFrame) -> dict:
    """Suggest ML task type based on data profile."""
    suggestion = {"task": "unknown", "target_candidates": [], "reason": ""}

    last_col = df.columns[-1] if len(df.columns) > 0 else None
    if last_col:
        n_unique = df[last_col].nunique()
        if n_unique <= 20 and df[last_col].dtype in [object, "category"] or n_unique <= 10:
            suggestion["task"] = "classification"
            suggestion["target_candidates"] = [last_col]
            suggestion["reason"] = (
                f"Last column '{last_col}' has {n_unique} unique values — "
                "likely classification target"
            )
        elif pd.api.types.is_numeric_dtype(df[last_col]):
            suggestion["task"] = "regression"
            suggestion["target_candidates"] = [last_col]
            suggestion["reason"] = (
                f"Last column '{last_col}' is numeric — likely regression target"
            )

    return suggestion
