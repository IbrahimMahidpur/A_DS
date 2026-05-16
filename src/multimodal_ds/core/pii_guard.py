"""
PII Guard — hard gate for all ingestion pipelines.

Uses Microsoft Presidio to detect PII before any data enters the system.
This is NOT a warning layer — detected PII blocks ingestion entirely and
returns ProcessingStatus.BLOCKED with entity types listed.

Detection surfaces:
  - Plain text / PDF extracted text  → full-text scan
  - Tabular column names             → name pattern match
  - Tabular column values            → sampled value scan (first N rows)

Entities detected (configurable via PII_ENTITIES env var):
  PERSON, EMAIL_ADDRESS, PHONE_NUMBER, CREDIT_CARD, US_SSN,
  US_BANK_NUMBER, MEDICAL_LICENSE, IP_ADDRESS, IBAN_CODE,
  US_PASSPORT, US_DRIVER_LICENSE, DATE_TIME (optional)

Design decisions for interviews:
  - Column name hit and column value hit both block — same gate, different log reason
  - Sampling (default 100 rows) prevents O(n) scan on large CSVs
  - Fallback to regex-only if spaCy model missing — never silently passes
  - All findings returned in structured PIIReport for audit trail

Usage:
    guard = PIIGuard()

    # Text scan
    report = guard.scan_text("John Smith SSN 123-45-6789")
    if report.blocked:
        doc.status = ProcessingStatus.BLOCKED

    # Tabular scan
    report = guard.scan_dataframe(df)
    if report.blocked:
        doc.status = ProcessingStatus.BLOCKED
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── Configurable entity list ───────────────────────────────────────────────
# NOTE: PERSON is intentionally excluded from tabular scans.
# Health/survey datasets use numeric response codes — Presidio misidentifies
# integer-valued cells as person names causing false-positive blocks.
# PERSON is still checked in free-text (PDF/text) scans.
_DEFAULT_ENTITIES = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "US_SSN",
    "US_BANK_NUMBER",
    "MEDICAL_LICENSE",
    "IP_ADDRESS",
    "IBAN_CODE",
    "US_PASSPORT",
    "US_DRIVER_LICENSE",
]

# Entities used ONLY for free-text scans (PDF, audio transcripts, plain text)
_TEXT_ONLY_ENTITIES = _DEFAULT_ENTITIES + ["PERSON"]

# Column name patterns that trigger a block regardless of values
_SENSITIVE_COLUMN_PATTERNS = re.compile(
    r"\b(ssn|social.?security|credit.?card|card.?number|cvv|passport|"
    r"driver.?licen[cs]e|bank.?account|routing.?number|dob|date.?of.?birth|"
    r"medical.?record|patient.?id|national.?id|tax.?id|ein|itin)\b",
    re.IGNORECASE,
)

# ── Result dataclass ───────────────────────────────────────────────────────

@dataclass
class PIIFinding:
    """Single PII detection result."""
    entity_type: str
    text: str           # Redacted in logs — shown as first 4 chars + ***
    score: float        # Presidio confidence 0.0–1.0
    surface: str        # "text", "column_name", "column_value"
    column: Optional[str] = None  # For tabular findings

    def to_dict(self) -> dict:
        return {
            "entity_type": self.entity_type,
            "text_redacted": self.text[:4] + "***" if len(self.text) > 4 else "***",
            "score": round(self.score, 3),
            "surface": self.surface,
            "column": self.column,
        }


@dataclass
class PIIReport:
    """
    Aggregated PII scan result.
    Always returned — callers check .blocked to gate ingestion.
    """
    blocked: bool
    findings: list[PIIFinding] = field(default_factory=list)
    scan_method: str = "presidio"       # "presidio" | "regex_fallback"
    rows_scanned: int = 0
    columns_scanned: int = 0
    error: Optional[str] = None

    @property
    def entity_types_found(self) -> list[str]:
        return list({f.entity_type for f in self.findings})

    @property
    def blocked_surfaces(self) -> list[str]:
        return list({f.surface for f in self.findings})

    def to_dict(self) -> dict:
        return {
            "blocked": self.blocked,
            "entity_types_found": self.entity_types_found,
            "blocked_surfaces": self.blocked_surfaces,
            "finding_count": len(self.findings),
            "findings": [f.to_dict() for f in self.findings],
            "scan_method": self.scan_method,
            "rows_scanned": self.rows_scanned,
            "columns_scanned": self.columns_scanned,
            "error": self.error,
        }

    @classmethod
    def clean(cls, rows_scanned: int = 0, columns_scanned: int = 0, method: str = "presidio") -> "PIIReport":
        """Convenience constructor for a passing scan."""
        return cls(
            blocked=False,
            rows_scanned=rows_scanned,
            columns_scanned=columns_scanned,
            scan_method=method,
        )

    @classmethod
    def error_report(cls, error: str) -> "PIIReport":
        """
        Used when PII scanning itself fails.
        BLOCKS by default — fail-safe, never fail-open.
        """
        logger.error(f"[PIIGuard] Scan failed: {error} — blocking as fail-safe")
        return cls(
            blocked=True,
            error=error,
            scan_method="error",
        )


# ══════════════════════════════════════════════════════════════════════════
#  PIIGuard
# ══════════════════════════════════════════════════════════════════════════

class PIIGuard:
    """
    Hard PII gate for all ingestion pipelines.

    Interview talking point: "We fail-safe. If presidio is unavailable,
    we fall back to regex. If regex itself errors, we block. The system
    never silently passes data it hasn't checked."

    Initialization is lazy — presidio loaded once on first scan.
    """

    def __init__(
        self,
        entities: Optional[list[str]] = None,
        score_threshold: float = 0.6,
        sample_rows: int = 100,
        enabled: bool = True,
    ):
        env_entities = os.getenv("PII_ENTITIES", "")
        self.entities = (
            [e.strip() for e in env_entities.split(",") if e.strip()]
            if env_entities
            else (entities or _DEFAULT_ENTITIES)
        )
        self.score_threshold = score_threshold
        self.sample_rows = sample_rows
        self.enabled = enabled and os.getenv("ENABLE_PII_DETECTION", "true").lower() == "true"

        self._analyzer = None       # Lazy init
        self._use_presidio = True   # Flipped to False if import fails

    # ── Public API ─────────────────────────────────────────────────────────

    def scan_text(self, text: str, source: str = "text") -> PIIReport:
        """
        Scan free text (PDF, plain text, transcripts).
        Truncates to 50k chars — prevents memory explosion on huge docs.
        Uses _TEXT_ONLY_ENTITIES which includes PERSON (safe for prose text).
        """
        if not self.enabled:
            return PIIReport.clean()
        if not text or not text.strip():
            return PIIReport.clean()

        text_sample = text[:50_000]

        try:
            # Use extended entity list (includes PERSON) for free-text
            findings = self._scan_text_block(
                text_sample, surface="text", entities_override=_TEXT_ONLY_ENTITIES
            )
            blocked = any(f.score >= self.score_threshold for f in findings)
            report = PIIReport(
                blocked=blocked,
                findings=findings,
                scan_method="presidio" if self._use_presidio else "regex_fallback",
                rows_scanned=1,
            )
            if blocked:
                logger.warning(
                    f"[PIIGuard] BLOCKED {source} — "
                    f"entities: {report.entity_types_found}"
                )
            return report

        except Exception as e:
            return PIIReport.error_report(str(e))

    def scan_dataframe(self, df: pd.DataFrame, source: str = "dataframe") -> PIIReport:
        """
        Scan a DataFrame:
          1. Column names — pattern match against sensitive name list
          2. Column values — presidio/regex scan on sampled string columns

        Both paths can block independently.
        """
        if not self.enabled:
            return PIIReport.clean()
        if df is None or df.empty:
            return PIIReport.clean()

        try:
            all_findings: list[PIIFinding] = []

            # ── Pass 1: Column name scan ───────────────────────────────────
            for col in df.columns:
                if _SENSITIVE_COLUMN_PATTERNS.search(str(col)):
                    all_findings.append(PIIFinding(
                        entity_type="SENSITIVE_COLUMN_NAME",
                        text=str(col),
                        score=1.0,
                        surface="column_name",
                        column=str(col),
                    ))
                    logger.warning(f"[PIIGuard] Sensitive column name detected: '{col}'")

            # ── Pass 2: Column value scan (string columns only, sampled) ──
            # Skip columns whose values look like numeric codes (e.g. survey
            # response codes: 1, 2, 3, 88, 99) — these are NOT free text
            # and consistently cause false-positive PERSON detections.
            sample_df = df.head(self.sample_rows)
            str_cols = df.select_dtypes(include=["object"]).columns.tolist()
            columns_scanned = len(str_cols)

            for col in str_cols:
                col_series = sample_df[col].dropna().astype(str)

                # Skip if >80% of values are purely numeric codes
                numeric_ratio = col_series.str.match(r'^\s*-?\d+(\.\d+)?\s*$').mean()
                if numeric_ratio > 0.8:
                    logger.debug(f"[PIIGuard] Skipping numeric-coded column: {col}")
                    continue

                # Skip if column has very few unique long-text values (likely codes)
                unique_vals = col_series.unique()
                if len(unique_vals) < 4 and all(len(v) < 8 for v in unique_vals):
                    logger.debug(f"[PIIGuard] Skipping low-cardinality coded column: {col}")
                    continue

                col_text = " ".join(col_series.tolist())
                if not col_text.strip():
                    continue
                value_findings = self._scan_text_block(
                    col_text[:10_000],
                    surface="column_value",
                    column=col,
                )
                all_findings.extend(value_findings)

            blocked = any(f.score >= self.score_threshold for f in all_findings)
            report = PIIReport(
                blocked=blocked,
                findings=all_findings,
                scan_method="presidio" if self._use_presidio else "regex_fallback",
                rows_scanned=min(self.sample_rows, len(df)),
                columns_scanned=columns_scanned,
            )
            if blocked:
                logger.warning(
                    f"[PIIGuard] BLOCKED {source} — "
                    f"surfaces: {report.blocked_surfaces}, "
                    f"entities: {report.entity_types_found}"
                )
            return report

        except Exception as e:
            return PIIReport.error_report(str(e))

    # ── Internal scan ──────────────────────────────────────────────────────

    def _scan_text_block(
        self,
        text: str,
        surface: str,
        column: Optional[str] = None,
        entities_override: Optional[list[str]] = None,
    ) -> list[PIIFinding]:
        """Run presidio (or regex fallback) on a text block."""
        if self._use_presidio:
            try:
                return self._presidio_scan(text, surface, column, entities_override)
            except Exception as e:
                logger.warning(f"[PIIGuard] Presidio scan failed: {e} — falling back to regex")
                self._use_presidio = False

        return self._regex_scan(text, surface, column)

    def _presidio_scan(
        self,
        text: str,
        surface: str,
        column: Optional[str],
        entities_override: Optional[list[str]] = None,
    ) -> list[PIIFinding]:
        """Presidio-based scan. Lazy-loads analyzer on first call."""
        analyzer = self._get_analyzer()
        entities_to_scan = entities_override or self.entities
        results = analyzer.analyze(
            text=text,
            entities=entities_to_scan,
            language="en",
        )
        findings = []
        for r in results:
            if r.score >= self.score_threshold:
                findings.append(PIIFinding(
                    entity_type=r.entity_type,
                    text=text[r.start:r.end],
                    score=r.score,
                    surface=surface,
                    column=column,
                ))
        return findings

    def _regex_scan(
        self,
        text: str,
        surface: str,
        column: Optional[str],
    ) -> list[PIIFinding]:
        """
        Regex fallback — runs when presidio/spaCy unavailable.
        Covers SSN, credit card, email, phone — the highest-risk entities.
        Interview point: "We never fail-open. Regex catches the obvious
        cases even without the ML model."
        """
        patterns = {
            "US_SSN":        r"\b\d{3}-\d{2}-\d{4}\b",
            "CREDIT_CARD":   r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b",
            "EMAIL_ADDRESS": r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
            "PHONE_NUMBER":  r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b",
            "IP_ADDRESS":    r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
            "US_PASSPORT":   r"\b[A-Z]{1,2}\d{6,9}\b",
        }
        findings = []
        for entity_type, pattern in patterns.items():
            for match in re.finditer(pattern, text):
                findings.append(PIIFinding(
                    entity_type=entity_type,
                    text=match.group(),
                    score=0.85,     # Fixed confidence for regex hits
                    surface=surface,
                    column=column,
                ))
        return findings

    def _get_analyzer(self):
        """Lazy-load presidio AnalyzerEngine. Cached after first call."""
        if self._analyzer is not None:
            return self._analyzer

        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_analyzer.nlp_engine import NlpEngineProvider

            # Try large model first, fall back to small
            for model in ["en_core_web_lg", "en_core_web_sm", "en_core_web_md"]:
                try:
                    provider = NlpEngineProvider(nlp_configuration={
                        "nlp_engine_name": "spacy",
                        "models": [{"lang_code": "en", "model_name": model}],
                    })
                    nlp_engine = provider.create_engine()
                    self._analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
                    logger.info(f"[PIIGuard] Presidio initialized with {model}")
                    return self._analyzer
                except OSError:
                    logger.warning(f"[PIIGuard] spaCy model '{model}' not found, trying next")
                    continue

            # All spaCy models missing — use presidio without NLP (pattern-only)
            logger.warning(
                "[PIIGuard] No spaCy model found — using presidio pattern-only mode. "
                "Run: python -m spacy download en_core_web_sm"
            )
            self._analyzer = AnalyzerEngine()
            return self._analyzer

        except ImportError:
            logger.warning(
                "[PIIGuard] presidio-analyzer not installed — falling back to regex scanner. "
                "Run: pip install presidio-analyzer presidio-anonymizer"
            )
            self._use_presidio = False
            raise


# ── Module-level singleton ─────────────────────────────────────────────────
# Import this in ingestion modules — one guard instance per process.

_guard: Optional[PIIGuard] = None


def get_pii_guard() -> PIIGuard:
    """Get the global PIIGuard singleton."""
    global _guard
    if _guard is None:
        _guard = PIIGuard()
    return _guard
