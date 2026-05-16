"""
Ingestion Router Agent — detects file type and routes to correct ingestion module.
This is the entry point for ALL data ingestion in the system.

PII gate (added Conversation 5):
  Every document passes through PIIGuard before being returned.
  If PII is detected the document status is set to BLOCKED and ingestion
  stops — the structured_data and text_content are cleared so nothing
  leaks downstream.

  Block behavior is identical regardless of detection surface:
    - Column name match (e.g. a column literally named 'ssn')
    - Column value match (e.g. a column named 'id' containing '123-45-6789')
    - Free-text match (PDF/TXT containing credit card numbers)

  The PIIReport is stored in doc.metadata['pii_report'] for audit trail.
"""
import logging
from pathlib import Path
from typing import Union

from multimodal_ds.core.schema import DataType, ProcessingStatus, UnifiedDocument
from multimodal_ds.ingestion.pdf_ingestion import ingest_pdf
from multimodal_ds.ingestion.audio_ingestion import ingest_audio, SUPPORTED_AUDIO
from multimodal_ds.ingestion.image_ingestion import ingest_image, SUPPORTED_IMAGES
from multimodal_ds.ingestion.tabular_ingestion import ingest_tabular, SUPPORTED_TABULAR

logger = logging.getLogger(__name__)

# Lazy import — PIIGuard only loaded when config enables it
def _get_pii_guard():
    from multimodal_ds.config import ENABLE_PII
    if not ENABLE_PII:
        return None
    try:
        from multimodal_ds.core.pii_guard import get_pii_guard
        return get_pii_guard()
    except ImportError:
        logger.warning("[Router] pii_guard module not found — PII scanning disabled")
        return None


def _apply_pii_gate(doc: UnifiedDocument) -> UnifiedDocument:
    """
    Run PII scan appropriate to document type.
    Returns doc unchanged if clean, or doc with status=BLOCKED if PII found.

    Interview answer to "how do you handle sensitive data?":
      We gate at ingestion. Before any downstream agent sees the document,
      presidio scans text and tabular values. On a hit, we block and record
      entity types. Nothing leaks — structured_data and text_content are
      cleared on BLOCKED documents.
    """
    guard = _get_pii_guard()
    if guard is None:
        return doc

    # Skip if already blocked by module-level PII scan
    if doc.status == ProcessingStatus.BLOCKED:
        logger.debug("[Router] Document already blocked — skipping router PII gate")
        return doc

    pii_report = None

    try:
        if doc.data_type == DataType.TABULAR and doc.structured_data is not None:
            # Tabular: scan column names + sampled values
            pii_report = guard.scan_dataframe(
                doc.structured_data,
                source=Path(doc.provenance.source_path).name,
            )
        elif doc.text_content:
            # Text / PDF / Audio transcript: full text scan
            pii_report = guard.scan_text(
                doc.text_content,
                source=Path(doc.provenance.source_path).name,
            )

    except Exception as e:
        logger.error(f"[Router] PII scan raised unexpectedly: {e} — blocking as fail-safe")
        doc.status = ProcessingStatus.BLOCKED
        doc.metadata["pii_report"] = {"blocked": True, "error": str(e), "scan_method": "error"}
        doc.text_content = ""
        doc.structured_data = None
        return doc

    if pii_report is None:
        return doc

    # Store audit trail regardless of outcome
    doc.metadata["pii_report"] = pii_report.to_dict()

    if pii_report.blocked:
        logger.warning(
            f"[Router] PII BLOCKED — {Path(doc.provenance.source_path).name} "
            f"| entities: {pii_report.entity_types_found} "
            f"| surfaces: {pii_report.blocked_surfaces}"
        )
        doc.status = ProcessingStatus.BLOCKED
        # Clear sensitive content — never let it reach downstream agents
        doc.text_content = (
            f"[BLOCKED: PII detected — entity types: "
            f"{', '.join(pii_report.entity_types_found)}]"
        )
        doc.structured_data = None

    return doc


def route_and_ingest(file_path: str) -> UnifiedDocument:
    """
    Main entry point. Detects file type, routes to appropriate ingestion pipeline,
    then applies PII gate before returning.

    Returns a UnifiedDocument regardless of input type.
    Status will be one of: DONE | FAILED | BLOCKED
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    logger.info(f"[Router] Ingesting {path.name} (type: {ext})")

    if ext == ".pdf":
        doc = ingest_pdf(file_path)

    elif ext in SUPPORTED_AUDIO:
        doc = ingest_audio(file_path)

    elif ext in SUPPORTED_IMAGES:
        doc = ingest_image(file_path)

    elif ext in SUPPORTED_TABULAR:
        doc = ingest_tabular(file_path)

    elif ext in {".txt", ".md", ".rst"}:
        doc = _ingest_plain_text(file_path)

    else:
        logger.warning(f"[Router] Unknown file type: {ext} — attempting text ingestion")
        doc = _ingest_plain_text(file_path)

    # ── PII gate — runs on every document regardless of type ──────────────
    # Only gate documents that were successfully processed
    if doc.status == ProcessingStatus.DONE:
        doc = _apply_pii_gate(doc)

    return doc


def ingest_multiple(file_paths: list[str]) -> list[UnifiedDocument]:
    """
    Ingest multiple files and return list of UnifiedDocuments.
    BLOCKED documents are included in the result so callers can
    report them — they are not silently dropped.
    """
    results = []
    for fp in file_paths:
        try:
            doc = route_and_ingest(fp)
            status_label = doc.status.value.upper()
            if doc.status == ProcessingStatus.BLOCKED:
                logger.warning(
                    f"[Router] [BLOCKED] {Path(fp).name} — "
                    f"PII: {doc.metadata.get('pii_report', {}).get('entity_types_found', [])}"
                )
            else:
                logger.info(
                    f"[Router] [OK] {Path(fp).name} -> "
                    f"{doc.data_type.value} ({status_label})"
                )
            results.append(doc)
        except Exception as e:
            logger.error(f"[Router] [ERROR] Failed to ingest {fp}: {e}")
    return results


def _ingest_plain_text(file_path: str) -> UnifiedDocument:
    """Simple text file ingestion."""
    from multimodal_ds.core.schema import DataType, ProcessingStatus, Provenance, UnifiedDocument
    import time

    path = Path(file_path)
    doc = UnifiedDocument(
        data_type=DataType.TEXT,
        provenance=Provenance(
            source_path=str(path),
            processor="plain_text",
            raw_size_bytes=path.stat().st_size if path.exists() else 0,
        )
    )
    try:
        doc.text_content = path.read_text(encoding="utf-8", errors="replace")
        doc.metadata["char_count"] = len(doc.text_content)
        doc.metadata["word_count"] = len(doc.text_content.split())
        doc.status = ProcessingStatus.DONE
    except Exception as e:
        doc.status = ProcessingStatus.FAILED
        doc.metadata["error"] = str(e)
    return doc
