"""
PDF Ingestion — uses PyMuPDF for text extraction.
Falls back to LLaVA (vision model via Ollama) for scanned/image PDFs.

PII integration:
  After text extraction, PIIGuard scans the full extracted text.
  On hit: status → BLOCKED, text_content cleared, pii_report stored in metadata.
  Vision-extracted pages are also scanned before being appended.
"""
import logging
import time
from pathlib import Path

from multimodal_ds.config import OLLAMA_BASE_URL, VISION_MODEL
from multimodal_ds.core.schema import DataType, ProcessingStatus, Provenance, UnifiedDocument

logger = logging.getLogger(__name__)


def ingest_pdf(file_path: str) -> UnifiedDocument:
    """
    Extract text and structure from a PDF file.
    Strategy:
      1. Try PyMuPDF text extraction (fast, works for digital PDFs)
      2. If text is sparse (scanned PDF), use LLaVA vision model per page
      3. Run PII scan on all extracted text before returning
    """
    import fitz  # PyMuPDF

    path = Path(file_path)
    doc = UnifiedDocument(
        data_type=DataType.PDF,
        status=ProcessingStatus.PROCESSING,
        provenance=Provenance(
            source_path=str(path),
            processor="pdf_ingestion",
            raw_size_bytes=path.stat().st_size if path.exists() else 0,
        )
    )

    t0 = time.time()
    try:
        pdf = fitz.open(file_path)
        doc.page_count = len(pdf)
        all_text = []
        image_pages = []

        for page_num, page in enumerate(pdf):
            text = page.get_text().strip()
            if len(text) > 50:
                all_text.append(f"[Page {page_num + 1}]\n{text}")
            else:
                image_pages.append(page_num)

        doc.text_content = "\n\n".join(all_text)
        doc.metadata["total_pages"] = doc.page_count
        doc.metadata["text_pages"] = doc.page_count - len(image_pages)
        doc.metadata["image_pages"] = len(image_pages)

        # If >50% pages are image-based, use vision model
        if len(image_pages) > doc.page_count * 0.5:
            logger.info(f"[PDF] Scanned PDF detected — using vision model for {len(image_pages)} pages")
            vision_texts = _extract_with_vision(pdf, image_pages, file_path)
            doc.text_content += "\n\n" + "\n\n".join(vision_texts)
            doc.provenance.model_used = VISION_MODEL
        else:
            doc.provenance.model_used = "pymupdf"

        pdf.close()

        # ── PII scan — runs on all extracted text ──────────────────────────
        doc = _run_pdf_pii_scan(doc)

        # Only mark DONE if PII scan didn't block
        if doc.status != ProcessingStatus.BLOCKED:
            doc.status = ProcessingStatus.DONE

    except Exception as e:
        logger.error(f"[PDF] Ingestion failed for {file_path}: {e}")
        doc.status = ProcessingStatus.FAILED
        doc.metadata["error"] = str(e)

    doc.provenance.processing_time_s = round(time.time() - t0, 2)
    logger.info(
        f"[PDF] Ingested {path.name} — {doc.page_count} pages, "
        f"status={doc.status.value} in {doc.provenance.processing_time_s}s"
    )
    return doc


def _run_pdf_pii_scan(doc: UnifiedDocument) -> UnifiedDocument:
    """
    Run PIIGuard on all extracted PDF text.
    On BLOCKED: clears text_content, sets status, stores audit report.
    Fail-safe: any scan error blocks the document.
    """
    try:
        from multimodal_ds.config import ENABLE_PII
        if not ENABLE_PII:
            return doc
        from multimodal_ds.core.pii_guard import get_pii_guard
    except ImportError:
        return doc

    if not doc.text_content or not doc.text_content.strip():
        return doc

    guard = get_pii_guard()
    try:
        pii_report = guard.scan_text(
            doc.text_content,
            source=Path(doc.provenance.source_path).name,
        )
        doc.metadata["pii_report"] = pii_report.to_dict()

        if pii_report.blocked:
            logger.warning(
                f"[PDF] PII BLOCKED — "
                f"entities: {pii_report.entity_types_found}"
            )
            doc.status = ProcessingStatus.BLOCKED
            doc.text_content = (
                f"[BLOCKED: PII detected — entity types: "
                f"{', '.join(pii_report.entity_types_found)}]"
            )

    except Exception as e:
        logger.error(f"[PDF] PII scan failed: {e} — blocking as fail-safe")
        doc.status = ProcessingStatus.BLOCKED
        doc.metadata["pii_report"] = {"blocked": True, "error": str(e)}
        doc.text_content = "[BLOCKED: PII scan error — fail-safe block applied]"

    return doc


def _extract_with_vision(pdf, page_nums: list[int], file_path: str) -> list[str]:
    """Use either ColPali (preferred) or the fallback LLaVA vision model to describe image‑only PDF pages.
    Returns a list of page‑wise textual descriptions.
    """
    # Try ColPali first
    try:
        from multimodal_ds.ingestion.pdf_visual_parsing_agent import extract_layout_text
        colpali_results = extract_layout_text(file_path, page_nums)
        if colpali_results:
            logger.info(f"[PDF Vision] ColPali extracted {len(colpali_results)} pages")
            return colpali_results
    except Exception as e:
        logger.warning(f"[PDF Vision] ColPali extraction failed: {e}")

    # Fallback to original LLaVA‑based vision extraction
    import base64
    import httpx

    results = []
    for page_num in page_nums[:5]:  # Limit to first 5 image pages
        try:
            page = pdf[page_num]
            pix = page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes("png")
            img_b64 = base64.b64encode(img_bytes).decode()

            response = httpx.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": VISION_MODEL.replace("ollama/", ""),
                    "prompt": (
                        "Extract and describe all text, tables, charts, and figures "
                        "visible in this document page. Be thorough and structured."
                    ),
                    "images": [img_b64],
                    "stream": False,
                },
                timeout=120,
            )
            if response.status_code == 200:
                text = response.json().get("response", "")
                results.append(f"[Page {page_num + 1} — Vision]\n{text}")
        except Exception as e:
            logger.warning(f"[PDF Vision] Page {page_num} failed: {e}")

    return results

