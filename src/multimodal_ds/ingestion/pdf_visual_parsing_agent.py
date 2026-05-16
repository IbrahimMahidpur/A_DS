"""
PDF Visual Parsing Agent – uses the ColPali model for layout-aware extraction.
Falls back to returning an empty list if ColPali or its dependencies are unavailable.
"""

import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def extract_layout_text(pdf_path: str, page_numbers: List[int]) -> List[str]:
    """Extract layout-aware text from the specified PDF pages using ColPali.

    Parameters
    ----------
    pdf_path: str
        Path to the PDF file.
    page_numbers: List[int]
        Zero-based page indices to process.

    Returns
    -------
    List[str]
        A list of strings, one per page, containing the model's description.
    """
    try:
        # Heavy dependencies – import lazily to avoid import errors when model not installed.
        from transformers import AutoModelForVision2Seq, AutoProcessor
        from PIL import Image
        import fitz  # PyMuPDF
        import io
    except Exception as e:
        logger.warning(f"[ColPali] Required libraries not available: {e}. Skipping visual parsing.")
        return []

    try:
        model_name = "vidore/colpali"
        model = AutoModelForVision2Seq.from_pretrained(model_name)
        processor = AutoProcessor.from_pretrained(model_name)
    except Exception as e:
        logger.warning(f"[ColPali] Failed to load model '{model_name}': {e}. Falling back.")
        return []

    results: List[str] = []
    try:
        pdf = fitz.open(pdf_path)
        for pn in page_numbers:
            try:
                page = pdf[pn]
                pix = page.get_pixmap(dpi=150)
                img_bytes = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                inputs = processor(images=image, return_tensors="pt")
                generated_ids = model.generate(**inputs)
                page_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                results.append(f"[Page {pn + 1}] {page_text}")
            except Exception as page_err:
                logger.warning(f"[ColPali] Page {pn} extraction error: {page_err}")
                continue
        pdf.close()
    except Exception as e:
        logger.error(f"[ColPali] Unexpected error processing PDF: {e}")
        return []

    return results
