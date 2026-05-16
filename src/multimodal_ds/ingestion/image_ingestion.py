"""
Image Ingestion — CLIP embeddings + LLaVA description via local Ollama.
No external APIs needed.

Offline-first design:
  - CLIP is attempted only if transformers + torch are already installed
    AND the model is already cached locally.
  - If the model is not cached (or HuggingFace is unreachable), CLIP is
    skipped silently — LLaVA description still runs via Ollama.
  - Set env var MMADS_SKIP_CLIP=1 to always skip CLIP (useful in CI or
    air-gapped environments).
  - To pre-cache the model once: python -c "from transformers import
    CLIPModel, CLIPProcessor; CLIPModel.from_pretrained(
    'openai/clip-vit-base-patch32'); CLIPProcessor.from_pretrained(
    'openai/clip-vit-base-patch32')"
"""
import logging
import os
import time
from pathlib import Path
from typing import Optional

from multimodal_ds.config import OLLAMA_BASE_URL, VISION_MODEL
from multimodal_ds.core.schema import DataType, ProcessingStatus, Provenance, UnifiedDocument

logger = logging.getLogger(__name__)

SUPPORTED_IMAGES = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}

# Respect explicit skip flag — useful for CI / air-gapped machines
_SKIP_CLIP = os.getenv("MMADS_SKIP_CLIP", "0").strip() == "1"


def ingest_image(file_path: str) -> UnifiedDocument:
    """
    Process an image:
    1. Generate CLIP embeddings (offline-only, skipped if model not cached)
    2. Use LLaVA (via Ollama) for natural language description
    """
    path = Path(file_path)
    doc = UnifiedDocument(
        data_type=DataType.IMAGE,
        status=ProcessingStatus.PROCESSING,
        provenance=Provenance(
            source_path=str(path),
            processor="image_ingestion",
            raw_size_bytes=path.stat().st_size if path.exists() else 0,
        )
    )

    t0 = time.time()
    try:
        from PIL import Image

        img = Image.open(file_path)
        doc.metadata["width"] = img.width
        doc.metadata["height"] = img.height
        doc.metadata["mode"] = img.mode
        doc.metadata["format"] = img.format

        # Step 1: CLIP embeddings — offline only, never blocks
        if not _SKIP_CLIP:
            embeddings = _get_clip_embeddings_offline(img)
            if embeddings is not None:
                doc.embeddings = embeddings
                doc.provenance.model_used = "clip-vit-base-patch32"
                doc.metadata["clip_embedded"] = True
            else:
                doc.metadata["clip_embedded"] = False
                doc.metadata["clip_skip_reason"] = "model not cached locally"
        else:
            doc.metadata["clip_embedded"] = False
            doc.metadata["clip_skip_reason"] = "MMADS_SKIP_CLIP=1"

        # Step 2: LLaVA description via local Ollama (always attempted)
        description = _describe_with_llava(file_path)
        doc.text_content = description
        doc.image_descriptions = [description]
        doc.metadata["llava_model"] = VISION_MODEL

        doc.status = ProcessingStatus.DONE

    except Exception as e:
        logger.error(f"[Image] Ingestion failed for {file_path}: {e}")
        doc.status = ProcessingStatus.FAILED
        doc.metadata["error"] = str(e)

    doc.provenance.processing_time_s = round(time.time() - t0, 2)
    return doc


def _get_clip_embeddings_offline(img) -> Optional[list[float]]:
    """
    Generate CLIP embeddings using locally cached model only.
    Never triggers a network download — returns None immediately if:
      - transformers / torch not installed
      - model not present in HuggingFace cache
      - any other error
    """
    try:
        # Check availability without importing heavy libs unnecessarily
        import importlib.util
        if not importlib.util.find_spec("transformers") or not importlib.util.find_spec("torch"):
            return None

        from transformers import CLIPProcessor, CLIPModel
        from transformers.utils import cached_file
        import torch

        model_name = "openai/clip-vit-base-patch32"

        # Check if config is cached — if not, skip without downloading
        try:
            cached_file(
                model_name,
                "config.json",
                local_files_only=True,   # ← KEY: never go to network
            )
        except Exception:
            logger.debug("[CLIP] Model not in local cache — skipping embeddings")
            return None

        # Model is cached — safe to load
        model = CLIPModel.from_pretrained(model_name, local_files_only=True)
        processor = CLIPProcessor.from_pretrained(model_name, local_files_only=True)

        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            features = model.get_image_features(**inputs)
        return features[0].tolist()

    except Exception as e:
        logger.debug(f"[CLIP] Embedding skipped: {e}")
        return None


def _describe_with_llava(file_path: str) -> str:
    """Use LLaVA via Ollama to describe the image."""
    import base64
    import httpx

    try:
        with open(file_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        model_name = VISION_MODEL.replace("ollama/", "")
        response = httpx.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model_name,
                "prompt": (
                    "Describe this image in detail. Include: "
                    "1) Main subject/content, "
                    "2) Any text visible, "
                    "3) Charts, graphs, or data if present, "
                    "4) Key visual features relevant for data analysis."
                ),
                "images": [img_b64],
                "stream": False,
            },
            timeout=120,
        )
        if response.status_code == 200:
            return response.json().get("response", "")
        return f"[LLaVA description unavailable: HTTP {response.status_code}]"

    except Exception as e:
        logger.warning(f"[LLaVA] Description failed: {e}")
        return f"[Image: {Path(file_path).name}] — Vision description unavailable"
