"""
Audio Ingestion — uses local OpenAI Whisper (runs on CPU/GPU, no API key).
Produces diarized transcript with entity extraction.
"""
import logging
import time
from pathlib import Path

from multimodal_ds.core.schema import DataType, ProcessingStatus, Provenance, UnifiedDocument

logger = logging.getLogger(__name__)

SUPPORTED_AUDIO = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".mp4", ".webm"}


def ingest_audio(file_path: str, model_size: str = "base") -> UnifiedDocument:
    """
    Transcribe audio using local Whisper model and perform speaker diarization.
    model_size options: tiny, base, small, medium, large
    (base = good balance of speed/accuracy for local use)
    """
    path = Path(file_path)
    doc = UnifiedDocument(
        data_type=DataType.AUDIO,
        status=ProcessingStatus.PROCESSING,
        provenance=Provenance(
            source_path=str(path),
            processor="whisper_local",
            model_used=f"whisper-{model_size}",
            raw_size_bytes=path.stat().st_size if path.exists() else 0,
        )
    )

    t0 = time.time()
    try:
        import whisper

        logger.info(f"[Audio] Loading Whisper {model_size} model...")
        model = whisper.load_model(model_size)

        logger.info(f"[Audio] Transcribing {path.name}...")
        result = model.transcribe(
            str(file_path),
            verbose=False,
            word_timestamps=True,
        )

        # Build structured transcript
        segments = result.get("segments", [])
        transcript_lines = []
        for seg in segments:
            start = _format_time(seg["start"])
            end = _format_time(seg["end"])
            text = seg["text"].strip()
            transcript_lines.append(f"[{start} → {end}] {text}")

        doc.text_content = "\n".join(transcript_lines)
        doc.metadata["language"] = result.get("language", "unknown")
        doc.metadata["duration_s"] = segments[-1]["end"] if segments else 0
        doc.metadata["segment_count"] = len(segments)
        doc.metadata["full_text"] = result.get("text", "")

        # Simple entity extraction from transcript
        entities = _extract_entities(result.get("text", ""))
        doc.metadata["entities"] = entities

        # ------------------------------------------------------------
        # Speaker Diarization (optional – uses pyannote.audio if available)
        # ------------------------------------------------------------
        speaker_transcript = {}
        try:
            from pyannote.audio import Pipeline
            diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                                    use_auth_token=False)
            diarization = diarization_pipeline(file_path)
            # diarization is an Annotation; iterate over turns
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                # Find overlapping transcription segments
                overlapping_texts = []
                for seg in segments:
                    seg_start, seg_end = seg["start"], seg["end"]
                    if seg_start < turn.end and seg_end > turn.start:
                        overlapping_texts.append(seg["text"].strip())
                if overlapping_texts:
                    speaker_key = f"speaker_{speaker}"
                    speaker_transcript.setdefault(speaker_key, []).extend(overlapping_texts)
        except Exception as e:
            logger.warning(f"[Audio] Diarization unavailable or failed: {e}")

        # Store speaker‑wise transcript (concatenated)
        if speaker_transcript:
            doc.metadata["speaker_transcript"] = {k: " ".join(v) for k, v in speaker_transcript.items()}

        doc.status = ProcessingStatus.DONE

    except ImportError:
        logger.error("[Audio] whisper not installed. Run: pip install openai-whisper")
        doc.status = ProcessingStatus.FAILED
        doc.metadata["error"] = "whisper not installed"
    except Exception as e:
        logger.error(f"[Audio] Transcription failed: {e}")
        doc.status = ProcessingStatus.FAILED
        doc.metadata["error"] = str(e)

    doc.provenance.processing_time_s = round(time.time() - t0, 2)
    logger.info(f"[Audio] Transcribed {path.name} in {doc.provenance.processing_time_s}s")
    return doc


def _format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _extract_entities(text: str) -> dict:
    """Simple regex-based entity extraction (no external model needed)."""
    import re
    return {
        "numbers": re.findall(r'\b\d+(?:\.\d+)?\b', text)[:20],
        "percentages": re.findall(r'\b\d+(?:\.\d+)?%', text)[:10],
        "dates": re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b', text)[:10],
        "word_count": len(text.split()),
    }
