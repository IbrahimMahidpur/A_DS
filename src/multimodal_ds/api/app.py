"""
FastAPI application for the Multimodal Agentic Data Science Engine.

Endpoints:
  POST /ingest          — Upload + ingest a file, returns UnifiedDocument summary
  POST /analyse         — Upload files + objective → full agentic run
  POST /plan            — Generate analysis plan without executing tasks
  POST /visualize       — Generate chart gallery for uploaded data (standalone)
  GET  /session/{id}    — Fetch all memory entries for a session
  GET  /health          — Liveness check
  GET  /docs            — Auto-generated Swagger UI (built-in)
"""
import logging
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from multimodal_ds.config import OUTPUT_DIR, LOG_LEVEL, API_HOST, API_PORT
from multimodal_ds.ingestion.router import route_and_ingest
from multimodal_ds.graph import build_graph, make_initial_state
from multimodal_ds.agents.planner_agent import run_planner
from multimodal_ds.memory.agent_memory import AgentMemory
from multimodal_ds.core.schema import UnifiedDocument

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Multimodal Agentic DS Engine",
    description=(
        "100% local LLM-powered data science pipeline. "
        "Ingest PDFs, images, audio, and tabular data. "
        "Auto-generate analysis plans and execute them with Ollama models."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared state ───────────────────────────────────────────────────────────
graph = build_graph()
memory = AgentMemory()

# ── Pydantic models ────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    memory_entries: int


class IngestResponse(BaseModel):
    document_id: str
    data_type: str
    status: str
    text_preview: str
    schema_info: dict
    metadata: dict
    provenance: dict


class PlanResponse(BaseModel):
    session_id: str
    hypotheses: list[str]
    task_count: int
    final_plan: str
    tasks: list[dict]


class AnalyseResponse(BaseModel):
    session_id: str
    status: str
    objective: str
    final_report: str
    files_created: list[str]
    tasks_total: int
    tasks_completed: int
    errors: list[str]


class VisualizeResponse(BaseModel):
    session_id: str
    target_col: Optional[str]
    chart_count: int
    charts: list[dict]
    output_url: str


class SessionResponse(BaseModel):
    session_id: str
    entry_count: int
    entries: list[dict]


# ── In-memory document store (per-process) ─────────────────────────────────
_document_store: dict[str, UnifiedDocument] = {}


# ── Utilities ──────────────────────────────────────────────────────────────

def _save_upload(upload: UploadFile) -> Path:
    """Save an uploaded file to a temp location and return its path."""
    suffix = Path(upload.filename).suffix if upload.filename else ""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    shutil.copyfileobj(upload.file, tmp)
    tmp.close()
    return Path(tmp.name)


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Liveness probe — confirms the API is running."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        memory_entries=memory.count(),
    )


@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_file(file: UploadFile = File(...)):
    """
    Upload any supported file and ingest it through the appropriate pipeline.

    Supported formats:
    - **Tabular**: CSV, XLSX, Parquet, JSON, TSV
    - **PDF**: text-based or scanned (vision fallback)
    - **Image**: JPG, PNG, GIF, TIFF, WebP
    - **Audio**: MP3, WAV, M4A, FLAC, OGG
    - **Text**: TXT, MD, RST

    Returns a summary of the ingested document including schema, stats, and text preview.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    tmp_path = _save_upload(file)
    try:
        doc = route_and_ingest(str(tmp_path))
        doc.provenance.source_path = file.filename
        _document_store[doc.id] = doc

        return IngestResponse(
            document_id=doc.id,
            data_type=doc.data_type.value,
            status=doc.status.value,
            text_preview=doc.text_content[:500],
            schema_info=doc.schema_info,
            metadata=doc.metadata,
            provenance={
                "source":            file.filename,
                "processor":         doc.provenance.processor,
                "model_used":        doc.provenance.model_used,
                "processing_time_s": doc.provenance.processing_time_s,
                "raw_size_bytes":    doc.provenance.raw_size_bytes,
            },
        )
    except Exception as e:
        logger.exception(f"Ingestion error for {file.filename}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/analyse", response_model=AnalyseResponse, tags=["Analysis"])
async def analyse(
    files: list[UploadFile] = File(...),
    objective: str = Form(...),
    session_id: Optional[str] = Form(None),
):
    """
    Full end-to-end agentic analysis pipeline using LangGraph.

    1. Ingest all uploaded files
    2. Run statistical validation (tabular data)
    3. Generate a hypothesis-driven analysis plan
    4. Execute tasks with RAG context
    5. Generate an executive Markdown report
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    if not objective.strip():
        raise HTTPException(status_code=400, detail="objective cannot be empty")

    session_id = session_id or str(uuid.uuid4())
    tmp_paths: list[Path] = []

    try:
        # Move files to a more permanent location for the session
        session_dir = OUTPUT_DIR / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        for f in files:
            p = session_dir / (f.filename or "upload")
            with open(p, "wb") as buffer:
                shutil.copyfileobj(f.file, buffer)
            saved_paths.append(str(p))

        config = {"configurable": {"thread_id": session_id}}
        initial_state = make_initial_state(
            user_query=objective,
            uploaded_files=saved_paths,
            session_id=session_id
        )
        
        logger.info(f"[API] Starting LangGraph run for session {session_id}")
        result = graph.invoke(initial_state, config=config)
        
        return AnalyseResponse(
            session_id=session_id,
            status="done",
            objective=objective,
            final_report=result.get("final_report", ""),
            files_created=result.get("visualizations", []),
            tasks_total=result.get("steps_total", 0),
            tasks_completed=result.get("current_step", 0),
            errors=result.get("errors", []),
        )

    except Exception as e:
        logger.exception("Analysis pipeline error")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for p in tmp_paths:
            p.unlink(missing_ok=True)


@app.post("/plan", response_model=PlanResponse, tags=["Analysis"])
async def generate_plan(
    files: list[UploadFile] = File(...),
    objective: str = Form(...),
    session_id: Optional[str] = Form(None),
):
    """
    Generate an analysis plan **without executing** any tasks.
    Useful for previewing the plan before committing to a full run.
    """
    session_id = session_id or str(uuid.uuid4())
    tmp_paths: list[Path] = []

    try:
        for f in files:
            tmp_paths.append(_save_upload(f))

        from multimodal_ds.ingestion.router import ingest_multiple
        documents = ingest_multiple([str(p) for p in tmp_paths])

        plan_state = run_planner(
            user_objective=objective,
            documents=documents,
            session_id=session_id,
        )

        tasks = plan_state.get("analysis_plan", [])
        return PlanResponse(
            session_id=session_id,
            hypotheses=plan_state.get("hypotheses", []),
            task_count=len(tasks),
            final_plan=plan_state.get("final_plan", ""),
            tasks=tasks,
        )
    except Exception as e:
        logger.exception("Plan generation error")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for p in tmp_paths:
            p.unlink(missing_ok=True)


@app.post("/visualize", response_model=VisualizeResponse, tags=["Analysis"])
async def visualize(
    files: list[UploadFile] = File(...),
    target_col: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
):
    """
    Generate a Plotly chart gallery for uploaded tabular data.

    Standalone endpoint — does **not** require a prior `/analyse` run.

    Charts generated (auto-selected by data shape):
    - Data quality / missing value overview
    - Feature distributions (split by target if provided)
    - Pearson correlation heatmap
    - Target analysis — class balance + box plots
    - Scatter matrix (colored by target)
    - Feature importance (if model artifact found in session dir)
    - ROC curve (Logistic Regression baseline)

    Each chart includes an LLM-generated statistical narrative paragraph.

    - **target_col**: Binary/classification target column (auto-detected if omitted)
    - **session_id**: Directory charts are written into; creates new session if omitted
    """
    from multimodal_ds.agents.visualization_agent import VisualizationAgent
    from multimodal_ds.ingestion.router import ingest_multiple
    from multimodal_ds.core.schema import DataType

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    session_id = session_id or str(uuid.uuid4())
    tmp_paths: list[Path] = []

    try:
        for f in files:
            tmp_paths.append(_save_upload(f))

        documents = ingest_multiple([str(p) for p in tmp_paths])
        tabular_docs = [
            d for d in documents
            if d.data_type == DataType.TABULAR and d.structured_data is not None
        ]

        if not tabular_docs:
            raise HTTPException(
                status_code=422,
                detail="No tabular data found. Upload a CSV, XLSX, Parquet, or JSON file.",
            )

        primary_df = tabular_docs[0].structured_data

        # Auto-detect target column if not provided
        resolved_target = target_col
        if not resolved_target:
            for doc in tabular_docs:
                suggestion = doc.metadata.get("automl_suggestion", {})
                candidates = suggestion.get("target_candidates", [])
                if candidates:
                    resolved_target = candidates[0]
                    break

        viz_agent = VisualizationAgent(
            session_id=session_id,
            working_dir=str(OUTPUT_DIR),
        )
        manifest = viz_agent.generate(df=primary_df, target_col=resolved_target)
        manifest_dict = manifest.to_dict()

        return VisualizeResponse(
            session_id=session_id,
            target_col=resolved_target,
            chart_count=manifest_dict["chart_count"],
            charts=manifest_dict["charts"],
            output_url=f"/output/{session_id}",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Visualization error")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for p in tmp_paths:
            p.unlink(missing_ok=True)


@app.get("/session/{session_id}", response_model=SessionResponse, tags=["Memory"])
def get_session(session_id: str):
    """Retrieve all stored memory entries for a given session."""
    entries = memory.get_session_history(session_id)
    return SessionResponse(
        session_id=session_id,
        entry_count=len(entries),
        entries=entries,
    )


@app.get("/output/{session_id}", tags=["Output"])
def list_output_files(session_id: str):
    """List all files generated during a session (plots, CSVs, models, chart manifest)."""
    session_dir = OUTPUT_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    files = [
        {
            "filename":   f.name,
            "size_bytes": f.stat().st_size,
            "url":        f"/output/{session_id}/download/{f.name}",
            "type":       "chart" if f.suffix == ".html" else
                          "model" if f.suffix == ".pkl" else
                          "data"  if f.suffix == ".csv" else
                          "manifest" if f.name == "chart_manifest.json" else "other",
        }
        for f in sorted(session_dir.iterdir())
        if f.is_file()
    ]
    return {"session_id": session_id, "file_count": len(files), "files": files}


@app.get("/output/{session_id}/download/{filename}", tags=["Output"])
def download_output_file(session_id: str, filename: str):
    """Download a generated output file (chart HTML, CSV, model pkl, etc.)."""
    file_path = OUTPUT_DIR / session_id / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=str(file_path), filename=filename)


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("multimodal_ds.api.app:app", host=API_HOST, port=API_PORT, reload=True)
