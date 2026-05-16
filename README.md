# Multimodal Agentic Data Science Engine

A fully local, API-key-free agentic data science pipeline. Upload any file — CSV, PDF, image, audio — and get a hypothesis-driven analysis plan executed automatically using locally-hosted Ollama models.

---

## Architecture

```
File Upload
    │
    ▼
┌─────────────┐     ┌───────────────────────────────────────────┐
│   Router    │────▶│           Ingestion Pipeline              │
│  (router.py)│     │  PDF │ Image │ Audio │ Tabular │ Text     │
└─────────────┘     └───────────────────────────────────────────┘
                                      │
                                      ▼  UnifiedDocument
                    ┌─────────────────────────────────────────────┐
                    │            Agent Orchestrator               │
                    │  ┌─────────┐  ┌──────────┐  ┌──────────┐  │
                    │  │ Planner │→ │StatAgent │→ │CodeAgent │  │
                    │  │ +LangGraph  │(stats)   │  │(exec)    │  │
                    │  └─────────┘  └──────────┘  └──────────┘  │
                    │                    │                        │
                    │              AgentMemory                    │
                    │            (ChromaDB)                       │
                    └─────────────────────────────────────────────┘
                                      │
                                      ▼
                            FastAPI REST API
                          (plots, CSVs, reports)
```

### Components

| Module | Role |
|---|---|
| `graph.py` | **Core Engine**: LangGraph StateGraph topology (Production) |
| `agents/planner_agent.py` | Hypothesis generation + Task decomposition |
| `agents/code_execution_agent.py` | RAG-augmented Python code generation + Sandboxed execution |
| `agents/reporter.py` | Generates executive Markdown reports from session artifacts |
| `agents/statistical_agent.py` | Validates statistical assumptions (Normality, VIF, etc.) |
| `core/state.py` | Canonical `AgentState` schema (Msgpack serializable) |
| `ingestion/router.py` | Multimodal routing + PIIGuard gating |
| `memory/agent_memory.py` | ChromaDB vector store for RAG and Audit Trails |
| `cli.py` | Typer CLI (`mmads run`, `mmads serve`) |

---

## Quick Start

### Option A — Docker Compose (recommended)

```bash
# Clone and start everything (Ollama + models + API)
git clone <your-repo>
cd multimodal-agentic-ds

docker compose up --build

# On first run, model bootstrap pulls qwen2.5:7b, llava:7b, nomic-embed-text
# This takes a few minutes depending on your connection speed.

# API is available at:
open http://localhost:8000/docs
```

### Option B — Local Python

**Prerequisites:** Python 3.12, [Ollama](https://ollama.com) running locally

```bash
# 1. Pull required models
ollama pull qwen2.5:7b
ollama pull llava:7b
ollama pull nomic-embed-text

# 2. Install
git clone <your-repo>
cd multimodal-agentic-ds
cp .env.example .env
pip install -e ".[dev]"

# 3. Start API
mmads serve
# or: uvicorn multimodal_ds.api.app:app --reload

# 4. Open Swagger UI
open http://localhost:8000/docs
```

---

## API Reference

### `GET /health`
Liveness probe. Returns API version and memory entry count.

### `POST /ingest`
Upload a file for ingestion. Returns document ID, extracted text preview, schema info.

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@data/sales.csv"
```

### `POST /analyse`
Full end-to-end pipeline: ingest → statistical checks → plan → execute.

```bash
curl -X POST http://localhost:8000/analyse \
  -F "files=@data/sales.csv" \
  -F "objective=Predict monthly churn and identify the top 3 risk factors" \
  -F "max_tasks=6"
```

### `POST /plan`
Generate a plan **without executing** tasks. Good for reviewing before a full run.

```bash
curl -X POST http://localhost:8000/plan \
  -F "files=@data/sales.csv" \
  -F "objective=Segment customers by lifetime value"
```

### `GET /session/{session_id}`
Retrieve all stored memory entries for a session.

### `GET /output/{session_id}`
List all files generated during a session (plots, CSVs, models).

### `GET /output/{session_id}/download/{filename}`
Download a generated file.

---

## CLI Reference

```bash
# Start API server
mmads serve --port 8000 --reload

# Ingest a single file
mmads ingest data/sales.csv
mmads ingest report.pdf --json   # Raw JSON output

# Full pipeline run
mmads run data/sales.csv \
  --objective "Forecast next quarter revenue" \
  --max-tasks 5

# Multiple files
mmads run sales.csv notes.pdf \
  --objective "Reconcile sales data against written reports"

# Inspect session memory
mmads memory abc12345 --n 20
```

---

## Configuration

All settings via environment variables (copy `.env.example` → `.env`):

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `PLANNER_MODEL` | `ollama/qwen2.5:7b` | Hypothesis + planning LLM |
| `CODER_MODEL` | `ollama/qwen2.5:7b` | Code generation LLM |
| `REVIEWER_MODEL` | `ollama/qwen2.5:7b` | Statistical interpretation LLM |
| `VISION_MODEL` | `ollama/llava:7b` | Image/scanned PDF description |
| `EMBED_MODEL` | `ollama/nomic-embed-text` | Vector embeddings for memory |
| `CHROMA_PERSIST_DIR` | `./data/chroma` | ChromaDB persistence path |
| `WORKING_DIR` | `./agentic_output` | Output directory for generated files |
| `MAX_ITERATIONS` | `10` | Max agent loop iterations |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `API_PORT` | `8000` | FastAPI port |

---

## Running Tests

```bash
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src/multimodal_ds --cov-report=term-missing
```

Tests cover: schema, tabular ingestion, text ingestion, router, statistical agent, memory, and all API endpoints.

---

## Supported File Formats

| Category | Extensions |
|---|---|
| Tabular | `.csv` `.xlsx` `.xls` `.parquet` `.json` `.tsv` |
| PDF | `.pdf` (text or scanned) |
| Image | `.jpg` `.jpeg` `.png` `.gif` `.bmp` `.tiff` `.webp` |
| Audio | `.mp3` `.wav` `.m4a` `.ogg` `.flac` `.mp4` `.webm` |
| Text | `.txt` `.md` `.rst` |

---

## Project Structure

```
multimodal-agentic-ds/
├── src/
│   └── multimodal_ds/
│       ├── __init__.py
│       ├── config.py
│       ├── cli.py
│       ├── core/
│       │   └── schema.py          # UnifiedDocument, DataType, Provenance
│       ├── ingestion/
│       │   ├── router.py          # File type detection + routing
│       │   ├── tabular_ingestion.py
│       │   ├── pdf_ingestion.py
│       │   ├── image_ingestion.py
│       │   └── audio_ingestion.py
│       ├── agents/
│       │   ├── orchestrator.py    # End-to-end pipeline coordinator
│       │   ├── planner_agent.py   # LangGraph hypothesis + task planning
│       │   ├── code_execution_agent.py
│       │   └── statistical_agent.py
│       ├── memory/
│       │   └── agent_memory.py    # ChromaDB persistent memory
│       └── api/
│           └── app.py             # FastAPI application
├── tests/
│   └── test_pipeline.py
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── .env.example
```

---

## State Schema Updates

The `AgentState` now includes several new fields to support richer workflows and metadata tracking:

- **`web_results`** (optional `str`): Holds text returned from an external web‑search when the planner detects the objective references unknown data sources.
- **`session_verdict`** (optional `str`): Added to the evaluation report and surfaced in the final report to indicate overall quality (e.g., `PASS`, `FAIL`, `UNKNOWN`).
- **`overall_session_score`** (optional `float`): A unified quality score (0‑10) computed from the individual dimension scores in the evaluation step.
- **`flagged_count`** (optional `int`): Number of evaluation flags raised across tasks.
- **`current_step_success`** (bool): Tracks whether the most recent step succeeded, used for retry logic.
- **`_last_success`** (bool) and **`_last_files_created`** (list) remain for backward compatibility but are now documented.

These fields are added automatically by the planner, evaluation agent, and reviewer components. They enable downstream agents (e.g., the retry node) to make decisions based on external context and evaluation outcomes.

## AgentMemory TTL & Eviction

`AgentMemory` now supports automatic expiration of stored vectors:

- The constructor accepts an optional `ttl_seconds` argument (default **24 h**). Entries older than this TTL are purged on every `store` operation.
- Expired entries are removed from the underlying Chroma collection via the private `_purge_expired` method.
- Retrieval (`retrieve`) automatically filters out stale entries before returning results.

This keeps the memory footprint bounded and ensures that stale context does not influence new sessions.

## Enhanced Evaluation Scoring

The evaluation agent now produces a richer `EvalReport` structure:

```json
{
  "session_id": "...",
  "task_count": 12,
  "flagged_count": 3,
  "pass_count": 9,
  "overall_session_score": 7.4,
  "session_verdict": "PASS",
  "evaluations": [
    {
      "task_name": "task_1",
      "overall_score": 8,
      "flagged": false,
      "dimensions": [
        {"name": "statistical_validity", "score": 9, "reasoning": "..."},
        {"name": "hallucination_risk", "score": 7, "reasoning": "..."},
        {"name": "data_leakage", "score": 8, "reasoning": "..."},
        {"name": "output_completeness", "score": 6, "reasoning": "..."}
      ]
    }
  ]
}
```

Key improvements:
- **`overall_session_score`** aggregates weighted dimension scores across all tasks (weights configurable in `EvaluationAgent`).
- **`session_verdict`** provides a high‑level pass/fail/unknown classification derived from the overall score and flagged count.
- Each task includes **dimension‑level scores** and optional flagging reasons, making it easier to pinpoint problem areas.

The reporter agent now includes this enriched evaluation summary in the final markdown report.

---

*Documentation updated to reflect these changes. Please review the corresponding source files for implementation details.*

