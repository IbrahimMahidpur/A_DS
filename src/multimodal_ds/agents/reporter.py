"""
Reporter Agent — synthesises all pipeline outputs into a structured
markdown report. This is the final node in the LangGraph StateGraph.

Design:
  - Reads code_outputs, visualizations, errors, eval_report, analysis_plan
  - Calls Ollama to produce a structured narrative report
  - Stores the report in state['final_report'] and writes it to disk
  - Also saves eval_report.json to the session working directory

Report structure:
  1. Executive Summary
  2. Key Findings (numbered, quantitative)
  3. Methodology
  4. Results (with inline chart references)
  5. Evaluation Quality Scores
  6. Limitations
  7. Recommendations
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from multimodal_ds.config import REVIEWER_MODEL, OUTPUT_DIR
from multimodal_ds.core.state import AgentState

logger = logging.getLogger(__name__)

REPORTER_SYSTEM = """You are a senior data scientist writing a final analysis report.

Structure your report EXACTLY as:
# Executive Summary
(2-3 sentences summarising what was done and the main finding)

# Key Findings
1. (quantitative finding with numbers)
2. ...

# Methodology
(what analysis steps were executed)

# Results
(detailed results, reference charts as ![Chart](filename.png))

# Quality Assessment
(summarise evaluation scores if provided)

# Limitations
(data quality issues, model assumptions, caveats)

# Recommendations
(3-5 actionable next steps)

Use markdown. Be precise and quantitative. Reference actual numbers from the outputs."""


def _call_ollama(prompt: str, system: str = REPORTER_SYSTEM) -> str:
    """Call LLM for report generation with OpenCode Zen / Ollama fallback."""
    from multimodal_ds.core.llm_client import chat_with_fallback
    try:
        return chat_with_fallback(
            primary_model=REVIEWER_MODEL,
            fallback_model="ollama/qwen2.5:7b",
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=4000,
            temperature=0.2,
        )
    except Exception as e:
        logger.error(f"[Reporter] LLM call failed: {e}")
        return _fallback_report(prompt)


def _fallback_report(context: str) -> str:
    """Generate a minimal structured report when LLM is unavailable."""
    return f"""# Analysis Report

## Executive Summary
Analysis pipeline completed. LLM reporter was unavailable for narrative generation.
Raw outputs are included below for review.

## Raw Pipeline Outputs
```
{context[:3000]}
```

## Recommendations
- Review the raw code outputs above for key findings
- Re-run with the reporter LLM available for a structured narrative
"""


def reporter_agent(state: AgentState) -> AgentState:
    """
    LangGraph node: Generate the final report and save to disk.

    Reads:
      state['code_outputs'], state['visualizations'], state['errors'],
      state['analysis_plan'], state['eval_report'], state['user_query']

    Writes:
      state['final_report']  — markdown string
      {session_dir}/final_report.md  — on disk
      {session_dir}/eval_report.json — on disk
    """
    session_id = state.get("session_id", "default")
    logger.info(f"[Reporter] Generating final report for session {session_id}")

    # Assemble context for the LLM – clearer sections with markdown formatting
    all_outputs = "\n\n".join(state.get("code_outputs", []))
    charts = "\n".join(state.get("visualizations", []))
    artifacts = "\n".join(state.get("saved_artifacts", []))
    errors = "\n".join(state.get("errors", []))
    eval_report = state.get("eval_report", {})
    if not isinstance(eval_report, dict):
        eval_report = {
            "session_verdict": getattr(eval_report, "session_verdict", "UNKNOWN"),
            "overall_session_score": getattr(eval_report, "overall_session_score", "N/A"),
            "evaluations": getattr(eval_report, "evaluations", []),
        }
    plan = state.get("analysis_plan", "No plan recorded.")
    query = state.get("user_query", "")

    # Build evaluation markdown table
    eval_summary = ""
    if eval_report:
        verdict = eval_report.get("session_verdict", "UNKNOWN")
        score = eval_report.get("overall_session_score", "N/A")
        eval_summary = f"**Overall Quality Score:** {score}/10  \
**Verdict:** {verdict}\n\n| Task | Score | Flagged |\n|------|-------|--------|"
        for ev in eval_report.get("evaluations", [])[:5]:
            task_name = ev.get("task_name", "?")
            task_score = ev.get("overall_score", "?")
            flagged = "Yes" if ev.get("flagged") else "No"
            eval_summary += f"\n| {task_name} | {task_score} | {flagged} |"

    prompt = f"""Original query: {query}\n\nAnalysis plan executed:\n{plan}\n\nStep‑by‑step outputs (truncated to first 6000 characters):\n{all_outputs[:6000]}\n\nCharts generated (filenames):\n{charts or 'None'}\n\nArtifacts saved (filenames):\n{artifacts or 'None'}\n\nEvaluation summary (markdown table):\n{eval_summary or 'No evaluation data'}\n\nErrors encountered:\n{errors or 'None'}\n\nWrite a concise, fully structured markdown report following the prescribed sections. Include chart references using `![](filename)` syntax where appropriate. Use bullet points and tables for clarity.\n"""


    prompt = f"""Original query: {query}

Analysis plan executed:
{plan}

Step-by-step outputs:
{all_outputs[:6000]}

Charts generated:
{charts or 'None'}

Production Artifacts Saved:
{artifacts or 'None'}

Evaluation summary:
{eval_summary or 'No evaluation data'}

Errors encountered:
{errors or 'None'}

Write the complete analysis report now. Be sure to mention any saved models or data files in the Results or Recommendations section."""

    report = _call_ollama(prompt)

    # ── Persist to disk ───────────────────────────────────────────────────
    session_dir = Path(OUTPUT_DIR) / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    report_path = session_dir / "final_report.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info(f"[Reporter] Report saved → {report_path}")

    # Persist eval_report as JSON
    if eval_report:
        eval_path = session_dir / "eval_report.json"
        eval_path.write_text(json.dumps(eval_report, indent=2), encoding="utf-8")
        logger.info(f"[Reporter] Eval report saved → {eval_path}")

    return {"final_report": report}


# ── Standalone entry point (for direct calls from orchestrator) ────────────

def generate_report(
    user_query: str,
    analysis_plan: str,
    code_outputs: list[str],
    visualizations: list[str],
    errors: list[str],
    eval_report: dict,
    session_id: str,
    working_dir: Optional[str] = None,
) -> str:
    """
    Callable from AgentOrchestrator without going through the graph.
    Returns the markdown report string and saves it to disk.
    """
    state: AgentState = {
        "user_query": user_query,
        "uploaded_files": [],
        "_routing_flags": {},
        "parsed_documents": [],
        "image_embeddings": [],
        "audio_transcripts": [],
        "tabular_summaries": [],
        "statistical_report": {},
        "analysis_plan": analysis_plan,
        "analysis_tasks": [],
        "hypotheses": [],
        "current_step": 0,
        "steps_total": 0,
        "code_outputs": code_outputs,
        "visualizations": visualizations,
        "errors": errors,
        "retry_count": 0,
        "vector_store_id": "",
        "retrieved_context": "",
        "eval_report": eval_report,
        "final_report": "",
        "session_id": session_id,
        "messages": [],
    }

    result_state = reporter_agent(state)
    return result_state["final_report"]
