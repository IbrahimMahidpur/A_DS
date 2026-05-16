"""
CLI for the Multimodal Agentic DS Engine.

Usage:
    mmads serve              — Start the FastAPI server
    mmads ingest <file>      — Ingest a single file and print summary
    mmads run <file> <goal>  — Full analysis pipeline
    mmads memory <session>   — Show session memory
"""
import json
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

app = typer.Typer(
    name="mmads",
    help="Multimodal Agentic Data Science Engine — 100% local Ollama",
    add_completion=False,
)
console = Console()
logger = logging.getLogger(__name__)



@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind host"),
    port: int = typer.Option(8000, help="Port"),
    reload: bool = typer.Option(False, help="Enable hot-reload (dev mode)"),
    log_level: str = typer.Option("info", help="Uvicorn log level"),
):
    """Start the FastAPI server."""
    import uvicorn
    console.print(Panel(f"[bold green]MMADS API[/] -> http://{host}:{port}/docs", title="Starting"))
    uvicorn.run(
        "multimodal_ds.api.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
    )


@app.command()
def ingest(
    file: Path = typer.Argument(..., help="Path to file to ingest"),
    output_json: bool = typer.Option(False, "--json", help="Output raw JSON"),
):
    """Ingest a single file and display the result."""
    from multimodal_ds.ingestion.router import route_and_ingest

    if not file.exists():
        console.print(f"[red]File not found:[/] {file}")
        raise typer.Exit(1)

    with console.status(f"Ingesting [cyan]{file.name}[/]..."):
        doc = route_and_ingest(str(file))

    if output_json:
        print(json.dumps(doc.to_dict(), indent=2))
        return

    table = Table(title=f"Ingested: {file.name}", show_header=True)
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("ID", doc.id)
    table.add_row("Type", doc.data_type.value)
    table.add_row("Status", f"[green]{doc.status.value}[/]" if doc.status.value == "done" else f"[red]{doc.status.value}[/]")
    table.add_row("Processor", doc.provenance.processor)
    table.add_row("Time (s)", str(doc.provenance.processing_time_s))
    table.add_row("Text preview", doc.text_content[:200] + "..." if doc.text_content else "—")
    if doc.schema_info:
        table.add_row("Shape", str(doc.schema_info.get("shape", "—")))
        table.add_row("Columns", str(len(doc.schema_info.get("columns", []))))
    console.print(table)


@app.command()
def run(
    files: list[Path] = typer.Argument(..., help="One or more files to analyse"),
    objective: str = typer.Option(..., "--objective", "-o", help="Analysis goal"),
    session_id: Optional[str] = typer.Option(None, "--session", help="Session ID / Thread ID"),
    max_tasks: int = typer.Option(6, "--max-tasks", help="Max tasks to execute"),
    no_stats: bool = typer.Option(False, "--no-stats", help="Skip statistical checks (legacy)"),
    output_json: bool = typer.Option(False, "--json", help="Output raw JSON"),
):
    """Run the full agentic analysis pipeline using the LangGraph production engine."""
    from multimodal_ds.graph import build_graph, make_initial_state
    import uuid

    missing = [f for f in files if not f.exists()]
    if missing:
        console.print(f"[red]Files not found:[/] {missing}")
        raise typer.Exit(1)

    # Use provided session_id or generate a short one for this thread
    thread_id = session_id or str(uuid.uuid4())[:8]
    
    console.print(Panel(
        f"[bold]Objective:[/] {objective}\n"
        f"[bold]Files:[/] {[f.name for f in files]}\n"
        f"[bold]Thread ID:[/] {thread_id}", 
        title="[green]MMADS LangGraph Run[/]"
    ))

    # Build graph and run
    graph = build_graph()
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = make_initial_state(
        user_query=objective,
        uploaded_files=[str(f) for f in files],
        session_id=thread_id
    )

    with console.status(f"Executing graph (Thread: {thread_id})...") as status:
        try:
            final_state = initial_state
            # Use stream() to show progress through the graph nodes
            for event in graph.stream(initial_state, config=config, stream_mode="updates"):
                for node_name, state_update in event.items():
                    console.print(f"[blue]» Finished node:[/] [bold]{node_name}[/]")
                    # Only update scalar fields — list fields are managed by LangGraph
                    for k, v in state_update.items():
                        if not isinstance(v, list):
                            final_state[k] = v
                        else:
                            # For list fields, extend rather than replace to avoid duplication
                            if k not in final_state:
                                final_state[k] = v
                            else:
                                existing = final_state.get(k, [])
                                new_items = [x for x in v if x not in existing]
                                final_state[k] = existing + new_items

                    # Provide specific updates based on node
                    if node_name == "planner":
                        n_tasks = len(state_update.get("analysis_tasks", []))
                        if n_tasks:
                            console.print(f"   [green]✔[/] Created plan with {n_tasks} tasks.")
                    elif node_name == "executor":
                        step = state_update.get("current_step", 0)
                        if step:
                            console.print(f"   [green]✔[/] Executed task step {step}.")
                    elif node_name == "stats_val":
                        console.print(f"   [green]✔[/] Statistical validation complete.")

                    status.update(f"Executing: {node_name}...")
        except Exception as e:
            console.print(f"[bold red]Graph execution failed:[/] {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise typer.Exit(1)

    if output_json:
        # We filter the state for JSON output to avoid huge data dumps
        printable_state = {k: v for k, v in final_state.items() if k not in ["image_embeddings", "messages"]}
        print(json.dumps(printable_state, indent=2, default=str))
        return

    # ── Summary Output ─────────────────────────────────────────────────────
    status = "success" if not final_state.get("errors") else "partial"
    status_color = "green" if status == "success" else "yellow"
    
    tasks_done = len(final_state.get("code_outputs", []))
    files_count = len(final_state.get("visualizations", []))

    console.print(Panel(
        f"Status: [{status_color}]{status.upper()}[/]\n"
        f"Tasks Completed: {tasks_done}\n"
        f"Files Created: {files_count}\n"
        f"Session Dir: agentic_output/{thread_id}",
        title="Execution Summary"
    ))

    if final_state.get("final_report"):
        # Show first 1000 chars of the executive summary
        report_preview = final_state["final_report"]
        if len(report_preview) > 1000:
            report_preview = report_preview[:1000] + "\n\n... [report continues on disk] ..."
        
        console.print(Panel(report_preview, title="Executive Summary (Preview)"))

    if final_state.get("errors"):
        console.print(Panel("\n".join(final_state["errors"]), title="[red]Warnings/Errors[/]"))

    console.print(f"\n[bold green]✓ Done![/] Full report saved to: [cyan]agentic_output/{thread_id}/final_report.md[/]")


@app.command()
def memory(
    session_id: str = typer.Argument(..., help="Session ID to inspect"),
    n: int = typer.Option(10, "--n", help="Number of entries to show"),
):
    """Display stored memory entries for a session."""
    from multimodal_ds.memory.agent_memory import AgentMemory

    mem = AgentMemory()
    entries = mem.get_session_history(session_id)[:n]

    if not entries:
        console.print(f"[yellow]No memory found for session:[/] {session_id}")
        return

    for i, entry in enumerate(entries, 1):
        meta = entry.get("metadata", {})
        console.print(Panel(
            entry["content"][:400],
            title=f"[{i}] step={meta.get('step', '?')} | {meta.get('timestamp', '')}",
        ))


if __name__ == "__main__":
    app()
