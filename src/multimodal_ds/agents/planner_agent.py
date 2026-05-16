"""
Hypothesis Generation + Planning Agent — LangGraph + ReAct reasoning.
Decomposes user objectives into analysis task sequences.
Uses Ollama for all LLM calls — no API keys needed.
"""
import json
import logging
import re
from typing import Any, TypedDict, Annotated
# ReAct‑style web‑search detection utilities

def _needs_web_search(objective: str) -> bool:
    """Detect if the objective references unknown data sources or external info.
    Simple heuristic based on keyword presence.
    """
    keywords = [
        r"search",
        r"lookup",
        r"reference",
        r"external",
        r"find",
        r"url",
        r"website",
        r"online",
        r"web",
    ]
    pattern = re.compile(r"|".join(keywords), re.IGNORECASE)
    return bool(pattern.search(objective))

def _perform_web_search(objective: str) -> str:
    """Placeholder for actual WebSearch tool call.
    Returns an empty string; in production this would invoke the Claude Code
    WebSearch tool and return relevant snippets.
    """
    # TODO: integrate with the WebSearch tool when running under Claude Code.
    return ""

import operator

from multimodal_ds.config import PLANNER_MODEL, OLLAMA_BASE_URL, LLM_TIMEOUT
from multimodal_ds.memory.agent_memory import AgentMemory
from multimodal_ds.core.schema import UnifiedDocument

logger = logging.getLogger(__name__)


def _extract_json(text: str) -> str:
    """
    Robustly extract a JSON array or object from LLM output.
    Handles:
      - <think>...</think> reasoning blocks (qwen3, deepseek)
      - Markdown fences  (```json ... ```)
      - Leading/trailing prose around the JSON
      - Trailing commas before } or ]
    """
    # 1. Strip <think> reasoning blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    # 2. Try to extract from markdown code fence
    fence_match = re.search(r'```(?:json)?\s*([\[{].*?[\]}])\s*```', text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)

    # 3. Find the outermost JSON array [...] or object {...}
    for start_char, end_char in [('[', ']'), ('{', '}')]:
        start = text.find(start_char)
        if start == -1:
            continue
        depth = 0
        end = -1
        for i, ch in enumerate(text[start:], start):
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end != -1:
            candidate = text[start:end]
            # 4. Fix trailing commas (common LLM mistake)
            candidate = re.sub(r',\s*([\]}])', r'\1', candidate)
            return candidate

    return text  # return as-is; caller will handle parse error


class PlannerState(TypedDict):
    """LangGraph state for the planner agent."""
    session_id: str
    user_objective: str
    data_profiles: list[dict]          # From ingested documents
    analysis_plan: list[dict]          # Generated task sequence
    current_step: int
    messages: Annotated[list, operator.add]
    hypotheses: list[str]
    final_plan: str
    error: str


def _call_ollama(prompt: str, system: str = "", max_tokens: int = 4000) -> str:
    """Call Ollama with a prompt and return response text."""
    import httpx
    model = PLANNER_MODEL.replace("ollama/", "")
    try:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = httpx.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {"num_predict": max_tokens, "temperature": 0.3},
            },
            timeout=LLM_TIMEOUT,
        )
        if response.status_code == 200:
            return response.json().get("message", {}).get("content", "")
        return f"[Error: HTTP {response.status_code}]"
    except Exception as e:
        return f"[Error: {e}]"


def generate_hypotheses(state: PlannerState) -> PlannerState:
    """Node: Generate initial hypotheses from data profiles."""
    # Improve profile preparation — take the most relevant parts and truncate cleanly
    profiles_text = json.dumps(state["data_profiles"], indent=2)
    if len(profiles_text) > 8000:
        profiles_text = profiles_text[:8000] + "\n... [truncated for length] ..."

    prompt = f"""You are an expert data scientist. Given this data profile and user objective, generate 3-5 specific, testable hypotheses.

User Objective: {state['user_objective']}

Data Profile:
{profiles_text}

Generate hypotheses as a JSON array. Each hypothesis should have:
- "id": short identifier
- "statement": the hypothesis
- "analysis_method": how to test it
- "expected_outcome": what success looks like

Respond ONLY with valid JSON array, no other text."""

    response = _call_ollama(prompt, system="You are a data science hypothesis generator. Output only valid JSON.")
    
    try:
        cleaned = _extract_json(response)
        hypotheses = json.loads(cleaned)
        state["hypotheses"] = [h.get("statement", str(h)) for h in hypotheses]
        logger.info(f"[Planner] Generated {len(state['hypotheses'])} hypotheses")
    except Exception as e:
        logger.warning(f"[Planner] Hypothesis JSON parse failed: {e}")
        logger.debug(f"[Planner] Raw LLM response (first 500): {response[:500]!r}")
        # Fallback: treat entire response as a single hypothesis
        state["hypotheses"] = [response[:500]] if response.strip() else [state["user_objective"]]

    return state


def decompose_into_tasks(state: PlannerState) -> PlannerState:
    """Node: Decompose objective into ordered analysis tasks."""
    hypotheses_text = "\n".join(f"- {h}" for h in state.get("hypotheses", []))
    profiles_text = json.dumps(state["data_profiles"], indent=2)
    if len(profiles_text) > 8000:
        profiles_text = profiles_text[:8000] + "\n... [truncated for length] ..."

    # If the objective hints at external references, perform a (placeholder) web search and include results
    web_results = ""
    if _needs_web_search(state["user_objective"]):
        web_results = _perform_web_search(state["user_objective"])
        state["web_results"] = web_results

    prompt = f"""You are a senior data scientist creating an analysis plan.

STRICT RULE: Use ONLY the exact column names found in the 'Data available' section below.
Do NOT hallucinate or guess column names.

Objective: {state['user_objective']}

{('Web context:\n' + web_results) if web_results else ''}

Hypotheses to test:
{hypotheses_text}

Data available:
{profiles_text}

Create a detailed analysis plan as a JSON array of tasks. Each task:
{{
  "step": 1,
  "name": "task name",
  "type": "eda|feature_engineering|modeling|evaluation|visualization|reporting",
  "description": "what to do (STRICTLY use column names from profile)",
  "tools": ["pandas", "sklearn", "plotly"],
  "expected_output": "what this step produces (e.g. 'Saved model to model.pkl')",
  "depends_on": []
}}

IMPORTANT: If the objective involves prediction, explicitly include a task to 'Save the trained model as a .pkl or .joblib file'.

Include these task types in order: EDA → Feature Engineering →
     Model Selection → Evaluation → Visualization → Report.

     EFFICIENCY RULE: Maximum 6 tasks. Each task must be substantial and
     self-contained. Combine related operations into one task where logical:
     - Combine EDA + correlation analysis into one task
     - Combine model training + hyperparameter tuning into one task
     - Combine evaluation metrics + visualization into one task
     Never create a task for a single operation that takes under 10 lines
     of code. Quality over quantity.
Respond ONLY with valid JSON array."""

    response = _call_ollama(prompt, system="You are a data science task planner. Output only valid JSON.")

    try:
        cleaned = _extract_json(response)
        tasks = json.loads(cleaned)
        state["analysis_plan"] = tasks
        state["current_step"] = 0
        logger.info(f"[Planner] Created plan with {len(tasks)} tasks")
    except Exception as e:
        logger.warning(f"[Planner] Task decomposition JSON parse failed: {e}")
        logger.debug(f"[Planner] Raw LLM response (first 500): {response[:500]!r}")
        # Use a sensible default plan so execution can still proceed
        state["analysis_plan"] = _default_plan(state.get("data_profiles", []))

    return state


def create_final_plan(state: PlannerState) -> PlannerState:
    """Node: Synthesize final human-readable plan."""
    tasks_text = json.dumps(state.get("analysis_plan", []), indent=2)[:3000]

    prompt = f"""Create a clear, actionable analysis plan summary.

Objective: {state['user_objective']}
Tasks: {tasks_text}

Write a 200-word executive summary of the analysis approach, what will be done, and what insights are expected."""

    state["final_plan"] = _call_ollama(prompt)
    return state


def store_plan_to_memory(state: PlannerState) -> PlannerState:
    """Node: Persist plan to ChromaDB memory."""
    memory = AgentMemory()
    memory.store(
        content=f"Analysis Plan for: {state['user_objective']}\n\n{state['final_plan']}",
        metadata={"type": "analysis_plan", "session_id": state["session_id"]},
        doc_id=f"plan_{state['session_id']}"
    )
    for i, step in enumerate(state.get("analysis_plan", [])):
        memory.store(
            content=json.dumps(step),
            metadata={"type": "task", "step": str(i), "session_id": state["session_id"]}
        )
    return state


def build_planner_graph():
    """Build and compile the LangGraph planner workflow."""
    try:
        from langgraph.graph import StateGraph, END

        graph = StateGraph(PlannerState)
        graph.add_node("generate_hypotheses", generate_hypotheses)
        graph.add_node("decompose_tasks", decompose_into_tasks)
        graph.add_node("create_final_plan", create_final_plan)
        graph.add_node("store_to_memory", store_plan_to_memory)

        graph.set_entry_point("generate_hypotheses")
        graph.add_edge("generate_hypotheses", "decompose_tasks")
        graph.add_edge("decompose_tasks", "create_final_plan")
        graph.add_edge("create_final_plan", "store_to_memory")
        graph.add_edge("store_to_memory", END)

        return graph.compile()

    except ImportError:
        logger.warning("[Planner] langgraph not installed — using simple sequential planner")
        return None


def run_planner(
    user_objective: str,
    documents: list[UnifiedDocument],
    session_id: str = "default"
) -> dict:
    """
    Main entry point for the planning agent.
    Returns the complete analysis plan.
    """
    data_profiles = [doc.to_dict() for doc in documents]

    initial_state = PlannerState(
        session_id=session_id,
        user_objective=user_objective,
        data_profiles=data_profiles,
        analysis_plan=[],
        current_step=0,
        messages=[],
        hypotheses=[],
        final_plan="",
        error=""
    )

    graph = build_planner_graph()
    if graph:
        try:
            result = graph.invoke(initial_state)
            return result
        except Exception as e:
            logger.error(f"[Planner] Graph execution failed: {e}")

    # Fallback: run nodes sequentially
    state = initial_state
    state = generate_hypotheses(state)
    state = decompose_into_tasks(state)
    state = create_final_plan(state)
    state = store_plan_to_memory(state)
    return state


def _default_plan(data_profiles: list = None) -> list[dict]:
    """Default analysis plan when LLM fails. Uses data profile to contextualize tasks."""
    cols_hint = ""
    if data_profiles:
        try:
            cols = data_profiles[0].get("schema_info", {}).get("columns", [])
            if cols:
                cols_hint = f" Columns available: {', '.join(str(c) for c in cols[:15])}"
        except Exception:
            pass

    return [
        {"step": 1, "name": "EDA", "type": "eda", "description": f"Exploratory data analysis: distributions, missing values, correlations.{cols_hint}", "tools": ["pandas", "matplotlib", "seaborn"], "expected_output": "Statistical summary and visualizations", "depends_on": []},
        {"step": 2, "name": "Feature Engineering", "type": "feature_engineering", "description": "Encode categoricals, handle missing values, engineer useful features.", "tools": ["pandas", "sklearn"], "expected_output": "Feature matrix ready for modeling", "depends_on": [1]},
        {"step": 3, "name": "Model Training", "type": "modeling", "description": "Train a classification or regression model and tune hyperparameters.", "tools": ["sklearn"], "expected_output": "Trained model with performance metrics", "depends_on": [2]},
        {"step": 4, "name": "Evaluation", "type": "evaluation", "description": "Evaluate model with appropriate metrics (ROC-AUC, F1, RMSE, etc.).", "tools": ["sklearn"], "expected_output": "Evaluation report with confusion matrix / error analysis", "depends_on": [3]},
        {"step": 5, "name": "Visualization", "type": "visualization", "description": "Generate feature importance plot, ROC curve, and key insight charts.", "tools": ["plotly", "matplotlib"], "expected_output": "Interactive insight charts", "depends_on": [4]},
    ]
