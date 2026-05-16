"""
Evaluation Agent — LLM-as-judge for code execution outputs.
Specialist agent #5: critiques each task result across 4 dimensions.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx

from multimodal_ds.config import REVIEWER_MODEL, OLLAMA_BASE_URL, LLM_TIMEOUT, OUTPUT_DIR
from multimodal_ds.memory.agent_memory import AgentMemory

logger = logging.getLogger(__name__)

# ── Thresholds ─────────────────────────────────────────────────────────────
FLAG_DIMENSION_THRESHOLD = 4
FLAG_OVERALL_THRESHOLD = 6
FALLBACK_SCORE = 5

# ── Score dataclasses ─────────────────────────────────────────────────────────────

@dataclass
class DimensionScore:
    name: str
    score: int
    reasoning: str
    flagged: bool = False

    def to_dict(self) -> dict:
        return {"name": self.name, "score": self.score, "reasoning": self.reasoning, "flagged": self.flagged}

@dataclass
class TaskEvaluation:
    task_name: str
    overall_score: int
    flagged: bool
    flag_reasons: list[str] = field(default_factory=list)
    dimensions: list[DimensionScore] = field(default_factory=list)
    recommendation: str = ""
    llm_available: bool = True
    evaluated_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    @property
    def statistical_validity(self) -> Optional[int]:
        return self._score_by_name("statistical_validity")

    @property
    def hallucination_risk(self) -> Optional[int]:
        return self._score_by_name("hallucination_risk")

    @property
    def data_leakage(self) -> Optional[int]:
        return self._score_by_name("data_leakage")

    @property
    def output_completeness(self) -> Optional[int]:
        return self._score_by_name("output_completeness")

    def _score_by_name(self, name: str) -> Optional[int]:
        for d in self.dimensions:
            if d.name == name:
                return d.score
        return None

    def to_dict(self) -> dict:
        return {
            "task_name": self.task_name,
            "overall_score": self.overall_score,
            "flagged": self.flagged,
            "flag_reasons": self.flag_reasons,
            "dimensions": [d.to_dict() for d in self.dimensions],
            "recommendation": self.recommendation,
            "llm_available": self.llm_available,
            "evaluated_at": self.evaluated_at,
        }

@dataclass
class EvalReport:
    session_id: str
    task_count: int
    flagged_count: int
    pass_count: int
    overall_session_score: float
    evaluations: list[TaskEvaluation] = field(default_factory=list)
    session_verdict: str = ""

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "task_count": self.task_count,
            "flagged_count": self.flagged_count,
            "pass_count": self.pass_count,
            "overall_session_score": round(self.overall_session_score, 2),
            "session_verdict": self.session_verdict,
            "evaluations": [e.to_dict() for e in self.evaluations],
        }

    def save(self, output_dir: Path) -> Path:
        path = output_dir / "eval_report.json"
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

class EvaluationAgent:
    AGENT_NAME = "evaluation_agent"
    WEIGHTS = {
        "statistical_validity": 0.35,
        "hallucination_risk": 0.30,
        "data_leakage": 0.20,
        "output_completeness": 0.15,
    }
    JUDGE_SYSTEM_PROMPT = (
        "You are a senior data science reviewer and LLM judge. "
        "Your role is to critically evaluate Python code execution outputs "
        "for statistical correctness, factual accuracy, and safety. "
        "You must respond ONLY with valid JSON — no preamble, no markdown fences, "
        "no explanation outside the JSON structure."
    )

    def __init__(self, session_id: str = "default", working_dir: Optional[str] = None,
                 flag_dimension_threshold: int = FLAG_DIMENSION_THRESHOLD,
                 flag_overall_threshold: int = FLAG_OVERALL_THRESHOLD):
        self.session_id = session_id
        self.working_dir = Path(working_dir or OUTPUT_DIR) / session_id
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.flag_dimension_threshold = flag_dimension_threshold
        self.flag_overall_threshold = flag_overall_threshold
        self.memory = AgentMemory()

    # ── Bus helpers ────────────────────────────────────────────────────────
    def _publish_eval_request(self) -> None:
        try:
            from multimodal_ds.core.message_bus import AgentMessage, MessageType, get_bus
            get_bus().publish(AgentMessage(
                msg_type=MessageType.EVAL_REQUEST,
                payload={"session_id": self.session_id},
                sender="evaluation_agent",
                session_id=self.session_id,
            ))
        except Exception:
            pass

    def _publish_eval_complete(self, task_name: str, score: int) -> None:
        try:
            from multimodal_ds.core.message_bus import AgentMessage, MessageType, get_bus
            get_bus().publish(AgentMessage(
                msg_type=MessageType.EVAL_COMPLETE,
                payload={"session_id": self.session_id, "task_name": task_name, "score": score},
                sender="evaluation_agent",
                session_id=self.session_id,
            ))
        except Exception:
            pass

    def _publish_eval_flagged(self, task_name: str, reasons: list) -> None:
        try:
            from multimodal_ds.core.message_bus import AgentMessage, MessageType, Priority, get_bus
            get_bus().publish(AgentMessage(
                msg_type=MessageType.EVAL_FLAGGED,
                payload={"session_id": self.session_id, "task_name": task_name, "flag_reasons": reasons},
                sender="evaluation_agent",
                session_id=self.session_id,
                priority=Priority.HIGH,
            ))
        except Exception:
            pass

    # ── Main entry points ──────────────────────────────────────────────────
    def evaluate_task_results(
            self,
            task_results: list[dict],
            data_context: str = "",
            stat_report: Optional[dict] = None,
        ) -> EvalReport:
        t_start = time.time()
        logger.info(
            f"[EvalAgent] Evaluating {len(task_results)} task(s) for session {self.session_id}"
        )
        self._publish_eval_request()
        evaluations: list[TaskEvaluation] = []
        successful_tasks = [t for t in task_results if t.get("success")]
        import concurrent.futures

        def _eval_one(task_result):
            return self.evaluate_task(
                task_result=task_result,
                data_context=data_context,
                stat_report=stat_report,
            )

        # Run all judge calls simultaneously — each hits Ollama independently
        # Max 3 workers to avoid overwhelming local Ollama
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(_eval_one, t): t
                for t in successful_tasks
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    eval_result = future.result(timeout=180)
                    evaluations.append(eval_result)
                    if eval_result.flagged:
                        self._publish_eval_flagged(
                            eval_result.task_name, eval_result.flag_reasons
                        )
                    else:
                        self._publish_eval_complete(
                            eval_result.task_name, eval_result.overall_score
                        )
                    logger.info(
                        f"[EvalAgent] Task '{eval_result.task_name}' — "
                        f"score={eval_result.overall_score}/10, "
                        f"flagged={eval_result.flagged}"
                    )
                except Exception as e:
                    logger.warning(f"[EvalAgent] Parallel eval failed: {e}")
        report = self._build_report(evaluations)
        report.save(self.working_dir)
        self._publish_eval_complete("session", int(report.overall_session_score))
        self.memory.store_analysis_step(
            step_name="evaluation",
            result=(
                f"Session verdict: {report.session_verdict} | "
                f"Score: {report.overall_session_score:.1f}/10 | "
                f"Flagged: {report.flagged_count}/{report.task_count} tasks"
            ),
            session_id=self.session_id,
        )
        duration = round(time.time() - t_start, 2)
        logger.info(
            f"[EvalAgent] Session {self.session_id} complete — verdict={report.session_verdict}, "
            f"score={report.overall_session_score:.1f}, duration={duration}s"
        )
        return report

    def evaluate_task(
            self,
            task_result: dict,
            data_context: str = "",
            stat_report: Optional[dict] = None,
        ) -> TaskEvaluation:
        task_name = task_result.get("name", "unknown_task")
        output = task_result.get("output_preview", "")
        files = task_result.get("files_created", [])
        logger.info(f"[EvalAgent] Evaluating task: {task_name}")
        raw_scores = self._call_judge(
            task_name=task_name,
            output=output,
            files_created=files,
            data_context=data_context,
            stat_report=stat_report,
        )
        return self._build_task_evaluation(task_name, raw_scores, files)

    # ── LLM judge ─────────────────────────────────────────────────────────
    def _call_judge(
            self,
            task_name: str,
            output: str,
            files_created: list[str],
            data_context: str,
            stat_report: Optional[dict],
        ) -> dict:
        stat_context = ""
        if stat_report:
            non_normal = [
                k for k, v in stat_report.get("normality", {}).items()
                if isinstance(v, dict) and not v.get("is_normal", True)
            ]
            mc_detected = stat_report.get("multicollinearity", {}).get("multicollinearity_detected", False)
            stat_context = (
                f"Statistical context: non-normal columns={non_normal}, "
                f"multicollinearity_detected={mc_detected}"
            )
        prompt = f"""Evaluate this Python data science task execution result.

Task name: {task_name}
Files created: {files_created}
Statistical context: {stat_context}

Data context (first 800 chars):
{data_context[:800]}

Task output (first 1200 chars):
{output[:1200]}

Score each dimension from 0 to 10 and provide one-sentence reasoning per dimension.

Respond with ONLY this JSON structure (no markdown, no explanation):
{{
  "statistical_validity":  {{"score": <int 0-10>, "reasoning": "<one sentence>"}},
  "hallucination_risk":    {{"score": <int 0-10>, "reasoning": "<one sentence>"}},
  "data_leakage":          {{"score": <int 0-10>, "reasoning": "<one sentence>"}},
  "output_completeness":   {{"score": <int 0-10>, "reasoning": "<one sentence>"}},
  "recommendation":        "<one actionable sentence for the next step>"
}}"""
        model = REVIEWER_MODEL.replace("ollama/", "")
        try:
            response = httpx.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": self.JUDGE_SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                    "stream": False,
                    "options": {"num_predict": 1200, "temperature": 0.1},
                },
                timeout=LLM_TIMEOUT,
            )
            if response.status_code == 200:
                content = response.json().get("message", {}).get("content", "").strip()
                return self._parse_judge_response(content)
        except Exception as e:
            logger.warning(f"[EvalAgent] LLM judge call failed: {e}")
        return self._fallback_scores(files_created)

    def _parse_judge_response(self, content: str) -> dict:
        import re
        cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        fence_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', cleaned, re.DOTALL)
        if fence_match:
            cleaned = fence_match.group(1)
        start = cleaned.find("{")
        if start != -1:
            depth = 0
            end = -1
            for i, ch in enumerate(cleaned[start:], start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end != -1:
                cleaned = cleaned[start:end]
        cleaned = re.sub(r',\s*([\]}])', r'\1', cleaned)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"Cannot parse judge response: {content[:200]}") from e
        required = {"statistical_validity", "hallucination_risk", "data_leakage", "output_completeness"}
        missing = required - set(parsed.keys())
        if missing:
            raise ValueError(f"Judge response missing keys: {missing}")
        return parsed

    def _fallback_scores(self, files_created: list[str]) -> dict:
        completeness = 8 if files_created else 3
        return {
            "statistical_validity": {"score": FALLBACK_SCORE, "reasoning": "LLM judge unavailable — fallback score applied."},
            "hallucination_risk":   {"score": FALLBACK_SCORE, "reasoning": "LLM judge unavailable — fallback score applied."},
            "data_leakage":         {"score": FALLBACK_SCORE, "reasoning": "LLM judge unavailable — fallback score applied."},
            "output_completeness": {"score": completeness, "reasoning": f"{'Files produced: ' + str(files_created) if files_created else 'No files created.'}"},
            "recommendation": "Re-run evaluation with LLM available for accurate scoring.",
            "_fallback": True,
        }

    # ── Score assembly ─────────────────────────────────────────────────────
    def _build_task_evaluation(self, task_name: str, raw_scores: dict, files: list[str]) -> TaskEvaluation:
        llm_available = not raw_scores.pop("_fallback", False)
        recommendation = raw_scores.pop("recommendation", "")
        dimensions: list[DimensionScore] = []
        flag_reasons: list[str] = []
        weighted_sum = 0.0
        weight_total = 0.0
        for dim_name, weight in self.WEIGHTS.items():
            dim_data = raw_scores.get(dim_name, {})
            score = int(dim_data.get("score", FALLBACK_SCORE))
            score = max(0, min(10, score))
            reasoning = str(dim_data.get("reasoning", ""))
            flagged = score < self.flag_dimension_threshold
            if flagged:
                flag_reasons.append(f"{dim_name} scored {score}/10: {reasoning}")
            dimensions.append(DimensionScore(name=dim_name, score=score, reasoning=reasoning, flagged=flagged))
            weighted_sum += score * weight
            weight_total += weight
        overall = round(weighted_sum / weight_total) if weight_total > 0 else FALLBACK_SCORE
        overall = max(0, min(10, overall))
        if not llm_available:
            overall = FALLBACK_SCORE
        else:
            file_type_score = 0.0
            for f in files:
                name = f.lower()
                if name.endswith((".png", ".jpg", ".jpeg", ".webp", ".html")):
                    file_type_score += 0.5
                elif name.endswith((".pkl", ".joblib", ".pt", ".h5")):
                    file_type_score += 0.3
                elif name.endswith((".json", ".txt", ".csv")):
                    file_type_score += 0.2
            file_type_score = min(file_type_score, 2.0)
            overall = min(10, overall + file_type_score)
            for dim in dimensions:
                if dim.name == "output_completeness":
                    boost = int(round((file_type_score / 2) * 2))
                    dim.score = min(10, dim.score + boost)
                    dim.reasoning += f" (artifact weighting bonus: +{boost})"
                    break
        if not llm_available:
            is_flagged = bool(flag_reasons)
        else:
            is_flagged = bool(flag_reasons) or overall < self.flag_overall_threshold
        if is_flagged and overall < self.flag_overall_threshold:
            flag_reasons.insert(0, f"Overall score {overall}/10 below threshold {self.flag_overall_threshold}")
        return TaskEvaluation(
            task_name=task_name,
            overall_score=overall,
            flagged=is_flagged,
            flag_reasons=flag_reasons,
            dimensions=dimensions,
            recommendation=recommendation,
            llm_available=llm_available,
        )

    # ── Report building ─────────────────────────────────────────────────────
    def _build_report(self, evaluations: list[TaskEvaluation]) -> EvalReport:
        if not evaluations:
            return EvalReport(
                session_id=self.session_id,
                task_count=0,
                flagged_count=0,
                pass_count=0,
                overall_session_score=0.0,
                session_verdict="PASS",
            )
        flagged = [e for e in evaluations if e.flagged]
        passed = [e for e in evaluations if not e.flagged]
        mean_score = sum(e.overall_score for e in evaluations) / len(evaluations)
        flag_ratio = len(flagged) / len(evaluations)
        if flag_ratio == 0 and mean_score >= 7:
            verdict = "PASS"
        elif flag_ratio <= 0.3 or mean_score >= 5:
            verdict = "WARN"
        else:
            verdict = "FAIL"
        return EvalReport(
            session_id=self.session_id,
            task_count=len(evaluations),
            flagged_count=len(flagged),
            pass_count=len(passed),
            overall_session_score=mean_score,
            evaluations=evaluations,
            session_verdict=verdict,
        )
