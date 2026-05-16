"""
Code Execution Agent — hardened sandbox with resource limits.
FIX: working_dir now includes session_id for session isolation.
"""
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

from multimodal_ds.config import CODER_MODEL, OLLAMA_BASE_URL, LLM_TIMEOUT, OUTPUT_DIR
from multimodal_ds.memory.agent_memory import AgentMemory
from multimodal_ds.core.observability import agent_span, get_session_tracker

logger = logging.getLogger(__name__)

_CPU_SECONDS    = int(os.getenv("SANDBOX_CPU_SECONDS",  "60"))
_MEM_MB         = int(os.getenv("SANDBOX_MEM_MB",       "512"))
_STDOUT_CHARS   = int(os.getenv("SANDBOX_STDOUT_CHARS", "8000"))
_PROC_TIMEOUT_S = int(os.getenv("SANDBOX_TIMEOUT_S",    "300"))

SYSTEM_PROMPT = """You are a senior data scientist. You write precise, self‑contained Python.

MANDATORY RULES — follow every one, no exceptions:
1. First line of code: print(df.columns.tolist()) and print(df.shape)
2. Use ONLY column names confirmed by step 1 — never guess column names
3. Print descriptive stats: df.describe(), value_counts for every categorical column
4. ALWAYS split data: X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42, stratify=y)
       NEVER train and evaluate on the same data.
       For any model: print(classification_report(y_test, model.predict(X_test))),
       print(confusion_matrix(y_test, model.predict(X_test))),
       print('ROC-AUC:', roc_auc_score(y_test, model.predict_proba(X_test)[:,1])),
       print('feature importances:', dict(zip(feature_names, model.feature_importances_)))
5. ALWAYS set matplotlib backend as the very first two lines of code:
   import matplotlib
   matplotlib.use('Agg')
   import matplotlib.pyplot as plt
6. Save ALL plots: plt.savefig('filename.png', dpi=100, bbox_inches='tight',
   facecolor='white', edgecolor='none')
   Then IMMEDIATELY call plt.close('all') after every savefig call.
   Never reuse a figure across multiple plots.
7. Never call plt.show() under any circumstances.
8. When creating subplot grids with plt.subplots(rows, cols):
   - Calculate rows and cols AFTER loading data, based on actual column count
   - Never hardcode a grid size before knowing the data shape
   - Always use: fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
   - Always call fig.tight_layout() before savefig
   - If a subplot axis is unused, call ax.set_visible(False) on it
9. Save trained models: joblib.dump(model, 'model.pkl')
10. End with a FINDINGS block: print('=== FINDINGS ===') then 3‑5 quantitative sentences
11. NEVER evaluate a model on training data. Always use held-out test set.
   If cross-validation: use cross_val_score with cv=5 on training data only.

Output only valid Python code inside ```python ... ``` fences. No commentary outside the fences."""


def _sandbox_preexec() -> None:
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (_CPU_SECONDS, _CPU_SECONDS))
        mem_bytes = _MEM_MB * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    except Exception:
        pass


class CodeExecutionAgent:
    AGENT_NAME = "code_execution_agent"

    def __init__(self, working_dir: Optional[str] = None, session_id: str = "default"):
        # FIX: include session_id in working_dir for session isolation
        base = Path(working_dir) if working_dir else Path(OUTPUT_DIR)
        self.working_dir = base / session_id
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self.memory = AgentMemory()
        self._tracker = get_session_tracker(session_id)

    def execute_task(self, task: dict, data_context: str = "", file_paths: Optional[list] = None, max_retries: int = 2) -> dict:
        task_desc = task.get("description", str(task))
        task_name = task.get("name", "task")
        logger.info(f"[CodeAgent] Executing: {task_name}")

        with agent_span(self.AGENT_NAME, self.session_id, self._tracker) as span:
            span.set_metadata({"task_name": task_name})
            past_context = self._get_relevant_memory(task_desc)
            code = self._generate_code(task_desc, data_context, past_context)
            if not code:
                return {"success": False, "error": "Code generation failed", "code": "", "output": "", "files_created": []}
            span.set_chars(input_chars=len(task_desc) + len(data_context), output_chars=len(code))
            result = self._execute_with_retry(code, task_desc, data_context, file_paths, max_retries)
            span.set_metadata({"task_name": task_name, "success": result["success"], "files_created": result["files_created"]})

        status_msg = "successfully" if result["success"] else "with errors"
        self.memory.store_analysis_step(
            step_name=task_name,
            result=f"Code executed {status_msg}.\nOutput: {result['output'][:500]}\nFiles: {result['files_created']}",
            session_id=self.session_id,
        )
        return result

    def execute(self, task_description: str, data_context: str = "", file_paths: Optional[list] = None, max_retries: int = 2) -> dict:
        from pathlib import Path as _Path
        rag_context = self._retrieve_rag_context(task_description)
        if rag_context:
            data_context = f"Relevant document context (from ChromaDB):\n{rag_context}\n\n" + data_context
        if file_paths:
            file_list = "\n".join(f"  - {_Path(fp).name}" for fp in file_paths)
            data_context = f"Available data files (use exact names):\n{file_list}\n\n{data_context}"
        task = {"name": task_description[:80], "description": task_description}
        return self.execute_task(task=task, data_context=data_context, file_paths=file_paths, max_retries=max_retries)

    def _retrieve_rag_context(self, query: str, k: int = 4) -> str:
        try:
            results = self.memory.retrieve(query, n_results=k)
            if results:
                return "\n\n".join(r["content"] for r in results if r.get("content"))
        except Exception:
            pass
        return ""

    def _generate_code(self, task_desc: str, data_context: str, past_context: str) -> str:
        import httpx
        prompt = f"""Task: {task_desc}\nData Context:\n{data_context[:1500]}\nPrevious Context:\n{past_context[:500]}\nWorking directory: {self.working_dir}\nWrite Python code. Save all outputs to the current directory."""
        model = CODER_MODEL.replace("ollama/", "")
        try:
            response = httpx.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={"model": model, "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}], "stream": False, "options": {"num_predict": 4000, "temperature": 0.1}},
                timeout=LLM_TIMEOUT,
            )
            if response.status_code == 200:
                content = response.json().get("message", {}).get("content", "")
                return self._extract_code(content)
        except Exception as e:
            logger.error(f"[CodeAgent] Code generation failed: {e}")
        return ""

    def _execute_code(self, code: str, file_paths: Optional[list] = None):
        import shutil
        files_before = set(self.working_dir.glob("*"))
        script_path = None
        copied_files = []

        # Copy data files to working dir so code can find them locally
        if file_paths:
            for fp in file_paths:
                src = Path(fp)
                if src.exists():
                    dst = self.working_dir / src.name
                    if not dst.exists():
                        try:
                            shutil.copy2(src, dst)
                            copied_files.append(dst)
                        except Exception as e:
                            logger.warning(f"[CodeAgent] Failed to copy {src.name}: {e}")

        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", dir=self.working_dir, delete=False, encoding="utf-8") as f:
                f.write(code)
                script_path = Path(f.name)
            
            run_kwargs = {
                "args": [sys.executable, str(script_path)],
                "cwd": str(self.working_dir),
                "capture_output": True,
                "text": True,
                "timeout": _PROC_TIMEOUT_S
            }
            if sys.platform != "win32":
                run_kwargs["preexec_fn"] = _sandbox_preexec
            
            result = subprocess.run(**run_kwargs)
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            combined = stdout + (f"\n[stderr]:\n{stderr}" if stderr else "")
            
            if len(combined) > _STDOUT_CHARS:
                combined = combined[:_STDOUT_CHARS] + f"\n\n[OUTPUT TRUNCATED]"
            
            success = result.returncode == 0
        except subprocess.TimeoutExpired:
            return False, f"Execution timed out after {_PROC_TIMEOUT_S}s", []
        except Exception as e:
            return False, f"Execution error: {e}", []
        finally:
            if script_path and script_path.exists():
                try: script_path.unlink()
                except Exception: pass
            # Cleanup copied data files to keep sandbox clean
            for cf in copied_files:
                try: cf.unlink()
                except Exception: pass

        files_after = set(self.working_dir.glob("*"))
        new_files = [f.name for f in (files_after - files_before) if f.is_file() and f.suffix != ".py"]
        return success, combined, new_files

    def _execute_with_retry(self, code: str, task_desc: str, data_context: str, file_paths: Optional[list], max_retries: int) -> dict:
        import concurrent.futures

        success, output, files = self._execute_code(code, file_paths)
        if success:
            return {
                "success": True,
                "code": code,
                "output": output,
                "files_created": files,
                "error": "",
                "retries_used": 0,
            }

        for attempt in range(max_retries):
            # Generate fix and execute concurrently on retry
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fix_future = ex.submit(self._generate_fix, code, output, task_desc)
                fix_code = fix_future.result(timeout=120)

            if fix_code:
                success, output, files = self._execute_code(fix_code, file_paths)
                if success:
                    return {
                        "success": True,
                        "code": fix_code,
                        "output": output,
                        "files_created": files,
                        "error": "",
                        "retries_used": attempt + 1,
                    }
                code = fix_code

        return {
            "success": False,
            "code": code,
            "output": output,
            "files_created": files,
            "error": output,
            "retries_used": max_retries,
        }

    def _generate_fix(self, failed_code: str, error_output: str, task_desc: str) -> str:
        import httpx
        model = CODER_MODEL.replace("ollama/", "")
        prompt = f"""Fix this Python code that failed.\nTask: {task_desc}\nFailed code:\n```python\n{failed_code[:1500]}\n```\nError:\n{error_output[:500]}\nProvide ONE complete fixed Python script in ```python ... ``` fences."""
        try:
            response = httpx.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={"model": model, "messages": [{"role": "system", "content": "Fix Python code. Output only the fixed code in ```python``` fences."}, {"role": "user", "content": prompt}], "stream": False, "options": {"num_predict": 2000, "temperature": 0.1}},
                timeout=LLM_TIMEOUT,
            )
            if response.status_code == 200:
                return self._extract_code(response.json().get("message", {}).get("content", ""))
        except Exception:
            pass
        return ""

    def _extract_code(self, text: str) -> str:
        import re
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        if "```python" in text:
            parts = text.split("```python")
            if len(parts) > 1:
                return parts[1].split("```")[0].strip()
        if "```" in text:
            parts = text.split("```")
            if len(parts) > 1:
                return parts[1].strip()
        return text.strip()

    def _get_relevant_memory(self, query: str) -> str:
        memories = self.memory.retrieve(query, n_results=3)
        if not memories:
            return ""
        return "\n".join(m["content"][:200] for m in memories)
