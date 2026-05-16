"""
UIBusAdapter — bridges the MessageBus to the Streamlit frontend.

Subscribes to all message types on the global bus and pushes serialised
event dicts into a thread-safe queue. The Streamlit polling loop calls
drain_queue() on each refresh cycle to pick up new events without blocking
the UI thread.

Responsibilities:
  - Subscribe to every MessageType on startup
  - Convert AgentMessage → JSON-safe dict for the UI
  - Persist session logs to .session_logs/<session_id>.jsonl
  - Provide drain_queue() for the Streamlit polling loop
  - Provide write_log() as a static method for direct log writes

Thread safety:
  All queue operations use threading.Lock. The queue is a plain list;
  drain_queue() atomically swaps it for an empty list so no messages
  are lost between concurrent drain calls.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from multimodal_ds.core.message_bus import AgentMessage, MessageType, get_bus

logger = logging.getLogger(__name__)

# Default log directory (relative to project root)
_LOG_DIR = Path(".session_logs")


class UIBusAdapter:
    """
    Bridges the global MessageBus to the Streamlit UI.

    Usage (in streamlit_app.py):

        adapter = UIBusAdapter()   # subscribes immediately

        # In the polling loop:
        new_events = adapter.drain_queue()
        for event in new_events:
            st.write(event)
    """

    def __init__(self, log_dir: Optional[Path] = None):
        self._queue: List[Dict[str, Any]] = []
        self._lock  = threading.Lock()
        self._log_dir = Path(log_dir) if log_dir else _LOG_DIR

        # Subscribe to every known message type
        bus = get_bus()
        bus.subscribe_all(self._on_message)
        logger.info("[UIBusAdapter] Subscribed to all message types")

    # ── Internal handler ───────────────────────────────────────────────────

    def _on_message(self, msg: AgentMessage) -> None:
        """
        Called by the MessageBus for every published message.
        Converts to a JSON-safe dict and appends to the internal queue.
        """
        event = {
            "timestamp":  msg.timestamp,
            "type":       msg.msg_type.value,
            "sender":     msg.sender,
            "session_id": msg.session_id,
            "msg_id":     msg.msg_id,
            "priority":   msg.priority.value,
            "payload":    self._safe_payload(msg.payload),
        }
        with self._lock:
            self._queue.append(event)

    @staticmethod
    def _safe_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make payload JSON-safe: drop non-serialisable values, truncate large strings.
        Never raises — returns a best-effort dict.
        """
        safe = {}
        for k, v in payload.items():
            try:
                if isinstance(v, str) and len(v) > 500:
                    safe[k] = v[:500] + "…[truncated]"
                elif isinstance(v, (str, int, float, bool, list, dict)) or v is None:
                    # Quick round-trip check
                    json.dumps(v)
                    safe[k] = v
                else:
                    safe[k] = str(v)
            except (TypeError, ValueError):
                safe[k] = f"<non-serialisable: {type(v).__name__}>"
        return safe

    # ── Public API ─────────────────────────────────────────────────────────

    def drain_queue(self) -> List[Dict[str, Any]]:
        """
        Atomically return and clear all queued events.
        Call this from the Streamlit polling loop every refresh cycle.
        Returns an empty list if nothing new has arrived.
        """
        with self._lock:
            events, self._queue = self._queue, []
        return events

    def queue_size(self) -> int:
        """Return the number of events currently queued (without draining)."""
        with self._lock:
            return len(self._queue)

    # ── Log persistence ────────────────────────────────────────────────────

    @staticmethod
    def write_log(
        session_id: str,
        messages: List[Dict[str, Any]],
        log_dir: Optional[Path] = None,
    ) -> Path:
        """
        Append a list of event dicts to a JSONL log file for the session.

        File path: <log_dir>/<session_id>.jsonl
        Format:    one JSON object per line (newline-delimited JSON)

        Returns the path of the written file.

        Usage:
            log_path = UIBusAdapter.write_log("session_abc", adapter.drain_queue())
        """
        base = Path(log_dir) if log_dir else _LOG_DIR
        base.mkdir(parents=True, exist_ok=True)

        log_path = base / f"{session_id}.jsonl"
        with open(log_path, "a", encoding="utf-8") as f:
            for msg in messages:
                f.write(json.dumps(msg, default=str) + "\n")

        logger.debug(f"[UIBusAdapter] Wrote {len(messages)} events → {log_path}")
        return log_path

    def flush_to_log(self, session_id: str) -> Optional[Path]:
        """
        Drain the queue and immediately persist to disk.
        Convenience wrapper around drain_queue() + write_log().
        Returns None if there was nothing to write.
        """
        events = self.drain_queue()
        if not events:
            return None
        return self.write_log(session_id, events, self._log_dir)

    @staticmethod
    def read_log(session_id: str, log_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        """
        Read all persisted events for a session from disk.
        Returns an empty list if the log file doesn't exist.
        """
        base = Path(log_dir) if log_dir else _LOG_DIR
        log_path = base / f"{session_id}.jsonl"
        if not log_path.exists():
            return []

        events = []
        for line in log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"[UIBusAdapter] Skipping malformed log line: {line[:80]}")
        return events
