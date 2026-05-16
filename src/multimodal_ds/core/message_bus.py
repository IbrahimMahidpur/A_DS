"""
MessageBus — thread-safe pub/sub event system for inter-agent communication.

Design:
  - Singleton pattern via get_bus() / reset_bus()
  - Typed message envelopes (AgentMessage) with correlation IDs for request/reply
  - Dead-letter queue for undelivered messages
  - Per-session audit trail for observability
  - Middleware chain for cross-cutting concerns (PII filtering, rate limiting, etc.)
  - publish_and_wait() for synchronous request/reply patterns

MessageType taxonomy:
  INGEST_*   — ingestion pipeline events
  PLAN_*     — planner agent events
  CODE_*     — code execution agent events
  STATS_*    — statistical agent events
  VIZ_*      — visualization agent events
  EVAL_*     — evaluation agent events
  SESSION_*  — session lifecycle events
  HANDOFF    — agent-to-agent task handoff

Thread safety:
  All public methods are protected by a reentrant lock.
  publish_and_wait() uses a threading.Event per correlation ID.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
#  Enums
# ══════════════════════════════════════════════════════════════════════════

class Priority(Enum):
    LOW    = 1
    NORMAL = 2  # default
    HIGH   = 3
    URGENT = 4


class MessageType(Enum):
    # ── Ingestion ──────────────────────────────────────────────────────────
    INGEST_REQUEST  = "INGEST_REQUEST"
    INGEST_COMPLETE = "INGEST_COMPLETE"
    INGEST_FAILED   = "INGEST_FAILED"
    INGEST_BLOCKED  = "INGEST_BLOCKED"   # PII gate blocked the document

    # ── Planning ───────────────────────────────────────────────────────────
    PLAN_REQUEST  = "PLAN_REQUEST"
    PLAN_COMPLETE = "PLAN_COMPLETE"
    PLAN_FAILED   = "PLAN_FAILED"

    # ── Code execution ─────────────────────────────────────────────────────
    CODE_REQUEST  = "CODE_REQUEST"
    CODE_COMPLETE = "CODE_COMPLETE"
    CODE_FAILED   = "CODE_FAILED"
    CODE_RETRY    = "CODE_RETRY"

    # ── Statistical agent ──────────────────────────────────────────────────
    STATS_REQUEST  = "STATS_REQUEST"
    STATS_COMPLETE = "STATS_COMPLETE"

    # ── Visualization agent ────────────────────────────────────────────────
    VIZ_REQUEST  = "VIZ_REQUEST"
    VIZ_COMPLETE = "VIZ_COMPLETE"
    VIZ_FAILED   = "VIZ_FAILED"

    # ── Evaluation agent ───────────────────────────────────────────────────
    EVAL_REQUEST  = "EVAL_REQUEST"
    EVAL_COMPLETE = "EVAL_COMPLETE"
    EVAL_FLAGGED  = "EVAL_FLAGGED"   # any dimension below threshold

    # ── Session lifecycle ──────────────────────────────────────────────────
    SESSION_START = "SESSION_START"
    SESSION_END   = "SESSION_END"

    # ── Agent handoff ──────────────────────────────────────────────────────
    HANDOFF = "HANDOFF"


# ══════════════════════════════════════════════════════════════════════════
#  AgentMessage
# ══════════════════════════════════════════════════════════════════════════

def _short_id() -> str:
    """Generate a 12-char collision-resistant ID."""
    return uuid.uuid4().hex[:12]


@dataclass
class AgentMessage:
    """
    Typed message envelope for inter-agent communication.

    Correlation model:
      - msg_id:         unique ID for THIS message
      - correlation_id: shared across an entire request/reply chain
      - causation_id:   msg_id of the message that directly caused this one

    Example:
        request  → msg_id="aaa", correlation_id="aaa", causation_id=None
        reply    → msg_id="bbb", correlation_id="aaa", causation_id="aaa"
        follow-up→ msg_id="ccc", correlation_id="aaa", causation_id="bbb"
    """
    msg_type:       MessageType
    payload:        Dict[str, Any]
    sender:         str
    session_id:     str          = ""
    recipient:      str          = ""
    priority:       Priority     = Priority.NORMAL
    msg_id:         str          = field(default_factory=_short_id)
    correlation_id: str          = field(default_factory=_short_id)
    causation_id:   Optional[str] = None
    timestamp:      str          = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
    )
    metadata:       Dict[str, Any] = field(default_factory=dict)

    def reply(
        self,
        msg_type: MessageType,
        payload: Dict[str, Any],
        sender: str,
    ) -> "AgentMessage":
        """
        Create a reply that preserves the correlation chain.
        The reply's correlation_id matches the original; causation_id points
        to THIS message's msg_id.
        """
        return AgentMessage(
            msg_type=msg_type,
            payload=payload,
            sender=sender,
            session_id=self.session_id,
            recipient=self.sender,
            correlation_id=self.correlation_id,   # preserve chain
            causation_id=self.msg_id,             # caused by this message
        )

    def to_dict(self) -> dict:
        """
        Safe serialisation — payload VALUES are excluded (may contain PII).
        Only payload keys are exposed for logging/auditing.
        """
        return {
            "msg_id":         self.msg_id,
            "msg_type":       self.msg_type.value,
            "sender":         self.sender,
            "recipient":      self.recipient,
            "session_id":     self.session_id,
            "priority":       self.priority.value,
            "correlation_id": self.correlation_id,
            "causation_id":   self.causation_id,
            "timestamp":      self.timestamp,
            "payload_keys":   list(self.payload.keys()),  # keys only — no values
            "metadata":       self.metadata,
        }


# ══════════════════════════════════════════════════════════════════════════
#  HandoffContext
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class HandoffContext:
    """
    Structured payload for HANDOFF messages — agent-to-agent task delegation.

    Serialises to/from a plain dict so it can be stored in AgentMessage.payload
    and survive LangGraph checkpoint serialisation.
    """
    from_agent:    str
    to_agent:      str
    task:          Dict[str, Any]
    data_context:  str                      = ""
    prior_outputs: List[Dict[str, Any]]     = field(default_factory=list)
    instructions:  str                      = ""
    constraints:   Dict[str, Any]           = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "from_agent":    self.from_agent,
            "to_agent":      self.to_agent,
            "task":          self.task,
            "data_context":  self.data_context,
            "prior_outputs": self.prior_outputs,
            "instructions":  self.instructions,
            "constraints":   self.constraints,
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "HandoffContext":
        return cls(
            from_agent=payload["from_agent"],
            to_agent=payload["to_agent"],
            task=payload["task"],
            data_context=payload.get("data_context", ""),
            prior_outputs=payload.get("prior_outputs", []),
            instructions=payload.get("instructions", ""),
            constraints=payload.get("constraints", {}),
        )


# ══════════════════════════════════════════════════════════════════════════
#  MessageBus
# ══════════════════════════════════════════════════════════════════════════

class MessageBus:
    """
    Thread-safe pub/sub event bus for inter-agent messaging.

    Key design decisions:
      - Handlers are called synchronously in publish() — simple, predictable,
        no hidden async surprises. If you need async, wrap in a thread.
      - Handler exceptions are caught, logged, and routed to the DLQ so one
        bad handler never prevents other handlers from receiving the message.
      - Middleware runs before dispatch — can enrich, filter, or drop messages.
      - Wildcard subscribers receive ALL message types (useful for logging/tracing).
      - Session audit trail keyed by session_id for end-to-end traceability.
    """

    def __init__(self):
        self._lock = threading.RLock()

        # type → list of handlers
        self._subscribers:  Dict[MessageType, List[Callable]] = {}

        # wildcard subscribers — receive all messages
        self._wildcard_subs: List[Callable] = []

        # middleware chain — each callable receives a message, returns it (or None to drop)
        self._middleware: List[Callable] = []

        # dead-letter queue — undelivered or handler-crashed messages
        self._dlq: List[Dict[str, Any]] = []

        # per-session audit trail
        self._audit: Dict[str, List[Dict]] = {}

        # pending reply waiters: correlation_id → (Event, container)
        self._waiters: Dict[str, tuple] = {}

        # stats counters
        self._stats = {
            "published": 0,
            "delivered": 0,
            "dropped":   0,
            "dlq":       0,
        }

    # ── Subscription management ────────────────────────────────────────────

    def subscribe(self, msg_type: MessageType, handler: Callable) -> None:
        """Register a handler for a specific message type."""
        with self._lock:
            if msg_type not in self._subscribers:
                self._subscribers[msg_type] = []
            if handler not in self._subscribers[msg_type]:
                self._subscribers[msg_type].append(handler)
                logger.debug(f"[Bus] Subscribed {handler.__name__!r} → {msg_type.value}")

    def subscribe_all(self, handler: Callable) -> None:
        """Register a wildcard handler that receives every message."""
        with self._lock:
            if handler not in self._wildcard_subs:
                self._wildcard_subs.append(handler)

    def unsubscribe(self, msg_type: MessageType, handler: Callable) -> None:
        """Remove a handler for a specific message type."""
        with self._lock:
            if msg_type in self._subscribers:
                try:
                    self._subscribers[msg_type].remove(handler)
                except ValueError:
                    pass

    def add_middleware(self, middleware: Callable) -> None:
        """
        Add a middleware function to the chain.
        Middleware signature: (msg: AgentMessage) -> Optional[AgentMessage]
        Return None to drop the message; return the (possibly modified) message to continue.
        """
        with self._lock:
            self._middleware.append(middleware)

    # ── Publishing ─────────────────────────────────────────────────────────

    def publish(self, msg: AgentMessage) -> bool:
        """
        Publish a message to all registered subscribers.

        Returns True if at least one subscriber received the message.
        Returns False if the message was dropped (middleware) or went to DLQ.

        Handler exceptions are caught individually — one bad handler never
        blocks other handlers from receiving the message.
        """
        with self._lock:
            self._stats["published"] += 1

            # ── Run middleware chain ───────────────────────────────────────
            current = msg
            for mw in self._middleware:
                try:
                    result = mw(current)
                    if result is None:
                        # Middleware dropped the message
                        self._stats["dropped"] += 1
                        logger.debug(
                            f"[Bus] Message {msg.msg_id} dropped by middleware "
                            f"{mw.__name__!r}"
                        )
                        return False
                    current = result
                except Exception as e:
                    logger.error(f"[Bus] Middleware {mw.__name__!r} raised: {e}")

            # ── Audit trail ────────────────────────────────────────────────
            if current.session_id:
                if current.session_id not in self._audit:
                    self._audit[current.session_id] = []
                self._audit[current.session_id].append(current.to_dict())

            # ── Dispatch to specific + wildcard subscribers ────────────────
            handlers = list(self._subscribers.get(current.msg_type, []))
            wildcards = list(self._wildcard_subs)
            all_handlers = handlers + wildcards

            # ── Check for reply waiters BEFORE dispatching ─────────────────
            # Only trigger waiter for REPLY messages (causation_id set means
            # this message was caused by another — i.e. it IS the reply).
            # Never trigger for the originating message itself.
            waiter = self._waiters.get(current.correlation_id)
            if waiter and current.causation_id is not None:
                event, container = waiter
                container.append(current)
                event.set()

            if not all_handlers:
                if not waiter:
                    # No one listening — DLQ it
                    self._dlq.append({
                        **current.to_dict(),
                        "reason": "no_subscribers",
                    })
                    self._stats["dlq"] += 1
                    logger.debug(
                        f"[Bus] No subscribers for {current.msg_type.value} "
                        f"— sent to DLQ"
                    )
                return False

            delivered = False
            for handler in all_handlers:
                try:
                    handler(current)
                    delivered = True
                    self._stats["delivered"] += 1
                except Exception as e:
                    self._dlq.append({
                        **current.to_dict(),
                        "reason":    "handler_exception",
                        "handler":   getattr(handler, "__name__", repr(handler)),
                        "exception": str(e),
                    })
                    self._stats["dlq"] += 1
                    logger.error(
                        f"[Bus] Handler {getattr(handler, '__name__', '?')!r} "
                        f"raised on {current.msg_type.value}: {e}"
                    )

            return delivered

    def publish_and_wait(
        self,
        msg: AgentMessage,
        reply_type: MessageType,
        timeout_s: float = 30.0,
    ) -> Optional[AgentMessage]:
        """
        Publish a message and block until a reply with matching correlation_id arrives.
        Waiter is registered before publish to avoid missing fast replies.
        """
        event = threading.Event()
        container: List[AgentMessage] = []

        # Register waiter BEFORE publishing — prevents a race where the reply
        # arrives and is processed before we register the waiter
        with self._lock:
            self._waiters[msg.correlation_id] = (event, container)

        try:
            # publish() acquires self._lock — safe because self._lock is RLock
            self.publish(msg)
            signalled = event.wait(timeout=timeout_s)
            if signalled and container:
                return container[0]
            logger.debug(f"[Bus] publish_and_wait timed out after {timeout_s}s for correlation {msg.correlation_id[:8]}")
            return None
        except Exception as e:
            logger.error(f"[Bus] publish_and_wait error: {e}")
            return None
        finally:
            with self._lock:
                self._waiters.pop(msg.correlation_id, None)

    # ── Introspection ──────────────────────────────────────────────────────

    def get_dlq(self) -> List[Dict]:
        """Return a snapshot of the dead-letter queue."""
        with self._lock:
            return list(self._dlq)

    def get_session_trace(self, session_id: str) -> List[Dict]:
        """Return all messages audited for a session (by session_id)."""
        with self._lock:
            return list(self._audit.get(session_id, []))

    def clear_session(self, session_id: str) -> None:
        """Remove audit entries for a session (e.g., after export)."""
        with self._lock:
            self._audit.pop(session_id, None)

    def get_stats(self) -> Dict[str, int]:
        """Return a snapshot of publish/delivery/drop/DLQ counters."""
        with self._lock:
            return dict(self._stats)


# ══════════════════════════════════════════════════════════════════════════
#  Singleton
# ══════════════════════════════════════════════════════════════════════════

_bus_instance: Optional[MessageBus] = None
_bus_lock = threading.Lock()


def get_bus() -> MessageBus:
    """Return the process-level singleton MessageBus, creating it if needed."""
    global _bus_instance
    if _bus_instance is None:
        with _bus_lock:
            if _bus_instance is None:
                _bus_instance = MessageBus()
                logger.info("[Bus] Singleton MessageBus created")
    return _bus_instance


def reset_bus() -> None:
    """
    Replace the singleton with a fresh instance.
    Primarily for testing — ensures no state leakage between tests.
    """
    global _bus_instance
    with _bus_lock:
        _bus_instance = None
    logger.debug("[Bus] Singleton reset")
