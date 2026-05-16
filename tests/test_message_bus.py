"""
Tests for MessageBus and HandoffContext.
Run with: pytest tests/test_message_bus.py -v
"""
import threading
import time
import pytest

from multimodal_ds.core.message_bus import (
    AgentMessage, HandoffContext, MessageBus, MessageType,
    Priority, get_bus, reset_bus,
)


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def fresh_bus():
    """Each test gets a clean bus — no state leakage."""
    reset_bus()
    yield
    reset_bus()


@pytest.fixture
def bus():
    return MessageBus()


def make_msg(
    msg_type=MessageType.CODE_REQUEST,
    sender="test_agent",
    session_id="sess_001",
    payload=None,
) -> AgentMessage:
    return AgentMessage(
        msg_type=msg_type,
        payload=payload or {"data": "test"},
        sender=sender,
        session_id=session_id,
    )


# ── AgentMessage ───────────────────────────────────────────────────────────

class TestAgentMessage:
    def test_auto_ids(self):
        msg = make_msg()
        assert len(msg.msg_id) == 12
        assert len(msg.correlation_id) == 12
        assert msg.causation_id is None

    def test_reply_preserves_correlation(self):
        original = make_msg()
        reply = original.reply(
            MessageType.CODE_COMPLETE,
            payload={"result": "ok"},
            sender="code_agent",
        )
        assert reply.correlation_id == original.correlation_id
        assert reply.causation_id == original.msg_id
        assert reply.recipient == original.sender
        assert reply.sender == "code_agent"

    def test_to_dict_excludes_payload_values(self):
        msg = make_msg(payload={"secret": "password123"})
        d = msg.to_dict()
        # payload_keys shows keys but NOT values — PII protection
        assert "payload_keys" in d
        assert "secret" in d["payload_keys"]
        assert "password123" not in str(d)

    def test_priority_default_is_normal(self):
        msg = make_msg()
        assert msg.priority == Priority.NORMAL

    def test_timestamp_is_utc_iso(self):
        msg = make_msg()
        assert "T" in msg.timestamp   # ISO format
        assert "+" in msg.timestamp or "Z" in msg.timestamp or msg.timestamp.endswith("+00:00")


# ── MessageBus — Subscribe / Publish ──────────────────────────────────────

class TestMessageBusBasic:
    def test_subscribe_and_receive(self, bus):
        received = []
        bus.subscribe(MessageType.CODE_REQUEST, received.append)

        msg = make_msg(MessageType.CODE_REQUEST)
        result = bus.publish(msg)

        assert result is True
        assert len(received) == 1
        assert received[0].msg_id == msg.msg_id

    def test_multiple_subscribers_same_type(self, bus):
        log_a, log_b = [], []
        bus.subscribe(MessageType.INGEST_COMPLETE, log_a.append)
        bus.subscribe(MessageType.INGEST_COMPLETE, log_b.append)

        bus.publish(make_msg(MessageType.INGEST_COMPLETE))

        assert len(log_a) == 1
        assert len(log_b) == 1

    def test_no_subscriber_goes_to_dlq(self, bus):
        msg = make_msg(MessageType.VIZ_COMPLETE)   # Nobody subscribed
        result = bus.publish(msg)

        assert result is False
        dlq = bus.get_dlq()
        assert len(dlq) == 1
        assert dlq[0]["reason"] == "no_subscribers"

    def test_wildcard_subscriber_receives_all(self, bus):
        all_msgs = []
        bus.subscribe_all(all_msgs.append)

        bus.publish(make_msg(MessageType.CODE_REQUEST))
        bus.publish(make_msg(MessageType.PLAN_COMPLETE))
        bus.publish(make_msg(MessageType.STATS_COMPLETE))

        assert len(all_msgs) == 3

    def test_unsubscribe(self, bus):
        received = []
        bus.subscribe(MessageType.CODE_COMPLETE, received.append)
        bus.publish(make_msg(MessageType.CODE_COMPLETE))
        assert len(received) == 1

        bus.unsubscribe(MessageType.CODE_COMPLETE, received.append)
        bus.publish(make_msg(MessageType.CODE_COMPLETE))
        # Still 1 — second publish not received
        assert len(received) == 1

    def test_handler_exception_does_not_crash_bus(self, bus):
        def bad_handler(msg):
            raise RuntimeError("I crashed")

        good_log = []
        bus.subscribe(MessageType.CODE_REQUEST, bad_handler)
        bus.subscribe(MessageType.CODE_REQUEST, good_log.append)

        result = bus.publish(make_msg(MessageType.CODE_REQUEST))

        # Good handler still ran
        assert len(good_log) == 1
        # Bad handler error in DLQ
        assert any(e["reason"] == "handler_exception" for e in bus.get_dlq())


# ── MessageBus — Audit Log ────────────────────────────────────────────────

class TestAuditLog:
    def test_session_trace_records_messages(self, bus):
        bus.subscribe(MessageType.CODE_REQUEST, lambda m: None)
        bus.publish(make_msg(MessageType.CODE_REQUEST, session_id="s1"))
        bus.publish(make_msg(MessageType.CODE_REQUEST, session_id="s1"))
        bus.publish(make_msg(MessageType.CODE_REQUEST, session_id="s2"))

        trace_s1 = bus.get_session_trace("s1")
        trace_s2 = bus.get_session_trace("s2")

        assert len(trace_s1) == 2
        assert len(trace_s2) == 1

    def test_trace_contains_metadata(self, bus):
        bus.subscribe(MessageType.INGEST_COMPLETE, lambda m: None)
        bus.publish(make_msg(MessageType.INGEST_COMPLETE, session_id="s1"))

        trace = bus.get_session_trace("s1")
        entry = trace[0]

        assert "msg_id" in entry
        assert "timestamp" in entry
        assert "sender" in entry
        assert entry["msg_type"] == MessageType.INGEST_COMPLETE.value

    def test_no_session_id_not_audited(self, bus):
        bus.subscribe(MessageType.CODE_REQUEST, lambda m: None)
        msg = AgentMessage(
            msg_type=MessageType.CODE_REQUEST,
            payload={},
            sender="test",
            session_id="",       # Empty — should not be audited
        )
        bus.publish(msg)
        assert bus.get_session_trace("") == []

    def test_clear_session(self, bus):
        bus.subscribe(MessageType.CODE_REQUEST, lambda m: None)
        bus.publish(make_msg(session_id="cleanup_me"))
        assert len(bus.get_session_trace("cleanup_me")) == 1

        bus.clear_session("cleanup_me")
        assert len(bus.get_session_trace("cleanup_me")) == 0


# ── MessageBus — Middleware ───────────────────────────────────────────────

class TestMiddleware:
    def test_middleware_can_drop_message(self, bus):
        received = []
        bus.subscribe(MessageType.CODE_REQUEST, received.append)

        def drop_all(msg):
            return None   # Drop

        bus.add_middleware(drop_all)
        bus.publish(make_msg(MessageType.CODE_REQUEST))

        assert len(received) == 0
        assert bus.get_stats()["dropped"] == 1

    def test_middleware_can_enrich_message(self, bus):
        received = []
        bus.subscribe(MessageType.CODE_REQUEST, received.append)

        def add_trace_id(msg):
            msg.metadata["trace_id"] = "abc-123"
            return msg

        bus.add_middleware(add_trace_id)
        bus.publish(make_msg(MessageType.CODE_REQUEST))

        assert received[0].metadata["trace_id"] == "abc-123"

    def test_pii_filter_middleware(self, bus):
        """Simulate a PII-filtering middleware that drops flagged messages."""
        received = []
        bus.subscribe(MessageType.INGEST_COMPLETE, received.append)

        def pii_filter(msg):
            # Drop if payload contains a known PII key
            if "ssn" in msg.payload:
                return None
            return msg

        bus.add_middleware(pii_filter)

        bus.publish(AgentMessage(
            msg_type=MessageType.INGEST_COMPLETE,
            payload={"ssn": "123-45-6789"},
            sender="test",
            session_id="s1",
        ))
        bus.publish(AgentMessage(
            msg_type=MessageType.INGEST_COMPLETE,
            payload={"name": "clean data"},
            sender="test",
            session_id="s1",
        ))

        assert len(received) == 1   # Only the clean message


# ── MessageBus — publish_and_wait ─────────────────────────────────────────

class TestPublishAndWait:
    def test_request_reply_same_correlation(self, bus):
        """Simulate a code agent that replies after processing."""
        def code_agent_handler(request_msg: AgentMessage):
            time.sleep(0.05)   # Simulate work
            reply = request_msg.reply(
                MessageType.CODE_COMPLETE,
                payload={"files": ["plot.png"]},
                sender="code_agent",
            )
            bus.publish(reply)

        bus.subscribe(MessageType.CODE_REQUEST, code_agent_handler)

        request = make_msg(MessageType.CODE_REQUEST, session_id="rr_test")
        response = bus.publish_and_wait(request, MessageType.CODE_COMPLETE, timeout_s=5.0)

        assert response is not None
        assert response.msg_type == MessageType.CODE_COMPLETE
        assert response.payload["files"] == ["plot.png"]
        assert response.correlation_id == request.correlation_id

    def test_timeout_returns_none(self, bus):
        # Subscribe but never reply
        bus.subscribe(MessageType.CODE_REQUEST, lambda m: None)

        request = make_msg(MessageType.CODE_REQUEST)
        response = bus.publish_and_wait(request, MessageType.CODE_COMPLETE, timeout_s=0.1)

        assert response is None


# ── MessageBus — Thread Safety ────────────────────────────────────────────

class TestThreadSafety:
    def test_concurrent_publish(self, bus):
        received = []
        lock = threading.Lock()

        def handler(msg):
            with lock:
                received.append(msg.msg_id)

        bus.subscribe(MessageType.CODE_REQUEST, handler)

        threads = [
            threading.Thread(
                target=bus.publish,
                args=(make_msg(MessageType.CODE_REQUEST),)
            )
            for _ in range(50)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(received) == 50

    def test_concurrent_subscribe_publish(self, bus):
        """Subscribing and publishing simultaneously should not deadlock."""
        received = []

        def publisher():
            for _ in range(20):
                bus.publish(make_msg(MessageType.PLAN_COMPLETE))
                time.sleep(0.001)

        def subscriber():
            bus.subscribe(MessageType.PLAN_COMPLETE, received.append)

        t1 = threading.Thread(target=publisher)
        t2 = threading.Thread(target=subscriber)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)
        # No deadlock = pass


# ── HandoffContext ────────────────────────────────────────────────────────

class TestHandoffContext:
    def test_roundtrip_serialization(self):
        ctx = HandoffContext(
            from_agent="orchestrator",
            to_agent="code_execution_agent",
            task={"step": 1, "name": "EDA"},
            data_context="Dataset: 1000 rows × 10 cols",
            prior_outputs=[{"name": "ingest", "output": "done"}],
            instructions="Save all plots",
            constraints={"max_runtime_s": 300},
        )
        payload = ctx.to_payload()
        restored = HandoffContext.from_payload(payload)

        assert restored.from_agent == ctx.from_agent
        assert restored.to_agent == ctx.to_agent
        assert restored.task == ctx.task
        assert restored.constraints == ctx.constraints

    def test_handoff_published_on_bus(self, bus):
        """Verify a HANDOFF message with HandoffContext payload is bus-compatible."""
        received = []
        bus.subscribe(MessageType.HANDOFF, received.append)

        ctx = HandoffContext(
            from_agent="orchestrator",
            to_agent="code_agent",
            task={"name": "EDA"},
            data_context="...",
        )
        bus.publish(AgentMessage(
            msg_type=MessageType.HANDOFF,
            payload=ctx.to_payload(),
            sender="orchestrator",
            recipient="code_agent",
            session_id="s1",
        ))

        assert len(received) == 1
        restored = HandoffContext.from_payload(received[0].payload)
        assert restored.to_agent == "code_agent"


# ── Singleton ─────────────────────────────────────────────────────────────

class TestSingleton:
    def test_get_bus_returns_same_instance(self):
        b1 = get_bus()
        b2 = get_bus()
        assert b1 is b2

    def test_reset_bus_creates_fresh_instance(self):
        b1 = get_bus()
        reset_bus()
        b2 = get_bus()
        assert b1 is not b2

    def test_stats_track_across_calls(self):
        bus = get_bus()
        bus.subscribe(MessageType.CODE_REQUEST, lambda m: None)
        bus.publish(make_msg(MessageType.CODE_REQUEST))
        bus.publish(make_msg(MessageType.CODE_REQUEST))

        stats = bus.get_stats()
        assert stats["published"] == 2
