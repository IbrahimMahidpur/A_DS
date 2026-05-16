import json
from pathlib import Path

import pytest

from multimodal_ds.core.message_bus import (
    AgentMessage,
    MessageType,
    get_bus,
)
from multimodal_ds.frontend.ui_bus_adapter import UIBusAdapter

@pytest.fixture
def fresh_adapter():
    # Ensure a fresh adapter for each test – the bus is a singleton, so we create a new instance.
    return UIBusAdapter()

def test_adapter_receives_message_and_drain_queue(fresh_adapter):
    bus = get_bus()
    # Publish a simple message.
    msg = AgentMessage(
        msg_type=MessageType.SESSION_START,
        payload={"objective": "test", "file_count": 0},
        sender="test_sender",
        session_id="test123",
    )
    bus.publish(msg)

    # Drain the queue – the adapter should have captured the message.
    msgs = fresh_adapter.drain_queue()
    assert any(m["type"] == "SESSION_START" for m in msgs), "SESSION_START not found in queue"
    # The payload should match.
    captured = next(m for m in msgs if m["type"] == "SESSION_START")
    assert captured["payload"]["objective"] == "test"

def test_write_log_creates_file(tmp_path):
    # Use a temporary directory for logs.
    session_id = "logtest"
    messages = [
        {"timestamp": "2026-05-06T12:00:00Z", "type": "TEST", "sender": "me", "payload": {"msg": "hello"}}
    ]
    # Monkey‑patch the log directory to the temp path.
    original_dir = Path(".session_logs")
    # Ensure cleanup after test.
    try:
        # Write log using the adapter's static method.
        log_path = UIBusAdapter.write_log(session_id, messages)
        assert log_path.exists(), "Log file was not created"
        # Verify content is valid JSON lines.
        lines = log_path.read_text().splitlines()
        assert len(lines) == 1
        obj = json.loads(lines[0])
        assert obj["type"] == "TEST"
    finally:
        # Remove the created file and directory to avoid side effects.
        if log_path.exists():
            log_path.unlink()
        if original_dir.exists() and not any(original_dir.iterdir()):
            original_dir.rmdir()
