import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class Span:
    def __init__(self, agent_name, session_id, tracker=None):
        self.agent_name = agent_name
        self.session_id = session_id
        self.tracker = tracker
        self.metadata = {}

    def set_metadata(self, metadata: dict):
        self.metadata.update(metadata)

    def set_chars(self, input_chars: int = 0, output_chars: int = 0):
        pass

@contextmanager
def agent_span(agent_name, session_id, tracker=None):
    span = Span(agent_name, session_id, tracker)
    try:
        yield span
    finally:
        pass

def get_session_tracker(session_id: str):
    return None
