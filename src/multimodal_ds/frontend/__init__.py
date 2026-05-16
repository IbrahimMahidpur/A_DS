"""Multimodal Agentic Data Science Engine."""
__version__ = "1.0.0"

try:
    import langchain
    if not hasattr(langchain, "debug"):
        langchain.debug = False
except ImportError:
    pass
