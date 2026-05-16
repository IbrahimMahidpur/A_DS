import importlib
import sys

def test_streamlit_app_importable():
    """Import the Streamlit entry point to ensure it doesn't raise at import time.

    The script defines UI layout using Streamlit APIs, which are safe to import in a test
    environment. This test simply verifies that the module loads without SyntaxError
    or missing‑dependency errors.
    """
    try:
        import multimodal_ds.frontend.streamlit_app as app  # noqa: F401
    except Exception as e:
        raise AssertionError(f"Failed to import streamlit_app: {e}")
