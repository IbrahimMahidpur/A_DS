import pytest
from multimodal_ds.core.message_bus import reset_bus

@pytest.fixture(autouse=True, scope="session")
def reset_message_bus():
    """Reset the global MessageBus singleton for a clean test session."""
    reset_bus()

# Provide temporary output directory fixture expected by older tests
@pytest.fixture
def temp_output_dir(tmp_path):
    return tmp_path

@pytest.fixture
def tmp_output_dir(temp_output_dir):
    """Alias for backward compatibility"""
    return temp_output_dir
