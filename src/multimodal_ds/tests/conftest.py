import pytest

# Alias fixture for backward compatibility
@pytest.fixture
def tmp_output_dir(temp_output_dir):
    """Provide the same temporary output directory fixture name expected by older tests."""
    return temp_output_dir
