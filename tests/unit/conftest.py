"""Pytest configuration for unit tests."""
import pytest
from unittest.mock import MagicMock

from forge.clients.base import LLMClient


@pytest.fixture
def mock_client():
    """Create a mock LLMClient."""
    client = MagicMock(spec=LLMClient)
    client.last_thinking = ""
    client.api_format = "ollama"
    client.last_usage = {}
    return client
