"""Integration test fixtures for Streamlit UI testing."""
from __future__ import annotations

import pytest
import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture
def mock_streamlit_session():
    """Mock Streamlit session state for testing."""
    class MockSessionState:
        def __init__(self):
            self._state = {}
        
        def clear(self):
            self._state.clear()
        
        def __getitem__(self, key):
            return self._state[key]
        
        def __setitem__(self, key, value):
            self._state[key] = value
        
        def __contains__(self, key):
            return key in self._state
        
        def get(self, key, default=None):
            return self._state.get(key, default)
        
        def setdefault(self, key, default):
            if key not in self._state:
                self._state[key] = default
            return self._state[key]
        
        def pop(self, key, default=None):
            return self._state.pop(key, default)
    
    return MockSessionState()

