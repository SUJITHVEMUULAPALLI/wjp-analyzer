"""
Tests for DXF Editor core module (dxf_editor_core).

Tests session state management, document loading, entity operations, and save functionality.
"""
from __future__ import annotations

import os
import pytest
import ezdxf
from io import BytesIO
from pathlib import Path

# Import the modules under test
from wjp_analyser.web.modules import dxf_editor_core as core
from wjp_analyser.web.modules import dxf_utils as du


@pytest.fixture(autouse=True)
def clean_session(monkeypatch):
    """Reset Streamlit session state before each test."""
    # Mock streamlit session state
    import streamlit as st
    
    # Clear session state before each test
    if hasattr(st, 'session_state'):
        st.session_state.clear()
    else:
        # Create a mock session state if it doesn't exist
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
        
        st.session_state = MockSessionState()
    
    yield
    
    # Clean up after test
    if hasattr(st, 'session_state'):
        st.session_state.clear()


def test_ensure_session_keys_initializes_defaults():
    """Test that ensure_session_keys initializes all required session state keys."""
    import streamlit as st
    core.ensure_session_keys(st)
    
    for key in [
        du.SESSION_DXF_KEY,
        du.SESSION_PATH_KEY,
        du.SESSION_EDIT_LOG,
        du.SESSION_LAYER_VIS,
        du.SESSION_SELECTED,
    ]:
        assert key in st.session_state


def test_ensure_session_keys_sets_correct_defaults():
    """Test that ensure_session_keys sets correct default values."""
    import streamlit as st
    core.ensure_session_keys(st)
    
    assert st.session_state[du.SESSION_DXF_KEY] is None
    assert st.session_state[du.SESSION_PATH_KEY] is None
    assert isinstance(st.session_state[du.SESSION_EDIT_LOG], list)
    assert isinstance(st.session_state[du.SESSION_LAYER_VIS], dict)
    assert isinstance(st.session_state[du.SESSION_SELECTED], set)


def test_load_from_analyzer_or_upload_with_session_path(sample_dxf):
    """Test loading DXF from analyzer session state path."""
    import streamlit as st
    st.session_state["analyzed_dxf_path"] = sample_dxf
    
    doc, path = core.load_from_analyzer_or_upload(st, uploaded_file=None)
    
    assert doc is not None
    assert os.path.exists(path)
    assert isinstance(st.session_state[du.SESSION_LAYER_VIS], dict)
    assert st.session_state[du.SESSION_DXF_KEY] is not None
    assert st.session_state[du.SESSION_PATH_KEY] is not None


def test_load_from_analyzer_or_upload_with_alternative_keys(sample_dxf):
    """Test loading DXF from alternative session state keys."""
    import streamlit as st
    
    # Test _wjp_scaled_dxf_path
    st.session_state["_wjp_scaled_dxf_path"] = sample_dxf
    doc, path = core.load_from_analyzer_or_upload(st, uploaded_file=None)
    assert doc is not None
    assert os.path.exists(path)
    st.session_state.clear()
    
    # Test wjp_dxf_path
    st.session_state["wjp_dxf_path"] = sample_dxf
    doc, path = core.load_from_analyzer_or_upload(st, uploaded_file=None)
    assert doc is not None
    assert os.path.exists(path)


def test_load_from_analyzer_or_upload_with_upload(tmp_path):
    """Test loading DXF from uploaded file."""
    import streamlit as st
    
    # Create a test DXF file
    src = tmp_path / "upload.dxf"
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()
    msp.add_line((0, 0), (1, 1))
    msp.add_circle((5, 5), 2)
    doc.saveas(str(src))
    
    # Simulate file upload
    with open(src, "rb") as f:
        fake = BytesIO(f.read())
        fake.name = "upload.dxf"
        
        doc2, path = core.load_from_analyzer_or_upload(st, fake)
        
        assert doc2 is not None
        assert os.path.exists(path)
        assert st.session_state[du.SESSION_DXF_KEY] is not None


def test_load_from_analyzer_or_upload_no_file():
    """Test loading when no file is available."""
    import streamlit as st
    
    doc, path = core.load_from_analyzer_or_upload(st, uploaded_file=None)
    
    assert doc is None
    assert path is None


def test_get_entity_table(sample_dxf):
    """Test getting entity table from loaded document."""
    import streamlit as st
    st.session_state["analyzed_dxf_path"] = sample_dxf
    
    doc, _ = core.load_from_analyzer_or_upload(st, None)
    data = core.get_entity_table(st)
    
    assert isinstance(data, list)
    assert len(data) >= 1
    
    # Check structure of first entity
    if data:
        entity = data[0]
        assert "handle" in entity
        assert "type" in entity
        assert "layer" in entity
        assert "color" in entity


def test_get_entity_table_no_document():
    """Test getting entity table when no document is loaded."""
    import streamlit as st
    
    data = core.get_entity_table(st)
    
    assert isinstance(data, list)
    assert len(data) == 0


def test_apply_delete(sample_dxf):
    """Test deleting entities by handle."""
    import streamlit as st
    st.session_state["analyzed_dxf_path"] = sample_dxf
    
    doc, _ = core.load_from_analyzer_or_upload(st, None)
    data = core.get_entity_table(st)
    
    assert len(data) >= 1
    initial_count = len(data)
    
    # Delete first entity
    first_handle = data[0]["handle"]
    count = core.apply_delete(st, [first_handle])
    
    assert count == 1
    
    # Check edit log
    edit_log = st.session_state[du.SESSION_EDIT_LOG]
    assert len(edit_log) > 0
    assert any("delete" in a.get("action", "") for a in edit_log)
    
    # Verify entity count decreased
    new_data = core.get_entity_table(st)
    assert len(new_data) == initial_count - 1


def test_apply_delete_multiple(sample_dxf):
    """Test deleting multiple entities."""
    import streamlit as st
    st.session_state["analyzed_dxf_path"] = sample_dxf
    
    doc, _ = core.load_from_analyzer_or_upload(st, None)
    data = core.get_entity_table(st)
    
    if len(data) >= 2:
        handles = [data[0]["handle"], data[1]["handle"]]
        count = core.apply_delete(st, handles)
        
        assert count == 2
        assert len(st.session_state[du.SESSION_EDIT_LOG]) > 0


def test_apply_delete_no_handles(sample_dxf):
    """Test deleting with empty handle list."""
    import streamlit as st
    st.session_state["analyzed_dxf_path"] = sample_dxf
    
    doc, _ = core.load_from_analyzer_or_upload(st, None)
    count = core.apply_delete(st, [])
    
    assert count == 0


def test_apply_delete_no_document():
    """Test deleting when no document is loaded."""
    import streamlit as st
    
    count = core.apply_delete(st, ["some_handle"])
    
    assert count == 0


def test_save_as_creates_file(sample_dxf, tmp_path):
    """Test saving document to file."""
    import streamlit as st
    st.session_state["analyzed_dxf_path"] = sample_dxf
    
    doc, _ = core.load_from_analyzer_or_upload(st, None)
    out_path = tmp_path / "edited.dxf"
    
    saved_path = core.save_as(st, str(out_path))
    
    assert os.path.exists(saved_path)
    assert saved_path == str(out_path)
    
    # Verify we can load the saved file
    loaded_doc = du.load_document(saved_path)
    assert loaded_doc is not None


def test_save_as_no_document(tmp_path):
    """Test saving when no document is loaded."""
    import streamlit as st
    
    out_path = tmp_path / "edited.dxf"
    
    with pytest.raises(RuntimeError, match="No DXF loaded"):
        core.save_as(st, str(out_path))


def test_load_from_analyzer_or_upload_initializes_layer_visibility(sample_dxf):
    """Test that layer visibility is initialized when loading document."""
    import streamlit as st
    st.session_state["analyzed_dxf_path"] = sample_dxf
    
    doc, _ = core.load_from_analyzer_or_upload(st, None)
    
    layer_vis = st.session_state[du.SESSION_LAYER_VIS]
    assert isinstance(layer_vis, dict)
    assert len(layer_vis) > 0  # Should have at least default layer
    
    # All layers should be visible by default
    for layer_name, is_visible in layer_vis.items():
        assert isinstance(is_visible, bool)


def test_load_from_analyzer_or_upload_creates_temp_copy(sample_dxf):
    """Test that loading creates a temporary working copy."""
    import streamlit as st
    st.session_state["analyzed_dxf_path"] = sample_dxf
    
    doc, working_path = core.load_from_analyzer_or_upload(st, None)
    
    # Working path should be different from original
    assert working_path != sample_dxf
    assert os.path.exists(working_path)
    
    # Original file should still exist
    assert os.path.exists(sample_dxf)


# Integration test: Full workflow
def test_integration_load_delete_save_reload(sample_dxf, tmp_path):
    """Integration test: Load → Delete entity → Save → Reload → Verify count reduced."""
    import streamlit as st
    
    # Step 1: Load document
    st.session_state["analyzed_dxf_path"] = sample_dxf
    doc1, _ = core.load_from_analyzer_or_upload(st, None)
    initial_data = core.get_entity_table(st)
    initial_count = len(initial_data)
    
    assert initial_count >= 1
    
    # Step 2: Delete an entity
    first_handle = initial_data[0]["handle"]
    deleted_count = core.apply_delete(st, [first_handle])
    assert deleted_count == 1
    
    # Step 3: Save the modified document
    out_path = tmp_path / "modified.dxf"
    saved_path = core.save_as(st, str(out_path))
    assert os.path.exists(saved_path)
    
    # Step 4: Reload the saved document
    st.session_state.clear()
    st.session_state["analyzed_dxf_path"] = saved_path
    doc2, _ = core.load_from_analyzer_or_upload(st, None)
    final_data = core.get_entity_table(st)
    final_count = len(final_data)
    
    # Step 5: Verify entity count was reduced
    assert final_count == initial_count - 1
    
    # Verify the deleted handle is not in the new document
    handles_after = {e["handle"] for e in final_data}
    assert first_handle not in handles_after

