"""
Integration tests for Analyzer ↔ Editor workflow.

Tests session state handoff and edit workflows.
"""
from __future__ import annotations

import os
import pytest
import ezdxf
from pathlib import Path

from wjp_analyser.web.modules import dxf_editor_core as core
from wjp_analyser.web.modules import dxf_utils as du


@pytest.fixture(autouse=True)
def clean_session(monkeypatch):
    """Reset Streamlit session state before each test."""
    import streamlit as st
    st.session_state.clear()
    yield
    st.session_state.clear()


def test_analyzer_to_editor_handoff(sample_dxf):
    """Test that Analyzer can pass DXF path to Editor via session state."""
    import streamlit as st
    
    # Simulate Analyzer setting the path
    st.session_state["analyzed_dxf_path"] = sample_dxf
    
    # Editor loads from session state
    doc, path = core.load_from_analyzer_or_upload(st, uploaded_file=None)
    
    assert doc is not None
    assert os.path.exists(path)
    assert st.session_state[du.SESSION_DXF_KEY] is doc
    assert st.session_state[du.SESSION_PATH_KEY] == path


def test_editor_loads_svg_after_analyzer_handoff(sample_dxf):
    """Test that Editor can render SVG after loading from Analyzer."""
    import streamlit as st
    from wjp_analyser.web.modules import dxf_renderer as dr
    
    # Simulate Analyzer → Editor handoff
    st.session_state["analyzed_dxf_path"] = sample_dxf
    doc, _ = core.load_from_analyzer_or_upload(st, uploaded_file=None)
    
    # Editor should be able to render SVG
    svg_text = dr.render_svg(doc)
    assert isinstance(svg_text, str)
    assert "<svg" in svg_text


def test_edit_workflow_delete_save_reload(sample_dxf, tmp_path):
    """Test complete edit workflow: load → delete → save → reload."""
    import streamlit as st
    
    # 1. Load from Analyzer
    st.session_state["analyzed_dxf_path"] = sample_dxf
    doc1, initial_path = core.load_from_analyzer_or_upload(st, None)
    initial_count = len(list(doc1.modelspace()))
    assert initial_count > 0
    
    # 2. Delete an entity
    entity_table = core.get_entity_table(st)
    first_handle = entity_table[0]["handle"]
    deleted_count = core.apply_delete(st, [first_handle])
    assert deleted_count == 1
    assert len(list(doc1.modelspace())) == initial_count - 1
    
    # 3. Save edited DXF
    edited_path = tmp_path / "edited.dxf"
    saved_path = core.save_as(st, str(edited_path))
    assert os.path.exists(saved_path)
    
    # 4. Reload and verify
    st.session_state.clear()
    st.session_state["analyzed_dxf_path"] = str(edited_path)
    doc2, reloaded_path = core.load_from_analyzer_or_upload(st, None)
    reloaded_count = len(list(doc2.modelspace()))
    assert reloaded_count == initial_count - 1
    assert not any(e.dxf.handle == first_handle for e in list(doc2.modelspace()))


def test_edit_log_tracking(sample_dxf):
    """Test that edit log tracks all operations."""
    import streamlit as st
    
    st.session_state["analyzed_dxf_path"] = sample_dxf
    doc, _ = core.load_from_analyzer_or_upload(st, None)
    
    # Initial edit log should be empty
    assert len(st.session_state[du.SESSION_EDIT_LOG]) == 0
    
    # Delete entities
    entity_table = core.get_entity_table(st)
    handles_to_delete = [e["handle"] for e in entity_table[:2]]
    core.apply_delete(st, handles_to_delete)
    
    # Edit log should have one entry
    assert len(st.session_state[du.SESSION_EDIT_LOG]) == 1
    log_entry = st.session_state[du.SESSION_EDIT_LOG][0]
    assert log_entry["action"] == "delete"
    assert set(log_entry["handles"]) == set(handles_to_delete)


def test_layer_visibility_persistence(sample_dxf):
    """Test that layer visibility settings persist across operations."""
    import streamlit as st
    from wjp_analyser.web.modules import dxf_renderer as dr
    
    st.session_state["analyzed_dxf_path"] = sample_dxf
    doc, _ = core.load_from_analyzer_or_upload(st, None)
    
    # Get layers
    layers = du.list_layers(doc)
    assert len(layers) > 0
    
    # Set layer visibility
    vis = {layer: False for layer in layers}
    vis[layers[0]] = True  # Keep first layer visible
    st.session_state[du.SESSION_LAYER_VIS] = vis
    
    # Render with visibility settings
    svg_text = dr.render_svg(doc, layer_visibility=vis)
    assert "<svg" in svg_text
    
    # Visibility should persist in session state
    assert st.session_state[du.SESSION_LAYER_VIS] == vis


def test_multiple_edit_operations(sample_dxf):
    """Test that multiple edit operations are tracked correctly."""
    import streamlit as st
    
    st.session_state["analyzed_dxf_path"] = sample_dxf
    doc, _ = core.load_from_analyzer_or_upload(st, None)
    
    entity_table = core.get_entity_table(st)
    assert len(entity_table) >= 3
    
    # Perform multiple delete operations
    core.apply_delete(st, [entity_table[0]["handle"]])
    core.apply_delete(st, [entity_table[1]["handle"]])
    
    # Edit log should have 2 entries
    assert len(st.session_state[du.SESSION_EDIT_LOG]) == 2
    assert all(entry["action"] == "delete" for entry in st.session_state[du.SESSION_EDIT_LOG])


def test_editor_handles_missing_analyzer_path():
    """Test that Editor handles missing Analyzer path gracefully."""
    import streamlit as st
    
    # No path set
    doc, path = core.load_from_analyzer_or_upload(st, uploaded_file=None)
    
    assert doc is None
    assert path is None


def test_editor_prioritizes_upload_over_analyzer_path(sample_dxf, tmp_path):
    """Test that uploaded file takes priority over Analyzer path."""
    import streamlit as st
    from io import BytesIO
    
    # Set Analyzer path
    st.session_state["analyzed_dxf_path"] = sample_dxf
    
    # Create a different DXF file for upload
    upload_doc = ezdxf.new()
    upload_doc.modelspace().add_line((100, 100), (200, 200))
    upload_path = tmp_path / "upload.dxf"
    upload_doc.saveas(upload_path)
    
    with open(upload_path, "rb") as f:
        fake_upload = BytesIO(f.read())
        fake_upload.name = "upload.dxf"
        
        # Load with upload
        doc, path = core.load_from_analyzer_or_upload(st, fake_upload)
        
        # Should use uploaded file (different entity count)
        assert doc is not None
        assert len(list(doc.modelspace())) == 1  # Only the uploaded line
        assert "upload.dxf" in path

