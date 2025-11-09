"""
Additional tests to achieve 100% coverage for dxf_editor_core.py.

Tests the remaining defensive code paths:
- Work dict path in load_from_analyzer_or_upload (lines 50-52)
"""
from __future__ import annotations

import os
import pytest
import ezdxf
from pathlib import Path

from wjp_analyser.web.modules import dxf_editor_core as core
from wjp_analyser.web.modules import dxf_utils as du


def test_load_from_analyzer_or_upload_with_work_dict(sample_dxf):
    """Test loading DXF from _wjp_work dict path (lines 50-52)."""
    import streamlit as st
    
    # Clear other potential keys
    st.session_state.clear()
    
    # Set up work dict with dxf_path
    st.session_state["_wjp_work"] = {"dxf_path": sample_dxf}
    
    doc, path = core.load_from_analyzer_or_upload(st, uploaded_file=None)
    
    assert doc is not None
    assert os.path.exists(path)
    assert st.session_state[du.SESSION_DXF_KEY] is not None


def test_load_from_analyzer_or_upload_work_dict_invalid_path():
    """Test work dict with invalid path returns None."""
    import streamlit as st
    
    st.session_state.clear()
    st.session_state["_wjp_work"] = {"dxf_path": "nonexistent.dxf"}
    
    doc, path = core.load_from_analyzer_or_upload(st, uploaded_file=None)
    
    assert doc is None
    assert path is None


def test_load_from_analyzer_or_upload_work_dict_no_dxf_path():
    """Test work dict without dxf_path key."""
    import streamlit as st
    
    st.session_state.clear()
    st.session_state["_wjp_work"] = {"other_key": "value"}
    
    doc, path = core.load_from_analyzer_or_upload(st, uploaded_file=None)
    
    assert doc is None
    assert path is None

