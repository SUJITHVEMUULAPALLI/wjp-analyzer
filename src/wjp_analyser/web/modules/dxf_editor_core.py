# modules/dxf_editor_core.py
from __future__ import annotations
import os
import tempfile
from typing import Dict, List
from .dxf_utils import (
    SESSION_DXF_KEY, SESSION_PATH_KEY, SESSION_EDIT_LOG, SESSION_LAYER_VIS, SESSION_SELECTED,
    load_document, save_document, temp_copy, list_layers, entity_summary, delete_entities_by_handle
)


def ensure_session_keys(st):
    for k, default in [
        (SESSION_DXF_KEY, None),
        (SESSION_PATH_KEY, None),
        (SESSION_EDIT_LOG, []),
        (SESSION_LAYER_VIS, {}),
        (SESSION_SELECTED, set()),
    ]:
        st.session_state.setdefault(k, default)


def load_from_analyzer_or_upload(st, uploaded_file):
    """
    Priority: session path from Analyzer → uploaded file
    Analyzer should store path under various keys:
    - 'analyzed_dxf_path' (recommended)
    - '_wjp_scaled_dxf_path' (from analyze_dxf page)
    - 'wjp_dxf_path' (alternative)
    - work['dxf_path'] (from ensure_workdir)
    """
    ensure_session_keys(st)
    analyzer_key_candidates = [
        "analyzed_dxf_path",
        "_wjp_scaled_dxf_path",
        SESSION_PATH_KEY,
        "wjp_dxf_path"
    ]
    path = None
    for k in analyzer_key_candidates:
        p = st.session_state.get(k)
        if p and os.path.exists(str(p)):
            path = str(p)
            break
    
    # Also check work dict if available
    if not path:
        work = st.session_state.get("_wjp_work", {})
        if work and "dxf_path" in work:
            p = work["dxf_path"]
            if p and os.path.exists(str(p)):
                path = str(p)
    if uploaded_file is not None:
        # Save uploaded file to tmp and use that
        tmp = os.path.join(tempfile.gettempdir(), uploaded_file.name)
        with open(tmp, "wb") as f:
            f.write(uploaded_file.getbuffer())
        path = tmp

    if not path:
        return None, None

    # Work on a temp copy so original is preserved until user hits Save
    working_path = temp_copy(path)
    doc = load_document(working_path)
    st.session_state[SESSION_DXF_KEY] = doc
    st.session_state[SESSION_PATH_KEY] = working_path

    # Initialize layer visibility if empty
    if not st.session_state[SESSION_LAYER_VIS]:
        vis = {name: True for name in list_layers(doc)}
        st.session_state[SESSION_LAYER_VIS] = vis

    return doc, working_path


def get_entity_table(st) -> List[Dict]:
    doc = st.session_state.get(SESSION_DXF_KEY)
    if not doc:
        return []
    return entity_summary(doc)


def apply_delete(st, handles: List[str]) -> int:
    doc = st.session_state.get(SESSION_DXF_KEY)
    if not doc or not handles:
        return 0
    count = delete_entities_by_handle(doc, handles)
    if count:
        st.session_state[SESSION_EDIT_LOG].append({"action": "delete", "handles": handles})
    return count


def save_as(st, out_path: str) -> str:
    doc = st.session_state.get(SESSION_DXF_KEY)
    if not doc:
        raise RuntimeError("No DXF loaded")
    return save_document(doc, out_path)

