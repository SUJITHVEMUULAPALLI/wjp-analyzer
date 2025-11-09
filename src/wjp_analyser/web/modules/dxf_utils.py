# modules/dxf_utils.py
from __future__ import annotations

import os, shutil, tempfile
import ezdxf
from typing import Dict, List, Tuple

SESSION_DXF_KEY = "wjp_dxf_doc"
SESSION_PATH_KEY = "wjp_dxf_path"
SESSION_EDIT_LOG = "wjp_edit_log"
SESSION_LAYER_VIS = "wjp_layer_visibility"
SESSION_SELECTED = "wjp_selected_handles"
SESSION_HISTORY = "wjp_dxf_history_manager"
SESSION_EDIT_COUNT = "wjp_edit_count"  # Track edits for auto re-analyze


def load_document(path: str) -> ezdxf.EzDxfDrawing:
    if not os.path.exists(path):
        raise FileNotFoundError(f"DXF not found: {path}")
    return ezdxf.readfile(path)


def save_document(doc: ezdxf.EzDxfDrawing, out_path: str) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    doc.saveas(out_path)
    return out_path


def temp_copy(path: str) -> str:
    os.makedirs(tempfile.gettempdir(), exist_ok=True)
    base = os.path.basename(path)
    dst = os.path.join(tempfile.gettempdir(), f"wjp_edit_{base}")
    shutil.copy2(path, dst)
    return dst


def list_layers(doc) -> List[str]:
    return [l.dxf.name for l in doc.layers]


def entity_summary(doc) -> List[Dict]:
    """Flatten modelspace entities with minimal safe metadata."""
    msp = doc.modelspace()
    data = []
    for e in msp:
        try:
            handle = e.dxf.handle
            etype = e.dxftype()
            layer = e.dxf.layer if hasattr(e.dxf, "layer") else "0"
            color = getattr(e.dxf, "color", None)
            data.append({
                "handle": handle,
                "type": etype,
                "layer": layer,
                "color": color,
            })
        except Exception:
            # Skip malformed entities
            continue
    return data


def delete_entities_by_handle(doc, handles: List[str]) -> int:
    msp = doc.modelspace()
    count = 0
    # Convert to set for speed
    hset = set(handles or [])
    for e in list(msp):  # snapshot to avoid iteration issues
        try:
            if e.dxf.handle in hset:
                msp.delete_entity(e)
                count += 1
        except Exception:
            continue
    return count



