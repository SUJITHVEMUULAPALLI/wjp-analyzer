"""Interactive API blueprint for DXF entity operations.

Provides endpoints to move selected entities to a specific layer and
re-run analysis to refresh artifacts.
"""

from __future__ import annotations

import os
from typing import List, Dict, Any

from flask import Blueprint, request, jsonify

from ..analysis.dxf_analyzer import (
    AnalyzeArgs,
    analyze_dxf,
    relayer_entities_by_handle,
)


api_bp = Blueprint("interactive_api", __name__, url_prefix="/api")


@api_bp.post("/move_to_layer")
def move_to_layer() -> Any:
    data = request.get_json(silent=True) or {}
    dxf_path = data.get("dxf_path")
    handles: List[str] = data.get("handles") or []
    target_layer: str = data.get("target_layer") or ""
    out_dir = data.get("out_dir")
    sheet_w = float(data.get("sheet_width") or 1000.0)
    sheet_h = float(data.get("sheet_height") or 1000.0)

    if not dxf_path or not os.path.exists(dxf_path):
        return jsonify({"ok": False, "error": "dxf_path not found"}), 400
    if not handles:
        return jsonify({"ok": False, "error": "handles required"}), 400
    if not target_layer:
        return jsonify({"ok": False, "error": "target_layer required"}), 400

    # Prepare output directory
    if not out_dir:
        base_dir = os.path.dirname(os.path.abspath(dxf_path))
        out_dir = os.path.join(base_dir, "interactive_out")
    os.makedirs(out_dir, exist_ok=True)

    updated_dxf = relayer_entities_by_handle(dxf_path, handles, target_layer)

    args = AnalyzeArgs(out=out_dir)
    args.sheet_width = sheet_w
    args.sheet_height = sheet_h
    report = analyze_dxf(updated_dxf, args)

    return jsonify(
        {
            "ok": True,
            "updated_dxf": updated_dxf,
            "artifacts": report.get("artifacts", {}),
            "metrics": report.get("metrics", {}),
            "groups": report.get("groups", {}),
            "components": report.get("components", []),
        }
    )

