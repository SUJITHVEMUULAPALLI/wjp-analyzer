from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Dict, Iterable, Optional


def build_cache_key(file_hash: str, overrides: Optional[Dict[str, Any]] = None) -> str:
    """Create a deterministic cache key for DXF analysis reports."""
    normalized = overrides or {}
    payload = json.dumps(normalized, sort_keys=True, default=str).encode("utf-8")
    param_hash = hashlib.md5(payload).hexdigest()
    return f"analysis_{file_hash}_{param_hash}"


def filter_cached_report(
    cached_report: Dict[str, Any],
    selected_groups: Optional[Iterable[str]] = None,
    group_layer_overrides: Optional[Dict[str, str]] = None,
    logging_service: Any = None,
) -> Dict[str, Any]:
    """
    Return a filtered copy of the cached report based on selection overrides.

    Args:
        cached_report: Cached analysis report.
        selected_groups: Optional list of groups to keep.
        group_layer_overrides: Optional mapping of group -> target layer.
        logging_service: Optional logging interface.
    """
    if not cached_report:
        return cached_report

    report = deepcopy(cached_report)
    metrics = report.setdefault("metrics", {})
    components = report.get("components", [])

    if selected_groups:
        selected_groups_set = set(selected_groups)
        filtered_components = [
            comp for comp in components if comp.get("group") in selected_groups_set
        ]
        report["components"] = filtered_components
        report["selection"] = {
            "groups": list(selected_groups),
            "component_ids": [comp.get("id") for comp in filtered_components],
        }
        metrics.update({
            "total_area": sum(comp.get("area", 0.0) for comp in filtered_components),
            "total_perimeter": sum(comp.get("perimeter", 0.0) for comp in filtered_components),
            "object_count": len(filtered_components),
        })
        if logging_service:
            logging_service.debug(
                "Applied selection filter to cached results",
                {
                    "selected_groups": list(selected_groups),
                    "filtered_count": len(filtered_components),
                },
            )

    if group_layer_overrides:
        for comp in report.get("components", []):
            group_name = comp.get("group")
            if group_name in group_layer_overrides:
                comp["layer"] = group_layer_overrides[group_name]

        report["layers"] = {
            layer_name: len(
                [c for c in report.get("components", []) if c.get("layer") == layer_name]
            )
            for layer_name in cached_report.get("layers", {}).keys()
        }

    if selected_groups is not None and "selection" not in report:
        report["selection"] = {
            "groups": list(selected_groups),
            "component_ids": [
                comp.get("id")
                for comp in report.get("components", [])
                if comp.get("group") in set(selected_groups or [])
            ],
        }

    return report
