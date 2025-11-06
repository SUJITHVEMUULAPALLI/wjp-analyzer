def check_min_radius(entity, min_r=2.0):
    if entity.dxftype() == "CIRCLE":
        return entity.dxf.radius >= min_r, {"type": "radius", "value": entity.dxf.radius, "min": min_r}
    return True, {"type": "radius", "note": "not-applicable"}


def kerf_preview_value(kerf=1.1):
    return float(kerf)





