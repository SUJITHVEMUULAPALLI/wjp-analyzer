def compute_metrics(used_union, frame_poly, placed, cfg):
    frame_area = frame_poly.area if frame_poly else 1
    used_area = used_union.area if used_union else 0
    util = (used_area / frame_area) * 100.0
    total_cut_len_mm = used_union.length if used_union else 0
    pierces = len(placed)
    total_m = total_cut_len_mm / 1000.0
    cost = total_m * cfg.get('cost_per_meter', 825)
    return dict(
        utilization=round(util, 2),
        used_area_mm2=round(used_area, 1),
        frame_area_mm2=round(frame_area, 1),
        total_cut_len_mm=round(total_cut_len_mm, 1),
        pierces=pierces,
        est_cost_inr=round(cost)
    )
