import math
from shapely.geometry import box
from shapely.ops import unary_union
from .geometry import padded, can_place, rotate, translate
from .metrics import compute_metrics

def nest(objects, frame, cfg):
    frame_poly = box(0, 0, frame['width'], frame['height']).buffer(-frame['margin'])
    objs = sorted(objects, key=lambda o: o['area'], reverse=True)
    best = None
    for _ in range(cfg.get('retry_rounds', 3)):
        placed, placed_collision, used_union = [], None, None
        for o in objs:
            placed_ok = False
            for r in range(cfg.get('max_rotations', 12)):
                angle = r * cfg.get('rotation_step_deg', 15)
                rot_poly = rotate(o['polygon'], angle)
                step = max(5.0, min(o['bbox'][2]-o['bbox'][0], o['bbox'][3]-o['bbox'][1]) * 0.25)
                y = 0.0
                while y <= frame['height']:
                    x = 0.0
                    while x <= frame['width']:
                        cand = translate(rot_poly, x, y)
                        pad = padded(cand, (cfg['min_gap_mm'] + cfg['kerf_mm']) / 2.0)
                        if placed_collision is None:
                            ok = frame_poly.contains(pad)
                        else:
                            ok = can_place(frame_poly, placed_collision, pad)
                        if ok:
                            placed.append(dict(handle=o['handle'], rotation=angle, x=x, y=y))
                            placed_collision = pad if placed_collision is None else unary_union([placed_collision, pad])
                            used_union = cand if used_union is None else unary_union([used_union, cand])
                            placed_ok = True
                            break
                        x += step
                    if placed_ok: break
                    y += step
                if placed_ok: break
        if used_union:
            m = compute_metrics(used_union, frame_poly, placed, cfg)
            if not best or m['utilization'] > best['metrics']['utilization']:
                best = dict(placed=placed, metrics=m)
    return best
