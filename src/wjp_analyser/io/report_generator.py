import json, csv, os
from typing import Any, Dict, List

def save_json(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def save_lengths_csv(path: str, rows: list[dict]):
    keys = rows[0].keys() if rows else ["id","class","perimeter_mm","area_mm2"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
