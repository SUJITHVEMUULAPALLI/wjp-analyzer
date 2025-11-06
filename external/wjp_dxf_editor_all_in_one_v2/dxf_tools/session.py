import json
from pathlib import Path

DEFAULT_STATE = {
    "hidden_layers": [],
    "selected_handles": [],
    "grid_mm": 10.0,
    "units": "mm"
}

def load_session(path):
    p = Path(path)
    if not p.exists():
        return DEFAULT_STATE.copy()
    return json.loads(p.read_text())

def save_session(path, state):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(state, indent=2))
