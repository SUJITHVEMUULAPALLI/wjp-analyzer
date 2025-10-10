import os, json, datetime


def ensure_dirs(paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    return path


def append_log(entry, path="logs/agent_log.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    entry["time"] = str(datetime.datetime.now())
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


