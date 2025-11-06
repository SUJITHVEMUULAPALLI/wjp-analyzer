import json
import os


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config", "token_budget.json")
RULES_PATH = os.path.join(os.path.dirname(__file__), "config", "model_switch_rules.json")


def _load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def choose_model(task: str):
    budget = _load_json(CONFIG_PATH)
    rules = _load_json(RULES_PATH)
    used = float(budget.get("used_today_usd", 0))
    limit = float(budget.get("daily_limit_usd", 5))
    near_limit = used >= 0.9 * limit

    if near_limit:
        for r in rules.get("rules", []):
            if r["task"] == task and not r["model"].startswith("local_"):
                return "gpt-4o"
        return "gpt-4o"

    for r in rules.get("rules", []):
        if r["task"] == task:
            return r["model"]
    return "gpt-5"


def update_budget(usd_spent: float):
    data = _load_json(CONFIG_PATH)
    data["used_today_usd"] = round(data.get("used_today_usd", 0) + usd_spent, 2)
    with open(CONFIG_PATH, "w") as f:
        json.dump(data, f, indent=2)


