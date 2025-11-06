import json
import time
import os
from datetime import datetime


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config", "token_budget.json")


def reset_daily_budget():
    data = {
        "daily_limit_usd": 5,
        "used_today_usd": 0,
        "note": f"Auto reset {datetime.now().date()}",
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print("✅ Daily GPT budget reset")


if __name__ == "__main__":
    while True:
        now = datetime.now()
        if now.hour == 0 and now.minute == 0:
            reset_daily_budget()
            time.sleep(60)
        time.sleep(30)


