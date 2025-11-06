from __future__ import annotations

import argparse
from wjp_analyser.web.app import app as flask_app


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Waterjet DXF Analyzer web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    flask_app.run(host=args.host, port=args.port, debug=False, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


