#!/usr/bin/env python3
"""Launch the Waterjet DXF Analyzer web UI and open it in a browser."""

import argparse
import threading
import time
import webbrowser
from urllib.error import URLError
from urllib.request import urlopen

from app import app as flask_app


def wait_and_open(url: str, timeout: float = 30.0) -> None:
    """Ping the given URL until it responds, then open it in the browser."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urlopen(url):  # nosec - local dev server
                pass
        except URLError:
            time.sleep(0.5)
            continue
        except Exception:
            # Other transient errors (e.g. connection reset); retry
            time.sleep(0.5)
            continue
        try:
            webbrowser.open(url)
            print(f"Opened browser at {url}")
        except Exception as exc:  # pragma: no cover - best effort
            print(f"Could not open browser automatically: {exc}")
        return
    print(f"Server did not respond within {timeout:.0f}s. Open manually: {url}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Waterjet DXF Analyzer web UI")
    parser.add_argument("--host", default="127.0.0.1", help="Host/interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Port to serve on (default: 5000)")
    parser.add_argument("--no-browser", action="store_true", help="Skip opening the browser automatically")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    url = f"http://{args.host}:{args.port}/"

    print("Starting Waterjet DXF Analyzer web UI...")
    print(f"Listening on {url}")

    if not args.no_browser:
        opener = threading.Thread(target=wait_and_open, args=(url,), daemon=True)
        opener.start()

    try:
        flask_app.run(host=args.host, port=args.port, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        print("\nStopping web server...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
