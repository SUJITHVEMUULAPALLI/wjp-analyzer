"""Console entry point for launching the web UI.

Provides an importable entry so packaging can expose `wjdx-web`.
"""

from run_web_ui import main as _main


def main() -> int:
    return _main()

