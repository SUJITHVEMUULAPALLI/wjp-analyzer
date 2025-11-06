from flask import Flask, request, jsonify
import subprocess
import json
import os
import sys
from model_switcher import choose_model, update_budget


app = Flask(__name__)


@app.route("/mcp", methods=["POST"])
def handle_mcp():
    data = request.get_json(force=True)
    tool = data.get("tool")
    args = data.get("args", {})

    handlers = {
        "run_tests": lambda: run_tests(args.get("file_path", "")),
        "lint_file": lambda: lint_file(args.get("file_path", "")),
        "budget_check": read_budget,
        "choose_model": lambda: {
            "task": args.get("task", "generic"),
            "chosen_model": choose_model(args.get("task", "generic")),
        },
    }
    return jsonify(handlers.get(tool, lambda: {"error": "unknown tool"})())


def run_tests(path):
    try:
        # Ensure the test run can import local package modules under ./src
        env = os.environ.copy()
        project_root = os.path.dirname(os.path.dirname(__file__))
        existing = env.get("PYTHONPATH", "")
        base_paths = [project_root, os.path.join(project_root, "src")]
        env["PYTHONPATH"] = os.pathsep.join(base_paths + ([existing] if existing else []))

        project_root = project_root

        # Ensure the package is importable for both `wjp_analyser` and `src.*` style imports
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", "."],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
            cwd=project_root,
        )

        # Write a .pth file into site-packages to make project_root discoverable as a top-level "src" package
        try:
            import sysconfig
            site_packages = sysconfig.get_paths().get("purelib")
            if site_packages and os.path.isdir(site_packages):
                pth_file = os.path.join(site_packages, "wjp_src_root.pth")
                with open(pth_file, "w", encoding="utf-8") as f:
                    f.write(project_root + "\n")
        except Exception:
            pass
        result = subprocess.run(
            ["pytest", path, "--maxfail=1", "--disable-warnings", "-q"],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
            cwd=project_root,
        )
        update_budget(0.05)
        return {"status": "ok", "output": result.stdout.strip()}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def lint_file(path):
    try:
        result = subprocess.run(
            ["flake8", path, "--max-line-length=120"],
            capture_output=True,
            text=True,
        )
        update_budget(0.02)
        return {"lint_output": result.stdout.strip() or "No issues"}
    except Exception as e:
        return {"error": str(e)}


def read_budget():
    try:
        with open("wjp_upgrade/config/token_budget.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"error": "token_budget.json not found"}


if __name__ == "__main__":
    port = int(os.environ.get("MCP_PORT", "8000"))
    app.run(port=port)


