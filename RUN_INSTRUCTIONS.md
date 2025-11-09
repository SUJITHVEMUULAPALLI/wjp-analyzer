# WJP ANALYSER - How to Run

## Quick Start

**Use the clean entry point:**

```bash
python run.py
```

This will start the Streamlit web interface at `http://127.0.0.1:8501`

## Options

```bash
# Custom port
python run.py --port 8080

# Don't open browser automatically
python run.py --no-browser

# Listen on all interfaces (for remote access)
python run.py --host 0.0.0.0
```

## Alternative: Direct Streamlit

You can also run Streamlit directly:

```bash
streamlit run src/wjp_analyser/web/streamlit_app.py
```

## Deprecated Files

The following files are deprecated but kept for backward compatibility:

- `run_one_click.py` - Use `run.py` instead
- `main.py` - Use `run.py` instead  
- `wjp_main_ui.py` - Standalone app, not the main entry point
- `app.py` - Flask app (legacy)

**Always use `run.py` as the main entry point.**



