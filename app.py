import os
import json
from flask import Flask, request, jsonify
from openai import OpenAI

# Import analyzer from package
from src.wjp_analyser.analysis.dxf_analyzer import analyze_dxf  # type: ignore


app = Flask(__name__)

def _load_api_key() -> str | None:
    """Load OpenAI API key from env or fallback file.

    Order:
    1) Env var OPENAI_API_KEY
    2) File C:\\WJP ANALYSER\\wjp.env.txt (either RAW key or KEY=VALUE form)
    """
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key
    # Fallback file path (Windows-safe)
    fallback_path = os.path.join("C:\\", "WJP ANALYSER", "wjp.env.txt")
    try:
        if os.path.exists(fallback_path):
            with open(fallback_path, "r", encoding="utf-8") as f:
                txt = f.read().strip()
            # Support KEY=VALUE or just VALUE
            if "OPENAI_API_KEY=" in txt:
                for line in txt.splitlines():
                    line = line.strip()
                    if line.startswith("OPENAI_API_KEY="):
                        return line.split("=", 1)[1].strip().strip('"')
            # Otherwise assume whole file is the key
            if txt:
                return txt.strip().strip('"')
    except Exception:
        pass
    return None

# Initialize OpenAI client using loader
OPENAI_API_KEY = _load_api_key()
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Allow model override via env; pick a sensible default
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")


def _error(message: str, status: int = 400):
    return jsonify({"error": message}), status


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "openai": bool(client),
        "chat_model": CHAT_MODEL,
        "image_model": IMAGE_MODEL,
    })


def _summarize_report(report: dict, component_limit: int = 50) -> dict:
    """Return a compact summary of the analyzer report to keep token usage low.
    Removes large geometry arrays and keeps only essential metrics.
    """
    material = report.get("material")
    if isinstance(material, dict):
        mat_name = material.get("name", "Unknown")
        thickness = material.get("thickness_mm", 0)
    else:
        mat_name, thickness = material or "Unknown", 0

    metrics = report.get("metrics", {})
    components = report.get("components", [])
    layers = report.get("layers", {})
    groups = report.get("groups", {})
    quality = report.get("quality", {})

    comp_sample = []
    for comp in components[:max(0, int(component_limit))]:
        comp_sample.append({
            "id": comp.get("id"),
            "group": comp.get("group"),
            "layer": comp.get("layer"),
            "area": round(float(comp.get("area", 0)), 3),
            "perimeter": round(float(comp.get("perimeter", 0)), 3),
            "vertices": len(comp.get("points", [])),
        })

    return {
        "file": report.get("file"),
        "material": mat_name,
        "thickness_mm": thickness,
        "kerf_mm": report.get("kerf_mm"),
        "metrics": {
            "length_internal_mm": metrics.get("length_internal_mm", 0),
            "length_outer_mm": metrics.get("length_outer_mm", 0),
            "estimated_cutting_cost_inr": metrics.get("estimated_cutting_cost_inr", 0),
            "pierce_count": metrics.get("pierce_count", 0),
        },
        "layers": {k: int(v) for k, v in (layers or {}).items()},
        "groups_count": len(groups or {}),
        "components_total": len(components or []),
        "components_sample": comp_sample,
        "quality": quality or {},
    }


def _cap_summary_size(summary: dict, max_chars: int = 40000) -> dict:
    """Ensure the serialized summary stays under max_chars by truncating samples.

    This focuses on shrinking components_sample; if still too large, drops quality.
    """
    try:
        s = json.dumps(summary)
    except Exception:
        return summary

    if len(s) <= max_chars:
        return summary

    summary = dict(summary)  # shallow copy
    sample = list(summary.get("components_sample", []))

    # Iteratively reduce sample size
    while len(json.dumps({**summary, "components_sample": sample})) > max_chars and len(sample) > 5:
        # keep first half
        sample = sample[: max(5, len(sample) // 2)]

    summary["components_sample"] = sample

    # As a last resort, drop quality block
    if len(json.dumps(summary)) > max_chars:
        summary.pop("quality", None)

    return summary


def _chat_with_optional_json(model: str, messages: list, max_tokens: int = 700, temperature: float = 0.4):
    """Call chat.completions with response_format=json_object when supported; fallback otherwise."""
    try:
        return client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        # Retry without response_format if unsupported
        if "response_format" in str(e).lower():
            return client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        raise


# -------------------------------
# 1. Designing Module
# -------------------------------
@app.route('/design', methods=['POST'])
def design():
    """
    Generate waterjet-compatible design image based on user template.
    Input: JSON body with design_type, material, dimensions, style
    Output: Generated image URL + prompt
    """
    if not client:
        return _error("OpenAI client not initialized. Set OPENAI_API_KEY.", 500)

    try:
        data = request.get_json(force=True, silent=False) or {}
    except Exception:
        return _error("Invalid JSON body.")

    design_type = data.get("design_type", "tile")
    material = data.get("material", "Generic stone")
    dimensions = data.get("dimensions", "600x600 mm")
    style = data.get("style", "geometric")

    prompt = (
        f"Generate a waterjet-compatible {design_type} design using {material}, "
        f"with dimensions {dimensions}, in a {style} style. "
        "Waterjet constraints: ≥3 mm spacing, ≥2 mm inner radius curves, "
        "no floating parts, clean geometry, continuous contours."
    )

    try:
        def _do_gen(model_name: str):
            return client.images.generate(
                model=model_name,
                prompt=prompt,
                size="1024x1024",
            )

        try:
            result = _do_gen(IMAGE_MODEL)
        except Exception as e:
            # Fallback to dall-e-3 if gpt-image-1 is unavailable for the org
            fallback_model = "dall-e-3"
            try:
                result = _do_gen(fallback_model)
            except Exception:
                raise e
        image_url = None
        # Prefer URL if present; else use base64 payload
        if getattr(result.data[0], 'url', None):
            image_url = result.data[0].url
            payload = {"prompt": prompt, "image_url": image_url}
        else:
            b64 = getattr(result.data[0], 'b64_json', None)
            if not b64:
                return _error("Image generation returned no URL or base64 data.", 502)
            # Save to uploads and return file path
            os.makedirs('uploads', exist_ok=True)
            out_path = os.path.join('uploads', 'design_output.png')
            import base64
            with open(out_path, 'wb') as f:
                f.write(base64.b64decode(b64))
            payload = {"prompt": prompt, "image_path": out_path}
    except Exception as e:
        return _error(f"OpenAI image generation failed: {e}", 502)

    return jsonify(payload)


# -------------------------------
# 2. Insights Module
# -------------------------------
@app.route('/insights', methods=['POST'])
def insights():
    # Legacy AI Insights endpoint removed per UI restructuring.
    return _error("Insights API has been removed.", 410)
    """
    Evaluate DXF analysis results using GPT.
    Input: DXF file (form field: dxf_file)
    Output: Analyzer report + AI insights
    """
    if 'dxf_file' not in request.files:
        return _error("Missing file field 'dxf_file'.")

    dxf_file = request.files['dxf_file']
    if not dxf_file.filename:
        return _error("Empty filename for uploaded file.")

    os.makedirs("uploads", exist_ok=True)
    os.makedirs("out", exist_ok=True)
    # Sanitize and use absolute path
    fname = os.path.basename(dxf_file.filename)
    dxf_path = os.path.abspath(os.path.join("uploads", fname))
    try:
        dxf_file.save(dxf_path)
    except Exception as e:
        return _error(f"Failed to save upload: {e}", 500)

    # Run analyzer (returns dict report)
    try:
        # Some analyzers write to stdout and certain Windows hosts can raise
        # OSError on print; safely capture stdout during analysis.
        import io, contextlib
        _buf = io.StringIO()
        with contextlib.redirect_stdout(_buf):
            report = analyze_dxf(dxf_path)
    except Exception as e:
        import traceback
        return jsonify({
            "error": f"DXF analysis failed: {e}",
            "trace": traceback.format_exc(),
            "path": dxf_path,
        }), 500

    # Send report to GPT for evaluation (optional if API key configured)
    insights_text = None
    if client:
        # Use compact summary to avoid token overuse
        summary = _summarize_report(report, component_limit=50)
        summary = _cap_summary_size(summary, max_chars=40000)
        ai_prompt = (
            "You are a Waterjet Analyzer expert. Using ONLY the summarized report below, "
            "provide:\n"
            "1) Violations summary (open contours, spacing, radius)\n"
            "2) Suggested corrections\n"
            "3) Optimization opportunities (cutting order, nesting, cost)\n\n"
            f"Summary Report (compact):\n{json.dumps(summary, indent=2)}\n"
        )
        try:
            try:
                result = _chat_with_optional_json(
                    CHAT_MODEL,
                    messages=[
                        {"role": "system", "content": "You are an expert in CNC waterjet cutting analysis. Respond in concise, clear language."},
                        {"role": "user", "content": ai_prompt},
                    ],
                    max_tokens=700,
                    temperature=0.4,
                )
            except Exception as e:
                # If rate-limited by TPM, retry with smaller sample
                if 'rate_limit' in str(e).lower() or 'tokens per min' in str(e).lower():
                    summary = _cap_summary_size(_summarize_report(report, component_limit=20), max_chars=25000)
                    ai_prompt = (
                        "You are a Waterjet Analyzer expert. Provide insights based on this compact summary only.\n\n"
                        f"Summary Report (more compact):\n{json.dumps(summary, indent=2)}\n"
                    )
                    result = _chat_with_optional_json(
                        CHAT_MODEL,
                        messages=[
                            {"role": "system", "content": "You are an expert in CNC waterjet cutting analysis. Respond in concise, clear language."},
                            {"role": "user", "content": ai_prompt},
                        ],
                        max_tokens=600,
                        temperature=0.4,
                    )
                else:
                    raise
            insights_text = result.choices[0].message.content
        except Exception as e:
            # Fall back gracefully if AI call fails
            insights_text = f"AI insights unavailable: {e}"
    else:
        insights_text = "OpenAI not configured. Set OPENAI_API_KEY to enable AI insights."

    return jsonify({
        "report": report,
        "insights": insights_text,
    })


def _extract_json(text: str):
    """Attempt to parse a JSON object from arbitrary model output.
    Handles fenced code blocks and best-effort brace extraction.
    """
    import re, json as _json
    if not isinstance(text, str):
        return None
    s = text.strip()
    # Strip triple backtick fences
    if s.startswith("```"):
        lines = [ln for ln in s.splitlines() if not ln.strip().startswith("```")]
        s = "\n".join(lines).strip()
    # Try direct JSON first
    try:
        return _json.loads(s)
    except Exception:
        pass
    # Try to find a top-level { ... }
    si = s.find('{')
    ei = s.rfind('}')
    if si != -1 and ei != -1 and ei > si:
        snippet = s[si:ei + 1]
        try:
            return _json.loads(snippet)
        except Exception:
            return None
    return None


@app.route('/', methods=['GET'])
def index():
    from flask import render_template
    return render_template('index.html', openai_ready=bool(client))


@app.route('/insights_dashboard', methods=['GET'])
def insights_dashboard():
    # Legacy route removed. Keep minimal placeholder to avoid 404s in older bookmarks.
    from flask import render_template
    return render_template('index.html', openai_ready=bool(client))

if __name__ == '__main__':
    # Bind to 0.0.0.0 for container/WSL friendliness
    port = int(os.getenv("PORT", "5000"))
    app.run(debug=True, host="0.0.0.0", port=port)
