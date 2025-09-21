#!/usr/bin/env python3
"""
Waterjet DXF Analyzer - Web Interface
=====================================

Flask application for uploading DXF files or raster images, converting images
into DXF via contour extraction, and displaying analysis results.
"""

import json
import os
import sys
import uuid
from typing import Dict, Tuple

from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from werkzeug.utils import secure_filename

# Ensure local imports resolve correctly when running as a script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cli.wjdx import Args, analyze
from waterjet_dxf.image_processor import ImageProcessor

app = Flask(__name__)
app.secret_key = "waterjet_analyzer_secret_key_2024"

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "web_output"
ALLOWED_EXTENSIONS = {"dxf", "jpg", "jpeg", "png", "bmp", "tiff"}

DEFAULT_FORM = {
    "material": "Tan Brown Granite",
    "thickness": 25.0,
    "kerf": 1.1,
    "rate_per_m": 825.0,
}

DEFAULT_IMAGE_PARAMS = {
    "edge_threshold": 0.33,
    "min_contour_area": 100,
    "simplify_tolerance": 0.02,
    "blur_kernel_size": 5,
    "scale_mm_per_px": 0.5,
}

ANALYSIS_FILES = {
    "report": "report.json",
    "lengths": "lengths.csv",
    "preview": "preview.png",
    "gcode": "program.nc",
    "image_metadata": "image_metadata.json",
    "image_gray": "image_gray.png",
    "image_threshold": "image_threshold.png",
    "image_edges": "image_edges.png",
}

# Ensure required folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)


def allowed_file(filename: str) -> bool:
    """Check if the incoming filename has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def parse_float(raw: str, default: float) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def parse_int(raw: str, default: int) -> int:
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return default


def normalise_image_params(params: Dict[str, float]) -> Dict[str, float]:
    """Clamp image parameters to sensible ranges."""
    clamped = dict(params)
    clamped["edge_threshold"] = max(0.01, min(clamped["edge_threshold"], 0.99))
    clamped["min_contour_area"] = max(10, int(clamped["min_contour_area"]))
    clamped["simplify_tolerance"] = max(0.0001, clamped["simplify_tolerance"])
    # Force blur kernel size to be odd and at least 3
    blur = max(3, int(clamped["blur_kernel_size"]))
    clamped["blur_kernel_size"] = blur | 1
    # Scale: allow small decimals but cap to avoid absurd values
    clamped["scale_mm_per_px"] = float(max(0.01, min(params.get("scale_mm_per_px", DEFAULT_IMAGE_PARAMS["scale_mm_per_px"]), 1000.0)))
    return clamped


def render_index(form_defaults=None, image_defaults=None):
    """Render the upload form with provided defaults."""
    form_values = {**DEFAULT_FORM, **(form_defaults or {})}
    image_values = {**DEFAULT_IMAGE_PARAMS, **(image_defaults or {})}
    return render_template(
        "index.html",
        form_defaults=form_values,
        image_defaults=image_values,
        allowed_extensions=sorted(ALLOWED_EXTENSIONS),
    )


@app.route("/")
def index():
    """Display the upload form."""
    return render_index()


@app.route("/upload", methods=["POST"])
def upload_file():
    """Handle file uploads and run analysis/image conversion."""
    form_values = {
        "material": (request.form.get("material") or DEFAULT_FORM["material"]).strip(),
        "thickness": parse_float(request.form.get("thickness"), DEFAULT_FORM["thickness"]),
        "kerf": parse_float(request.form.get("kerf"), DEFAULT_FORM["kerf"]),
        "rate_per_m": parse_float(request.form.get("rate_per_m"), DEFAULT_FORM["rate_per_m"]),
    }

    image_values = normalise_image_params(
        {
            "edge_threshold": parse_float(
                request.form.get("edge_threshold"), DEFAULT_IMAGE_PARAMS["edge_threshold"]
            ),
            "min_contour_area": parse_int(
                request.form.get("min_contour_area"), DEFAULT_IMAGE_PARAMS["min_contour_area"]
            ),
            "simplify_tolerance": parse_float(
                request.form.get("simplify_tolerance"), DEFAULT_IMAGE_PARAMS["simplify_tolerance"]
            ),
            "blur_kernel_size": parse_int(
                request.form.get("blur_kernel_size"), DEFAULT_IMAGE_PARAMS["blur_kernel_size"]
            ),
            "scale_mm_per_px": parse_float(
                request.form.get("scale_mm_per_px"), DEFAULT_IMAGE_PARAMS["scale_mm_per_px"]
            ),
        }
    )

    uploaded_file = request.files.get("file")
    if uploaded_file is None or uploaded_file.filename == "":
        flash("Please choose a DXF or image file to upload.")
        return render_index(form_values, image_values)

    if not allowed_file(uploaded_file.filename):
        allowed_list = ", ".join(sorted(ALLOWED_EXTENSIONS))
        flash(f"Unsupported file. Allowed types: {allowed_list}")
        return render_index(form_values, image_values)

    filename = secure_filename(uploaded_file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    uploaded_file.save(file_path)

    base_name, file_ext = os.path.splitext(filename)
    file_ext = file_ext.lower().lstrip(".")

    analysis_id = f"{base_name}_{uuid.uuid4().hex[:8]}"
    output_dir = os.path.join(OUTPUT_FOLDER, analysis_id)
    os.makedirs(output_dir, exist_ok=True)

    dxf_path = file_path
    if file_ext in {"jpg", "jpeg", "png", "bmp", "tiff"}:
        dxf_path = os.path.join(output_dir, f"{base_name}_converted.dxf")
        processor = ImageProcessor(
            edge_threshold=image_values["edge_threshold"],
            min_contour_area=image_values["min_contour_area"],
            simplify_tolerance=image_values["simplify_tolerance"],
            blur_kernel_size=image_values["blur_kernel_size"],
        )
        contour_count = processor.process_image_to_dxf(
            file_path,
            dxf_path,
            scale_factor=image_values["scale_mm_per_px"],
            offset=(0.0, 0.0),
            debug_dir=output_dir,
        )
        if contour_count == 0:
            flash(
                "No contours detected in the uploaded image. Adjust the image processing "
                "parameters or try a clearer source image."
            )
            return render_index(form_values, image_values)

    analysis_args = Args(
        material=form_values["material"],
        thickness=form_values["thickness"],
        kerf=form_values["kerf"],
        rate_per_m=form_values["rate_per_m"],
        out=output_dir,
    )

    try:
        analyze(dxf_path, analysis_args)
    except Exception as exc:  # pragma: no cover - surface error to UI
        flash(f"Error during analysis: {exc}")
        return render_index(form_values, image_values)

    return redirect(url_for("results", analysis_id=analysis_id))


def _resolve_analysis_file(analysis_id: str, file_type: str) -> Tuple[str, str]:
    if file_type not in ANALYSIS_FILES:
        raise KeyError("Invalid file type")
    file_path = os.path.join(OUTPUT_FOLDER, analysis_id, ANALYSIS_FILES[file_type])
    return ANALYSIS_FILES[file_type], file_path


@app.route("/results/<analysis_id>")
def results(analysis_id: str):
    """Render the analysis results page."""
    output_dir = os.path.join(OUTPUT_FOLDER, analysis_id)
    if not os.path.isdir(output_dir):
        flash("Analysis results not found. Try running the analysis again.")
        return redirect(url_for("index"))

    try:
        _, report_path = _resolve_analysis_file(analysis_id, "report")
    except KeyError:
        flash("Analysis report not found.")
        return redirect(url_for("index"))

    if not os.path.exists(report_path):
        flash("Analysis report not found.")
        return redirect(url_for("index"))

    with open(report_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)

    lengths = []
    try:
        _, lengths_path = _resolve_analysis_file(analysis_id, "lengths")
        if os.path.exists(lengths_path):
            with open(lengths_path, "r", encoding="utf-8") as handle:
                rows = handle.readlines()
            for row in rows[1:]:  # Skip header
                parts = row.strip().split(",")
                if len(parts) >= 4:
                    lengths.append(
                        {
                            "id": parts[0],
                            "class": parts[1],
                            "perimeter": f"{float(parts[2]):.1f}",
                            "area": f"{float(parts[3]):.1f}",
                        }
                    )
    except Exception:
        lengths = []

    available_assets = {}
    for key in ANALYSIS_FILES:
        _, path = _resolve_analysis_file(analysis_id, key)
        available_assets[key] = os.path.exists(path)

    image_metadata = None
    if available_assets.get("image_metadata"):
        _, meta_path = _resolve_analysis_file(analysis_id, "image_metadata")
        with open(meta_path, "r", encoding="utf-8") as handle:
            image_metadata = json.load(handle)
    # Pre-fill reprocess form from metadata if present
    image_defaults = {
        **DEFAULT_IMAGE_PARAMS,
        **(image_metadata.get("parameters", {}) if image_metadata else {}),
    }

    download_order = [
        ("report", "Report (JSON)"),
        ("lengths", "Lengths (CSV)"),
        ("preview", "Preview (PNG)"),
        ("gcode", "G-code (NC)"),
        ("image_metadata", "Image Metadata (JSON)"),
        ("image_gray", "Grayscale Input (PNG)"),
        ("image_threshold", "Threshold Mask (PNG)"),
        ("image_edges", "Edge Map (PNG)"),
    ]

    download_links = [
        {
            "label": label,
            "href": url_for("download_file", analysis_id=analysis_id, file_type=key),
        }
        for key, label in download_order
        if available_assets.get(key)
    ]

    debug_images = [
        {
            "label": label,
            "url": url_for("analysis_asset", analysis_id=analysis_id, file_type=key),
        }
        for key, label in [
            ("preview", "Toolpath Preview"),
            ("image_gray", "Grayscale"),
            ("image_threshold", "Threshold"),
            ("image_edges", "Edges"),
        ]
        if available_assets.get(key)
    ]

    return render_template(
        "results.html",
        analysis_id=analysis_id,
        report=report,
        lengths=lengths,
        download_links=download_links,
        debug_images=debug_images,
        image_metadata=image_metadata,
        available_assets=available_assets,
        reprocess_defaults=image_defaults,
    )


@app.route("/download-dxf/<analysis_id>")
def download_converted_dxf(analysis_id: str):
    """Download the converted DXF produced from an image upload."""
    try:
        _, meta_path = _resolve_analysis_file(analysis_id, "image_metadata")
    except KeyError:
        flash("No converted DXF found for this analysis.")
        return redirect(url_for("results", analysis_id=analysis_id))
    if not os.path.exists(meta_path):
        flash("No converted DXF available.")
        return redirect(url_for("results", analysis_id=analysis_id))
    with open(meta_path, "r", encoding="utf-8") as h:
        meta = json.load(h)
    dxf_path = meta.get("output_dxf")
    if not dxf_path or not os.path.exists(dxf_path):
        flash("Converted DXF file is missing.")
        return redirect(url_for("results", analysis_id=analysis_id))
    name = os.path.basename(dxf_path)
    return send_file(dxf_path, as_attachment=True, download_name=name)


@app.route("/reprocess/<analysis_id>", methods=["POST"])
def reprocess(analysis_id: str):
    """Re-run image->DXF conversion and analysis with new parameters."""
    output_dir = os.path.join(OUTPUT_FOLDER, analysis_id)
    try:
        _, meta_path = _resolve_analysis_file(analysis_id, "image_metadata")
    except KeyError:
        flash("Reprocess requires an image-backed analysis.")
        return redirect(url_for("results", analysis_id=analysis_id))

    if not os.path.exists(meta_path):
        flash("Original image metadata not found.")
        return redirect(url_for("results", analysis_id=analysis_id))

    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)

    source_image = meta.get("source_image")
    dxf_path = meta.get("output_dxf")
    if not source_image or not dxf_path:
        flash("Reprocess is not possible without the original image.")
        return redirect(url_for("results", analysis_id=analysis_id))

    # Parse new params
    image_values = normalise_image_params(
        {
            "edge_threshold": parse_float(
                request.form.get("edge_threshold"), DEFAULT_IMAGE_PARAMS["edge_threshold"]
            ),
            "min_contour_area": parse_int(
                request.form.get("min_contour_area"), DEFAULT_IMAGE_PARAMS["min_contour_area"]
            ),
            "simplify_tolerance": parse_float(
                request.form.get("simplify_tolerance"), DEFAULT_IMAGE_PARAMS["simplify_tolerance"]
            ),
            "blur_kernel_size": parse_int(
                request.form.get("blur_kernel_size"), DEFAULT_IMAGE_PARAMS["blur_kernel_size"]
            ),
            "scale_mm_per_px": parse_float(
                request.form.get("scale_mm_per_px"), DEFAULT_IMAGE_PARAMS["scale_mm_per_px"]
            ),
        }
    )

    # Re-run conversion
    processor = ImageProcessor(
        edge_threshold=image_values["edge_threshold"],
        min_contour_area=image_values["min_contour_area"],
        simplify_tolerance=image_values["simplify_tolerance"],
        blur_kernel_size=image_values["blur_kernel_size"],
    )
    try:
        contour_count = processor.process_image_to_dxf(
            source_image,
            dxf_path,
            scale_factor=image_values["scale_mm_per_px"],
            offset=(0.0, 0.0),
            debug_dir=output_dir,
        )
        if contour_count == 0:
            flash("No contours detected with the chosen parameters.")
            return redirect(url_for("results", analysis_id=analysis_id))
    except Exception as exc:
        flash(f"Image processing failed: {exc}")
        return redirect(url_for("results", analysis_id=analysis_id))

    # Re-run analysis
    # Use prior material/kerf from report.json if available, else defaults
    report_path = os.path.join(output_dir, "report.json")
    material = DEFAULT_FORM["material"]
    thickness = DEFAULT_FORM["thickness"]
    kerf = DEFAULT_FORM["kerf"]
    rate = DEFAULT_FORM["rate_per_m"]
    if os.path.exists(report_path):
        try:
            with open(report_path, "r", encoding="utf-8") as h:
                rep = json.load(h)
            material = rep.get("material", {}).get("name", material)
            thickness = rep.get("material", {}).get("thickness_mm", thickness)
            kerf = rep.get("kerf_mm", kerf)
        except Exception:
            pass
    analysis_args = Args(
        material=material,
        thickness=thickness,
        kerf=kerf,
        rate_per_m=rate,
        out=output_dir,
    )
    try:
        analyze(dxf_path, analysis_args)
    except Exception as exc:
        flash(f"Analysis failed after reprocess: {exc}")
    return redirect(url_for("results", analysis_id=analysis_id))


@app.route("/download/<analysis_id>/<file_type>")
def download_file(analysis_id: str, file_type: str):
    """Download analysis artefacts as attachments."""
    try:
        name, file_path = _resolve_analysis_file(analysis_id, file_type)
    except KeyError:
        flash("Invalid file requested.")
        return redirect(url_for("index"))

    if not os.path.exists(file_path):
        flash("Requested file is not available.")
        return redirect(url_for("index"))

    return send_file(file_path, as_attachment=True, download_name=name)


@app.route("/assets/<analysis_id>/<file_type>")
def analysis_asset(analysis_id: str, file_type: str):
    """Serve analysis artefacts inline for previews."""
    try:
        _, file_path = _resolve_analysis_file(analysis_id, file_type)
    except KeyError:
        flash("Invalid asset requested.")
        return redirect(url_for("index"))

    if not os.path.exists(file_path):
        flash("Asset not available.")
        return redirect(url_for("index"))

    return send_file(file_path, as_attachment=False)


@app.route("/sample")
def generate_sample():
    """Generate and serve a sample DXF file for testing."""
    try:
        from scripts.make_sample_dxf import make_sample

        sample_path = os.path.join(UPLOAD_FOLDER, "sample_medallion.dxf")
        make_sample(sample_path)
        return send_file(sample_path, as_attachment=True, download_name="sample_medallion.dxf")
    except Exception as exc:  # pragma: no cover
        flash(f"Error generating sample: {exc}")
        return redirect(url_for("index"))


if __name__ == "__main__":
    print("Starting Waterjet DXF Analyzer Web Interface...")
    print("Open your browser and go to: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    app.run(debug=True, host="0.0.0.0", port=5000)
