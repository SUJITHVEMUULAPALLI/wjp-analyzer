"""
Lightweight quote exporters (PDF and XLSX) without external dependencies.

Generates a simple one-page PDF and a minimal XLSX workbook summarizing
key metrics from report.json. Designed as a pragmatic fallback so the
web routes work out-of-the-box.
"""
from __future__ import annotations

import json
import os
import time
import zipfile
from typing import Optional, Dict, Any, List
from datetime import datetime


def _load_report(report_path: str) -> Dict[str, Any]:
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------- PDF Export ---------------------------

def _pdf_escape_text(s: str) -> str:
    return s.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def _build_pdf_content(lines: List[str]) -> bytes:
    # Start at top margin and step down per line
    y_start = 780
    leading = 16
    chunks = [
        b"BT\n",
        b"/F1 14 Tf\n",
        f"1 0 0 1 50 {y_start} Tm\n".encode("ascii"),
    ]
    first = True
    for line in lines:
        text = _pdf_escape_text(line)
        if first:
            first = False
        else:
            chunks.append(f"0 -{leading} Td\n".encode("ascii"))
        chunks.append(f"({text}) Tj\n".encode("latin-1", errors="replace"))
    chunks.append(b"ET\n")
    return b"".join(chunks)


def _write_minimal_pdf(out_path: str, title: str, lines: List[str]) -> None:
    # Minimal PDF with one page and Helvetica font
    content = _build_pdf_content([title] + ["" ] + lines)
    objects: List[bytes] = []
    xref: List[int] = []

    def add_object(obj: bytes) -> None:
        xref.append(len(pdf))
        objects.append(obj)

    pdf = b"%PDF-1.4\n%\xff\xff\xff\xff\n"

    # 1: Catalog
    add_object(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    # 2: Pages
    add_object(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")
    # 3: Page
    page = (
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R "
        b"/MediaBox [0 0 595 842] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
    )
    add_object(page)
    # 4: Contents
    stream = content
    contents = (
        f"4 0 obj\n<< /Length {len(stream)} >>\nstream\n".encode("ascii")
        + stream
        + b"endstream\nendobj\n"
    )
    add_object(contents)
    # 5: Font
    add_object(b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")

    # Assemble PDF with xref
    # Rebuild byte stream to compute proper offsets
    pdf_parts = [b"%PDF-1.4\n%\xff\xff\xff\xff\n"]
    xref = []
    offset = sum(len(p) for p in pdf_parts)
    objs_serialized = []
    for i, obj in enumerate(objects, start=1):
        objs_serialized.append(obj)
    # compute offsets now
    cur = sum(len(p) for p in pdf_parts)
    offsets = []
    for obj in objs_serialized:
        offsets.append(cur)
        cur += len(obj)
    xref_pos = cur
    pdf_body = b"".join(objs_serialized)
    pdf_parts.append(pdf_body)
    # xref
    xref_lines = [b"xref\n", f"0 {len(offsets)+1}\n".encode("ascii"), b"0000000000 65535 f \n"]
    for off in offsets:
        xref_lines.append(f"{off:010d} 00000 n \n".encode("ascii"))
    trailer = (
        b"trailer\n<< /Size "
        + str(len(offsets) + 1).encode("ascii")
        + b" /Root 1 0 R >>\nstartxref\n"
        + str(xref_pos).encode("ascii")
        + b"\n%%EOF\n"
    )
    pdf_parts.append(b"".join(xref_lines))
    pdf_parts.append(trailer)

    with open(out_path, "wb") as f:
        f.write(b"".join(pdf_parts))


def make_pdf(report_path: str, csv_path: Optional[str], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rep = _load_report(report_path)
    file_name = rep.get("file", os.path.basename(report_path))
    mat = rep.get("material", {})
    metrics = rep.get("metrics", {})

    title = "Waterjet DXF Quotation"
    lines = [
        f"File: {file_name}",
        f"Material: {mat.get('name', 'N/A')} ({mat.get('thickness_mm', 'N/A')} mm)",
        f"Kerf: {rep.get('kerf_mm', 'N/A')}",
        "",
        "--- Metrics ---",
        f"Outer length: {metrics.get('length_outer_mm', 0):,.1f} mm",
        f"Internal length: {metrics.get('length_internal_mm', 0):,.1f} mm",
        f"Pierces: {metrics.get('pierces', 0)}",
        f"Estimated time: {metrics.get('est_time_min', 0)} min",
        f"Estimated cost: INR {metrics.get('cost_inr', 0):,.0f}",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M')}"
    ]
    _write_minimal_pdf(out_path, title, lines)


# --------------------------- XLSX Export ---------------------------

def _xlsx_xml_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\"", "&quot;")
    )


def _build_xlsx_zip(rows: List[List[str]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        # [Content_Types].xml
        z.writestr(
            "[Content_Types].xml",
            """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>""",
        )
        # _rels/.rels
        z.writestr(
            "_rels/.rels",
            """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="/xl/workbook.xml"/>
</Relationships>""",
        )
        # xl/workbook.xml
        z.writestr(
            "xl/workbook.xml",
            """<?xml version="1.0" encoding="UTF-8"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="Quote" sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>""",
        )
        # xl/_rels/workbook.xml.rels
        z.writestr(
            "xl/_rels/workbook.xml.rels",
            """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>""",
        )
        # xl/styles.xml (minimal)
        z.writestr(
            "xl/styles.xml",
            """<?xml version="1.0" encoding="UTF-8"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
</styleSheet>""",
        )
        # xl/worksheets/sheet1.xml
        cells_xml = []
        for r_idx, row in enumerate(rows, start=1):
            cells = []
            for c_idx, val in enumerate(row, start=1):
                # Convert column index to letters (A, B, ...)
                col = ""
                n = c_idx
                while n:
                    n, rem = divmod(n - 1, 26)
                    col = chr(65 + rem) + col
                esc = _xlsx_xml_escape(str(val))
                cells.append(f"<c r=\"{col}{r_idx}\" t=\"inlineStr\"><is><t>{esc}</t></is></c>")
            cells_xml.append(f"<row r=\"{r_idx}\">{''.join(cells)}</row>")
        sheet_xml = (
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
            "<worksheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\">"
            + "<sheetData>"
            + "".join(cells_xml)
            + "</sheetData></worksheet>"
        )
        z.writestr("xl/worksheets/sheet1.xml", sheet_xml)


def make_xlsx(report_path: str, csv_path: Optional[str], out_path: str) -> None:
    rep = _load_report(report_path)
    mat = rep.get("material", {})
    m = rep.get("metrics", {})

    rows = [
        ["Waterjet DXF Quote"],
        ["File", rep.get("file", os.path.basename(report_path))],
        ["Material", f"{mat.get('name', 'N/A')} ({mat.get('thickness_mm', 'N/A')} mm)"],
        ["Kerf (mm)", rep.get("kerf_mm", "N/A")],
        ["Outer length (mm)", f"{m.get('length_outer_mm', 0):,.1f}"],
        ["Internal length (mm)", f"{m.get('length_internal_mm', 0):,.1f}"],
        ["Pierces", m.get("pierces", 0)],
        ["Estimated time (min)", m.get("est_time_min", 0)],
        ["Estimated cost (INR)", f"{m.get('cost_inr', 0):,.0f}"],
    ]
    if csv_path and os.path.exists(csv_path):
        rows.append([""])
        rows.append(["Contours (id, class, perimeter, area)"])
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i > 2000:
                        break
                    parts = [p.strip() for p in line.strip().split(",")]
                    rows.append(parts)
        except Exception:
            pass
    _build_xlsx_zip(rows, out_path)


# --------------------------- Cutting Report Export ---------------------------

def export_cutting_report(analysis: Dict[str, Any], filename: str = "cut_report.pdf", 
                         garnet_rate: float = 0.35) -> str:
    """
    Export shopfloor cutting report as PDF.
    
    Args:
        analysis: Analysis results dictionary
        filename: Output filename
        garnet_rate: Garnet consumption rate in kg per minute
    
    Returns:
        Path to generated PDF file
    """
    # Extract metrics
    metrics = analysis.get('metrics', {})
    entities = analysis.get('entities', {})
    violations = analysis.get('violations', [])
    
    # Calculate cutting parameters
    total_length_mm = metrics.get('length_internal_mm', 0)
    total_length_m = total_length_mm / 1000.0
    pierce_count = metrics.get('pierces', 0)
    estimated_time_min = metrics.get('est_time_min', 0)
    cost_inr = metrics.get('cost_inr', 0)
    
    # Calculate garnet usage
    garnet_usage_kg = estimated_time_min * garnet_rate
    
    # Build PDF content
    lines = []
    
    # Header
    lines.append("WATERJET CUTTING REPORT")
    lines.append("=" * 50)
    lines.append("")
    
    # Job information
    lines.append(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"File: {analysis.get('dxf_path', 'Unknown')}")
    lines.append("")
    
    # Cutting parameters
    lines.append("CUTTING PARAMETERS")
    lines.append("-" * 30)
    lines.append(f"Total Cutting Length: {total_length_m:.2f} m ({total_length_mm:.0f} mm)")
    lines.append(f"Number of Pierce Points: {pierce_count}")
    lines.append(f"Estimated Cutting Time: {estimated_time_min:.1f} minutes")
    lines.append(f"Estimated Cost: ₹{cost_inr:,.0f}")
    lines.append("")
    
    # Material consumption
    lines.append("MATERIAL CONSUMPTION")
    lines.append("-" * 30)
    lines.append(f"Garnet Usage: {garnet_usage_kg:.2f} kg")
    lines.append(f"Garnet Rate: {garnet_rate} kg/min")
    lines.append("")
    
    # Geometry information
    lines.append("GEOMETRY INFORMATION")
    lines.append("-" * 30)
    lines.append(f"Total Polygons: {entities.get('polygons', 0)}")
    lines.append(f"Outer Contours: {entities.get('outer', 0)}")
    lines.append(f"Inner Contours: {entities.get('inner', 0)}")
    lines.append(f"Total Lines: {entities.get('lines', 0)}")
    lines.append("")
    
    # Quality issues
    if violations:
        lines.append("QUALITY ISSUES")
        lines.append("-" * 30)
        for i, violation in enumerate(violations[:10], 1):  # Show first 10
            lines.append(f"{i}. {violation}")
        if len(violations) > 10:
            lines.append(f"... and {len(violations) - 10} more issues")
        lines.append("")
    
    # Cutting instructions
    lines.append("CUTTING INSTRUCTIONS")
    lines.append("-" * 30)
    lines.append("1. Ensure material is properly secured")
    lines.append("2. Check water pressure and garnet supply")
    lines.append("3. Verify cutting parameters before starting")
    lines.append("4. Monitor cutting progress for quality issues")
    lines.append("5. Clean up after cutting completion")
    lines.append("")
    
    # Safety notes
    lines.append("SAFETY NOTES")
    lines.append("-" * 30)
    lines.append("• Wear appropriate PPE (safety glasses, hearing protection)")
    lines.append("• Ensure proper ventilation in cutting area")
    lines.append("• Keep hands away from cutting head during operation")
    lines.append("• Follow all machine safety procedures")
    lines.append("")
    
    # Footer
    lines.append("Generated by WJP ANALYSER")
    lines.append("Waterjet DXF Analysis System")
    
    # Generate PDF
    pdf_content = _build_pdf_content(lines)
    
    # Write to file
    with open(filename, 'wb') as f:
        f.write(pdf_content)
    
    print(f"[OK] Cutting report exported to: {filename}")
    return filename


def export_cutting_report_from_file(report_path: str, filename: str = "cut_report.pdf") -> str:
    """
    Export cutting report from analysis report file.
    
    Args:
        report_path: Path to analysis report JSON file
        filename: Output filename
    
    Returns:
        Path to generated PDF file
    """
    analysis = _load_report(report_path)
    return export_cutting_report(analysis, filename)

