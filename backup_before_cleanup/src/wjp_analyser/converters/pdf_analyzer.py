from __future__ import annotations

"""
PDF Analyzer: extract vector paths and embedded images from PDF pages.
Produces intermediate assets and DXF using existing tracing/vectorization.
"""

import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path

import fitz  # PyMuPDF
import ezdxf


@dataclass
class ExtractOptions:
    page: int = 0
    max_pages: int = 1
    trace_images: bool = True
    out_dir: str = "output/pdf_extract"


def _ensure_out(dir_path: str) -> str:
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def _export_vectors_to_dxf(doc: fitz.Document, page_index: int, out_path: str) -> str:
    page = doc.load_page(page_index)
    drawings = page.get_drawings()
    d = ezdxf.new(dxfversion="R2010")
    msp = d.modelspace()
    for item in drawings:
        for path in item.get("items", []):
            # path is a sequence of draw commands with points
            pts: List[Tuple[float, float]] = []
            for seg in path:
                if seg[0] == "l":
                    # line to
                    pts.append((seg[1], seg[2]))
                elif seg[0] == "m":
                    # move to
                    if pts:
                        try:
                            msp.add_lwpolyline(pts, close=False)
                        except Exception:
                            pass
                        pts = []
                    pts.append((seg[1], seg[2]))
                elif seg[0] == "c":
                    # cubic bezier approximation: sample end point only (simple fallback)
                    pts.append((seg[5], seg[6]))
            if pts:
                try:
                    msp.add_lwpolyline(pts, close=False)
                except Exception:
                    pass
    d.saveas(out_path)
    return out_path


def _export_images(doc: fitz.Document, page_index: int, out_dir: str) -> List[str]:
    page = doc.load_page(page_index)
    img_refs = page.get_images(full=True)
    saved: List[str] = []
    for idx, img in enumerate(img_refs):
        xref = img[0]
        pix = fitz.Pixmap(doc, xref)
        img_path = os.path.join(out_dir, f"page{page_index}_img{idx}.png")
        if pix.n > 4:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        pix.save(img_path)
        saved.append(img_path)
    return saved


def analyze_pdf(pdf_path: str, options: Optional[ExtractOptions] = None) -> Dict[str, Any]:
    opts = options or ExtractOptions()
    out_root = _ensure_out(opts.out_dir)
    result: Dict[str, Any] = {"pages": []}
    doc = fitz.open(pdf_path)
    try:
        maxp = min(doc.page_count, opts.max_pages)
        for p in range(min(opts.page, doc.page_count - 1), min(opts.page + maxp, doc.page_count)):
            page_dir = _ensure_out(os.path.join(out_root, f"page_{p}"))
            vectors_dxf = os.path.join(page_dir, "vectors.dxf")
            _export_vectors_to_dxf(doc, p, vectors_dxf)
            images = _export_images(doc, p, page_dir)
            result["pages"].append({
                "page": p,
                "vectors_dxf": vectors_dxf,
                "images": images,
            })
        result["success"] = True
    except Exception as e:
        result.update({"success": False, "error": str(e)})
    finally:
        doc.close()
    return result




