import io
import os
import tempfile
from typing import Any

import ezdxf
from ezdxf import recover


def _is_file_like(obj: Any) -> bool:
    return hasattr(obj, "read") and callable(getattr(obj, "read", None))


def load_dxf(path_or_file: Any):
    """
    Load a DXF document from a filesystem path or a file-like object.

    For file-like uploads (e.g., Streamlit UploadedFile), write to a temporary
    file and use readfile/recover.readfile for robust handling.
    """
    if _is_file_like(path_or_file):
        # Ensure we read from the beginning
        try:
            path_or_file.seek(0)
        except Exception:
            pass
        data = path_or_file.read()
        if isinstance(data, str):
            data = data.encode("utf-8", errors="ignore")

        if not isinstance(data, (bytes, bytearray)):
            # As a last resort, try using recover.read on the stream directly
            try:
                doc, _ = recover.read(path_or_file)
                return doc
            except Exception:
                raise TypeError("Unsupported DXF input type; expected bytes-like or path.")

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            try:
                return ezdxf.readfile(tmp_path)
            except ezdxf.DXFError:
                doc, _ = recover.readfile(tmp_path)
                return doc
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    # Treat as path-like
    path_str = os.fspath(path_or_file)
    try:
        return ezdxf.readfile(path_str)
    except ezdxf.DXFError:
        doc, _ = recover.readfile(path_str)
        return doc


def save_dxf(doc, out_path: str) -> str:
    doc.saveas(out_path)
    return out_path


