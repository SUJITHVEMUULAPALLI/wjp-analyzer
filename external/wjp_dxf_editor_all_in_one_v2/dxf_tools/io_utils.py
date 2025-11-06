import ezdxf
from ezdxf import recover
from pathlib import Path

def load_dxf(path_or_file):
    try:
        doc = ezdxf.readfile(path_or_file)
    except ezdxf.DXFError:
        doc, _ = recover.readfile(path_or_file)
    return doc

def save_dxf(doc, out_path):
    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(out_path)
    return out_path
