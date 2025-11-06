import ezdxf
from ezdxf import recover

def load_dxf(path_or_file):
    try:
        doc = ezdxf.readfile(path_or_file)
    except ezdxf.DXFError:
        doc, _ = recover.readfile(path_or_file)
    return doc

def save_dxf(doc, out_path):
    doc.saveas(out_path)
    return out_path
