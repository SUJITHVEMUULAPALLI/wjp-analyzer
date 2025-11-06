def get_layers(doc):
    """Return mapping of layer name -> ACI color with broad ezdxf compatibility."""
    # Preferred: iterate entries
    try:
        result = {}
        for entry in doc.layers:  # LayerTable is iterable in many versions
            try:
                name = entry.dxf.name
            except Exception:
                # Some versions use .name directly
                name = getattr(entry, "name", None)
            if not name:
                continue
            color = getattr(getattr(entry, "dxf", entry), "color", 7)
            try:
                color = int(color)
            except Exception:
                color = 7
            result[name] = color
        if result:
            return result
    except Exception:
        pass

    # Fallback: names()/get()
    names = []
    for attr in ("names", "keys"):
        try:
            method = getattr(doc.layers, attr)
            names = list(method())
            break
        except Exception:
            continue
    layers = {}
    for name in names:
        try:
            entry = doc.layers.get(name)
            color = getattr(getattr(entry, "dxf", entry), "color", 7)
            layers[name] = int(color) if isinstance(color, (int, float, str)) else 7
        except Exception:
            layers[name] = 7
    return layers


def ensure_layer(doc, name, color=7):
    if name in doc.layers:
        return doc.layers.get(name)
    return doc.layers.add(name=name, color=color)


def rename_layer(doc, old, new):
    if old == new or old not in doc.layers:
        return
    try:
        # Some ezdxf versions support entry.rename()
        entry = doc.layers.get(old)
        entry.rename(new)  # type: ignore[attr-defined]
        return
    except Exception:
        pass
    try:
        layer = doc.layers.get(old)
        color = getattr(layer.dxf, "color", 7)
        doc.layers.remove(old)
        doc.layers.add(name=new, color=color)
    except Exception:
        return


def recolor_layer(doc, name, color):
    if name in doc.layers:
        doc.layers.get(name).dxf.color = int(color)


def move_entities_to_layer(entities, layer_name):
    for e in entities:
        e.dxf.layer = layer_name



