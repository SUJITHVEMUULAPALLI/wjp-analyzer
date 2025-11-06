def get_layers(doc):
    return {name: layer.dxf.color for name, layer in doc.layers.items()}

def ensure_layer(doc, name, color=7):
    if name in doc.layers:
        return doc.layers.get(name)
    return doc.layers.add(name=name, color=color)

def rename_layer(doc, old, new):
    if old == new or old not in doc.layers:
        return
    layer = doc.layers.get(old)
    # ezdxf has no direct rename; recreate
    color = layer.dxf.color
    doc.layers.remove(old)
    doc.layers.add(name=new, color=color)

def recolor_layer(doc, name, color):
    if name in doc.layers:
        doc.layers.get(name).dxf.color = int(color)

def move_entities_to_layer(entities, layer_name):
    for e in entities:
        e.dxf.layer = layer_name
