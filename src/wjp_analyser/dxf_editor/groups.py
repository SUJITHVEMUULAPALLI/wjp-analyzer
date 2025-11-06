def create_group(doc, name, handles):
    return doc.groups.add(name, handles)


def list_groups(doc):
    return [g.name for g in doc.groups]


def get_group(doc, name):
    try:
        return doc.groups.get(name)
    except KeyError:
        return None





