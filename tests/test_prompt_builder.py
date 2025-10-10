from wjpanalyser.app.services.prompt_builder import build_prompt


class Dummy:
    category = "Inlay Tile"
    material = "White Marble"
    dimensions = "600x600 mm"
    style = "geometric floral"
    complexity = 3
    notes = "keep bold"
    top_view = True
    symmetry_required = True
    waterjet_safe = True


def test_prompt_contains_core_phrases():
    p = build_prompt(Dummy())
    assert "waterjet-safe" in p
    assert "dimensions 600x600 mm" in p
    assert "geometric floral" in p

