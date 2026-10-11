from atlas import grounding
from atlas.speech import spoken, spoken_es


def test_spoken_english():
    t = spoken("Q4 revenue is $644.7M, coverage 3.25x, FY27 P50 $2.48B, loss -$4.3M.")
    assert "Q four" in t and "644.7 million dollars" in t and "3.25 times" in t
    assert "F Y 27" in t and "P 50" in t and "2.48 billion dollars" in t and "minus 4.3 million dollars" in t


def test_spoken_spanish():
    t = spoken_es("Los ingresos de Q4 son $644.7M, margen 13.3%, HHI 2,123, cobertura 3.25x.")
    assert "del cuarto trimestre" in t and "644,7 millones de dólares" in t
    assert "13,3 por ciento" in t and "2123" in t and "3,25 veces" in t


def _r(facts):
    return {"facts": facts, "template": "", "headline": {}, "inputs": [], "percentiles": {}}


def test_grounding_formats():
    r = _r(["Revenue is $644.7M and margin 13.3%.", "Coverage is 3.25x.", "10,904 students."])
    assert grounding.check("We expect $644.7M.", r)["ok"]
    assert grounding.check("That is 644.7 million dollars.", r)["ok"]
    assert grounding.check("Margin is 13.3 percent.", r)["ok"]
    assert grounding.check("Coverage 3.25x with 10,904 students.", r)["ok"]
    assert not grounding.check("Margin is 15.0%.", r)["ok"]
    assert not grounding.check("About $700M.", r)["ok"]


def test_grounding_ignores_labels():
    r = _r(["Revenue is $1.0M."])
    assert grounding.check("In Q4 2026 and FY27 the picture holds.", r)["ok"]
