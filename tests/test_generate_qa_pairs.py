import json
import pytest
from shared.data_pipeline.generate_qa_pairs import parse, save_jsonl
from pathlib import Path

FIXTURE_GRAPHML = "tests/fixtures/sample.graphml"
FIXTURE_CGHD = "tests/fixtures/sample_cghd.json"
FIXTURE_AS1100 = "tests/fixtures/sample_as1100.json"


def test_parse_graphml_returns_list_of_dicts():
    pairs = parse("graphml", FIXTURE_GRAPHML)
    assert isinstance(pairs, list)
    assert len(pairs) > 0
    for p in pairs:
        assert "instruction" in p
        assert "input" in p
        assert "output" in p


def test_parse_graphml_finds_pump_isolation():
    pairs = parse("graphml", FIXTURE_GRAPHML)
    instructions = [p["instruction"] for p in pairs]
    assert any("P-101" in i for i in instructions)


def test_parse_graphml_finds_vessel_failures():
    pairs = parse("graphml", FIXTURE_GRAPHML)
    instructions = [p["instruction"] for p in pairs]
    assert any("TK-101" in i for i in instructions)


def test_parse_unknown_format_raises():
    with pytest.raises(ValueError, match="Unknown format"):
        parse("xml", "some/path")


def test_save_jsonl_writes_file(tmp_path):
    pairs = [{"instruction": "Q", "input": "", "output": "A"}]
    dest = str(tmp_path / "out.jsonl")
    save_jsonl(pairs, dest)
    lines = Path(dest).read_text().strip().split("\n")
    assert len(lines) == 1
    assert json.loads(lines[0]) == pairs[0]


def test_parse_cghd_returns_list_of_dicts():
    pairs = parse("cghd", FIXTURE_CGHD)
    assert isinstance(pairs, list)
    assert len(pairs) > 0
    for p in pairs:
        assert "instruction" in p
        assert "output" in p


def test_parse_as1100_returns_list_of_dicts():
    pairs = parse("as1100", FIXTURE_AS1100)
    assert isinstance(pairs, list)
    assert len(pairs) > 0
    for p in pairs:
        assert "instruction" in p
        assert "output" in p


def test_parse_as1100_finds_tightest_tolerance():
    pairs = parse("as1100", FIXTURE_AS1100)
    outputs = [p["output"] for p in pairs]
    assert any("shaft_diameter" in o for o in outputs)


def test_parse_circuit1k_yolo_returns_multiple_pairs():
    fixture = Path("tests/fixtures/sample_circuit.txt")
    pairs = parse("circuit1k_yolo", str(fixture))
    assert len(pairs) >= 4
    instructions = [p["instruction"] for p in pairs]
    assert any("Describe" in i for i in instructions)
    assert any("SPICE" in i for i in instructions)


def test_parse_circuit1k_yolo_empty_file(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    pairs = parse("circuit1k_yolo", str(empty))
    assert pairs == []


def test_parse_dimensioning_yolo_returns_pairs(tmp_path):
    label_file = tmp_path / "drawing1.txt"
    label_file.write_text(
        "0 0.5 0.3 0.1 0.05\n0 0.7 0.4 0.1 0.05\n1 0.2 0.6 0.15 0.04\n",
        encoding="utf-8",
    )
    pairs = parse("dimensioning_yolo", str(label_file))
    assert len(pairs) >= 2
    instructions = [p["instruction"] for p in pairs]
    assert any("dimension" in i.lower() for i in instructions)


def test_parse_dimensioning_yolo_empty_file(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    pairs = parse("dimensioning_yolo", str(empty))
    assert pairs == []
