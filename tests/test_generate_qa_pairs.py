import json
import pytest
from shared.data_pipeline.generate_qa_pairs import parse, save_jsonl
from pathlib import Path

FIXTURE_GRAPHML = "tests/fixtures/sample.graphml"


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
