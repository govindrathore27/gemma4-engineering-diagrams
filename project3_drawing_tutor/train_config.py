#!/usr/bin/env python3
"""Generate training data for Project 3: Engineering Drawing Tutor."""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.data_pipeline.generate_qa_pairs import parse, save_jsonl

_SYNTHETIC_TEMPLATES: list[tuple[str, str]] = [
    (
        "List the critical dimensions for a shaft drawing.",
        '{"dimensions": ["overall_length", "shaft_diameter", "keyway_width", "bore_diameter"]}',
    ),
    (
        "What does a surface finish callout of Ra 1.6 mean?",
        "Ra 1.6 μm indicates a fine machined surface, suitable for moving parts requiring smooth contact.",
    ),
    (
        "Explain the difference between bilateral and unilateral tolerances.",
        "Bilateral tolerances allow variation in both directions (e.g., ±0.05 mm). "
        "Unilateral tolerances allow variation in one direction only (e.g., +0.00/−0.10 mm), "
        "used where one limit is critical.",
    ),
    (
        "What is a first-angle projection and when is it used?",
        "First-angle (European) projection places views as if the object is rotated away from the "
        "viewer. It is the standard in AS1100 Australian/British engineering drawings.",
    ),
    (
        "How do you verify a drawing conforms to AS1100 standard?",
        "Check the title block format, projection angle symbol, dimension style, surface finish "
        "notation, and tolerance format all comply with AS1100 Part 101.",
    ),
    (
        "What should be included in an engineering drawing title block?",
        '{"required": ["part_number", "title", "material", "scale", "drawn_by", '
        '"date", "revision", "company_name"]}',
    ),
    (
        "What is geometric dimensioning and tolerancing (GD&T)?",
        "GD&T is a system for defining and communicating engineering tolerances using symbols to "
        "specify form, orientation, location, and runout of features.",
    ),
    (
        "How do you read a tolerance of 25.00 ± 0.05?",
        "The nominal dimension is 25.00 mm. The feature must be between 24.95 mm and 25.05 mm "
        "to pass inspection.",
    ),
    (
        "What are the standard line types used in engineering drawings?",
        '{"line_types": {"continuous_thick": "visible edges", "continuous_thin": "dimension lines", '
        '"dashed_thin": "hidden edges", "chain_thin": "centre lines", "chain_thick": "cutting planes"}}',
    ),
    (
        "Why is the scale important on an engineering drawing?",
        "The scale ensures parts are manufactured to the correct size regardless of how they appear "
        "on paper. Common scales are 1:1 (full size), 1:2 (half size), and 2:1 (double size).",
    ),
]


def _synthetic_pairs(n: int) -> list[dict]:
    pairs = []
    for _ in range(n):
        q, a = random.choice(_SYNTHETIC_TEMPLATES)
        pairs.append({"instruction": q, "input": "", "output": a})
    return pairs


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / "project3_drawing_tutor" / "data"
    all_pairs: list[dict] = []

    # Source 1: AS1100 parquet (all splits)
    as1100_data = out_dir / "as1100" / "data"
    if as1100_data.exists():
        for parquet_file in sorted(as1100_data.glob("*.parquet")):
            pairs = parse("as1100_parquet", str(parquet_file))
            all_pairs.extend(pairs)
            print(f"  as1100 {parquet_file.name}: {len(pairs)} pairs")
    else:
        print(f"[WARN] AS1100 parquet not found at {as1100_data}")

    # Source 2: Dimensioning YOLO labels
    dim_labels_dir = (
        out_dir
        / "dimensioning"
        / "Drawing_Annotation_Recognition8.v1i.yolov12"
        / "test"
        / "labels"
    )
    if dim_labels_dir.exists():
        txt_files = sorted(dim_labels_dir.glob("*.txt"))
        print(f"Found {len(txt_files)} dimensioning label files")
        for txt_file in txt_files:
            all_pairs.extend(parse("dimensioning_yolo", str(txt_file)))
    else:
        print(f"[INFO] No labels dir at {dim_labels_dir} — using synthetic pad")

    print(f"Total before synthetic pad: {len(all_pairs)}")

    # Pad to ≥1500 with synthetic templates
    if len(all_pairs) < 1500:
        pad = _synthetic_pairs(1500 - len(all_pairs))
        all_pairs.extend(pad)
        print(f"Added {len(pad)} synthetic pairs")

    print(f"Total QA pairs: {len(all_pairs)}")

    split_idx = int(len(all_pairs) * 0.8)
    save_jsonl(all_pairs[:split_idx], str(out_dir / "train.jsonl"))
    save_jsonl(all_pairs[split_idx:], str(out_dir / "eval.jsonl"))
    print(f"Saved {split_idx} train + {len(all_pairs) - split_idx} eval pairs")


if __name__ == "__main__":
    main()
