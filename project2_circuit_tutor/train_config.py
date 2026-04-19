#!/usr/bin/env python3
"""Generate training data for Project 2: Circuit Diagram Explanation Tutor."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.data_pipeline.generate_qa_pairs import parse, save_jsonl


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / "project2_circuit_tutor" / "data"
    all_pairs: list[dict] = []

    # Source 1: circuit1k YOLO annotations
    circuit1k_dir = out_dir / "digitize_hcd" / "circuits(1k)"
    if circuit1k_dir.exists():
        txt_files = sorted(circuit1k_dir.glob("*.txt"))
        print(f"Found {len(txt_files)} circuit1k annotation files")
        for txt_file in txt_files:
            all_pairs.extend(parse("circuit1k_yolo", str(txt_file)))
    else:
        print(f"[WARN] circuit1k not found at {circuit1k_dir}")

    # Source 2: CGHD JSON annotations (after zip is extracted)
    cghd_dir = out_dir / "cghd"
    json_files = list(cghd_dir.glob("**/*.json")) if cghd_dir.exists() else []
    if json_files:
        print(f"Found {len(json_files)} CGHD annotation files")
        for json_file in json_files[:500]:
            all_pairs.extend(parse("cghd", str(json_file)))
    else:
        print("[INFO] CGHD annotations not yet extracted — using circuit1k only")

    print(f"Generated {len(all_pairs)} total QA pairs")

    # 80/20 split
    split_idx = int(len(all_pairs) * 0.8)
    save_jsonl(all_pairs[:split_idx], str(out_dir / "train.jsonl"))
    save_jsonl(all_pairs[split_idx:], str(out_dir / "eval.jsonl"))
    print(f"Saved {split_idx} train + {len(all_pairs) - split_idx} eval pairs")


if __name__ == "__main__":
    main()
