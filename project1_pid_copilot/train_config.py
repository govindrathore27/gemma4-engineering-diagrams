# project1_pid_copilot/train_config.py
from pathlib import Path
from shared.data_pipeline.generate_qa_pairs import parse, save_jsonl
import json, random

DATA_DIR = Path("project1_pid_copilot/data/pid2graph/PID2Graph/Complete/Dataset PID")
OUTPUT_JSONL = "project1_pid_copilot/data/train.jsonl"
EVAL_JSONL = "project1_pid_copilot/data/eval.jsonl"


def generate() -> int:
    all_pairs = []
    graphml_files = sorted(DATA_DIR.glob("*.graphml"))
    print(f"Found {len(graphml_files)} graphML files")

    for graphml_file in graphml_files:
        try:
            pairs = parse("graphml", str(graphml_file))
            all_pairs.extend(pairs)
        except Exception as e:
            print(f"  Skipping {graphml_file.name}: {e}")

    print(f"Generated {len(all_pairs)} total QA pairs")

    # 80/20 train/eval split
    random.shuffle(all_pairs)
    split = int(len(all_pairs) * 0.8)
    train_pairs = all_pairs[:split]
    eval_pairs = all_pairs[split:]

    save_jsonl(train_pairs, OUTPUT_JSONL)
    save_jsonl(eval_pairs, EVAL_JSONL)
    print(f"Train: {len(train_pairs)} pairs -> {OUTPUT_JSONL}")
    print(f"Eval:  {len(eval_pairs)} pairs -> {EVAL_JSONL}")
    return len(train_pairs)


if __name__ == "__main__":
    generate()
