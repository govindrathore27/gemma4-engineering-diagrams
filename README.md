# Gemma 4 Engineering Diagrams

**[Kaggle Gemma 4 Good Hackathon](https://www.kaggle.com/competitions/gemma-4-good-hackathon)**
| Deadline: 2026-05-18 | Team: govindsrathore

Three independent Gemma 4 E4B fine-tuning projects targeting industrial safety and
vocational education. All models run fully offline on consumer hardware (T4 GPU or
CPU via llama.cpp).

---

## Projects

| # | Project | Theme | Notebook |
|---|---|---|---|
| 1 | [P&ID Safety Copilot](project1_pid_copilot/) | Climate & Environmental Impact | [notebook.ipynb](project1_pid_copilot/notebook.ipynb) |
| 2 | [Circuit Diagram Tutor](project2_circuit_tutor/) | Education in Underserved Communities | [notebook.ipynb](project2_circuit_tutor/notebook.ipynb) |
| 3 | [Engineering Drawing Tutor](project3_drawing_tutor/) | Education in Underserved Communities | [notebook.ipynb](project3_drawing_tutor/notebook.ipynb) |

---

## Quick Start

### Prerequisites

- Python 3.11+
- Kaggle CLI (`pip install kaggle`)
- Kaggle API token

### Install

```bash
git clone https://github.com/govindrathore27/gemma4-engineering-diagrams.git
cd gemma4-engineering-diagrams
pip install -r requirements.txt
```

### Download Datasets

```bash
export KAGGLE_API_TOKEN=<your_token>
python shared/data_pipeline/download_datasets.py
```

### Generate Training Data

```bash
# Project 1: P&ID
python project1_pid_copilot/train_config.py

# Project 2: Circuit
python project2_circuit_tutor/train_config.py

# Project 3: Drawing
python project3_drawing_tutor/train_config.py
```

### Run Tests

```bash
pytest tests/ -v
```

---

## Architecture

```
gemma4-engineering-diagrams/
├── shared/
│   ├── data_pipeline/
│   │   ├── download_datasets.py     # Zenodo, Kaggle, HuggingFace fetch
│   │   └── generate_qa_pairs.py     # 6 parsers → JSONL QA pairs
│   ├── model/
│   │   ├── train.py                 # Unsloth LoRA fine-tune loop
│   │   ├── inference.py             # Load adapter + run inference
│   │   └── quantize.py              # Export to GGUF 4-bit
│   └── eval/
│       └── metrics.py               # F1, exact-match, BLEU
├── project1_pid_copilot/            # P&ID Safety Copilot
├── project2_circuit_tutor/          # Circuit Explanation Tutor
├── project3_drawing_tutor/          # Engineering Drawing Tutor
└── tests/                           # 31 unit tests
```

## Stack

| Component | Tool |
|---|---|
| Base model | Gemma 4 E4B (`google/gemma-4-e4b-it`) |
| Fine-tuning | Unsloth LoRA 4-bit NF4 |
| Training hardware | Kaggle T4 (15 GB VRAM, free tier) |
| Graph parsing | NetworkX (GraphML → BFS/DFS) |
| Offline export | Unsloth → GGUF 4-bit (llama.cpp) |

## Training Datasets on Kaggle

| Project | Dataset |
|---|---|
| P&ID Safety Copilot | [govindsrathore/gemma4-pid-train-data](https://www.kaggle.com/datasets/govindsrathore/gemma4-pid-train-data) |
| Circuit Tutor | [govindsrathore/gemma4-circuit-train-data](https://www.kaggle.com/datasets/govindsrathore/gemma4-circuit-train-data) |
| Drawing Tutor | [govindsrathore/gemma4-drawing-train-data](https://www.kaggle.com/datasets/govindsrathore/gemma4-drawing-train-data) |
