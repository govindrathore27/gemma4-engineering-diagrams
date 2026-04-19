# Circuit Diagram Explanation Tutor — Technical Write-Up

## Overview

The Circuit Diagram Explanation Tutor is a fine-tuned Gemma 4 E4B model that
explains electronic circuits in plain language, generates SPICE-compatible netlists,
and answers what-if questions about component changes. It is designed for students
in underserved communities who learn from textbooks but lack access to expensive
simulation tools like LTspice or Multisim.

## Motivation

Hundreds of millions of students in developing countries study electronics using
photocopied diagrams with no interactive software. A model that can explain "what
does this circuit do?" from a component list — and generate a simulation-ready
netlist — acts as a zero-cost, offline teaching assistant.

This directly addresses the **Education in Underserved Communities** theme.

## Technical Approach

### Data Pipeline

Circuit annotations were sourced from **circuit1k**, a dataset of 1,000 annotated
circuit images in YOLO bounding-box format. Each annotation file lists component
class IDs (resistor, capacitor, inductor, diode, battery) with bounding box positions.

QA pairs generated per circuit:
- **Circuit description:** summarize component types and topology pattern
- **Component count:** structured JSON of counts per type
- **Current limiter:** identify the resistor protecting a diode/LED
- **SPICE netlist:** auto-generate netlist from component list using standard node labeling
- **What-if:** explain effect of doubling resistor values (Ohm's law application)

This produced **4,355 QA pairs** (3,484 train + 871 eval) from 1,000 circuit annotations.

When the CGHD dataset (Zenodo 14042961, 59 circuit component classes) is fully extracted,
re-running `train_config.py` automatically incorporates its JSON annotations via the
existing `parse_cghd` parser for up to ~2,500 additional training pairs.

### Model

- **Base:** `google/gemma-4-e4b-it` (4B edge instruct)
- **Method:** LoRA fine-tuning via Unsloth (r=16, α=32, dropout=0.05)
- **Quantization:** 4-bit NF4 during training
- **Hardware:** Kaggle T4 (15 GB VRAM, free tier)
- **Training:** 3 epochs, batch size 2 + gradient accumulation 8

### Evaluation

Automatic evaluation uses BLEU-1 on plain-language explanations and Token F1 on
structured JSON outputs (component counts, netlists).

## Offline Capability

GGUF 4-bit export via Unsloth enables CPU-only inference via llama.cpp — suitable
for schools with unreliable power, limited RAM, and no internet.

## Judging Criteria Alignment

| Criterion | Evidence |
|---|---|
| Innovation | Auto-SPICE netlist generation from bounding-box annotations; novel dataset pipeline |
| Impact | Serves 100M+ electronics students without simulation software access |
| Technical | LoRA + 4-bit training on T4; BLEU + F1 evaluation; GGUF offline export |
| Accessibility | No software beyond llama.cpp needed; works on decade-old laptops |
