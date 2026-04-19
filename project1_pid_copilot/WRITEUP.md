# P&ID Safety Copilot — Technical Write-Up

## Overview

The P&ID Safety Copilot is a fine-tuned Gemma 4 E4B model that answers
safety-critical questions about Piping and Instrumentation Diagrams (P&IDs).
Given a component graph representation of a P&ID, the model identifies isolation
valve paths, lists components by type, and flags single-point failure risks —
answering questions a process safety engineer might ask during a HAZOP review.

## Motivation

Industrial accidents at chemical plants, water treatment facilities, and oil & gas
installations frequently stem from failures to correctly trace isolation paths or
identify single-point failures on P&IDs. Engineers currently trace these manually,
a tedious and error-prone process. This model enables instant, auditable answers in
natural language, reducing human error and response time in safety-critical scenarios.

This directly addresses the **Climate & Environmental Impact** theme: industrial
accidents are a leading cause of environmental contamination.

## Technical Approach

### Data Pipeline

P&IDs were sourced from the **PID2Graph dataset** (Zenodo 14803338, CC BY-SA),
which contains 500 annotated P&ID images with GraphML topology annotations.

Graph traversal via NetworkX generates QA pairs automatically:
- **Component listing:** iterate node attributes → `List all valves on this P&ID`
- **Isolation paths:** BFS from pump/vessel nodes → find shutoff valves within 3 hops
- **Single-point failures:** find degree-1 upstream nodes for vessels
- **Instrumentation audit:** filter nodes by label containing "instrument"
- **Component counts:** aggregate by label type

This produced **3,986 training pairs** and **997 eval pairs** without any manual
annotation beyond the source dataset's graph labels.

### Model

- **Base:** `google/gemma-4-e4b-it` (4B edge instruct)
- **Method:** LoRA fine-tuning via Unsloth (r=16, α=32, dropout=0.05)
- **Quantization:** 4-bit NF4 during training
- **Hardware:** Kaggle T4 (15 GB VRAM, free tier)
- **Training:** 3 epochs, batch size 2 + gradient accumulation 8 (effective batch 16)

### Evaluation

Automatic evaluation uses Token F1 on held-out valve/component queries.
The structured JSON output format makes exact-match evaluation reliable.

## Offline Capability

After fine-tuning, the adapter is merged and exported to GGUF 4-bit format using
Unsloth's `save_pretrained_gguf`. This enables inference via llama.cpp with no
internet connection or GPU required — target hardware for remote industrial sites.

## Judging Criteria Alignment

| Criterion | Evidence |
|---|---|
| Innovation | Graph-traversal QA generation from P&ID topology; no existing P&ID copilot |
| Impact | Industrial accident prevention; 40,000+ refinery accidents/year worldwide |
| Technical | LoRA fine-tune on T4; structured JSON output; GGUF export for offline use |
| Accessibility | Free hardware (Kaggle T4); open datasets (CC BY-SA); offline GGUF deployment |
