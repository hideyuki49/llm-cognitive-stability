# SCRC: Structural Consistency & Reasoning Capability Evaluation

This repository provides a reproducible evaluation framework for analyzing
**structural consistency, abstention behavior, and reasoning stability**
across large language models (LLMs).

The framework is designed to reveal *qualitative differences in model behavior*
that are not captured by conventional accuracy benchmarks, focusing instead on
**how models handle uncertainty, ambiguity, constraint-following, and semantic drift**.

---

## Repository Structure

.
├── README.md
├── scrc_eval.py # Main evaluation script
├── tasks.json # Task definitions
└── logs/
├── summary.txt # Human-readable summaries per model
└── scrc_out.json # Raw structured outputs (machine-readable)

---

## Evaluated Models

This repository contains evaluation results for the following models:

- **Qwen series**
  - Qwen2.5-0.5B / 1.5B / 3B / 7B / 14B / 32B (Instruct)
  - Qwen2-57B-A14B-Instruct (MoE)
  - Qwen3-14B

- **Mistral series**
  - Mistral-7B-Instruct-v0.3
  - Mixtral-8x7B-Instruct-v0.1 (MoE)

These models span:
- Dense vs MoE architectures
- Small → large parameter regimes
- Distinct training and alignment philosophies

---

## Execution Environment

All evaluations were conducted under the following environment:

- **GPU**: NVIDIA H200 SXM (141GB VRAM)
- **CUDA**: 12.1 (nvrtc 12.1.105, driver supports up to CUDA 12.8)
- **Framework**: PyTorch + HuggingFace Transformers
- **Precision**: fp16 / bf16 (model default)
- **Inference mode**: single-GPU, no quantization

Models were evaluated sequentially under identical task definitions,
with fixed random seeds to ensure comparability.

---

## Experimental Objective

The goal of SCRC is **not** to rank models by raw correctness.

Instead, we aim to characterize:

- How models behave when **answers are undefined**
- Whether abstention (`I don't know`) is used appropriately
- How stable reasoning remains under **noise, negation, or scope shifts**
- Whether models preserve the **semantic intent** of a task under perturbation

In short, SCRC evaluates **behavioral structure**, not just outputs.

---

## Task Design Overview

The evaluation consists of multiple task classes, including:

### 1. Constraint Following
- Strict output formats
- Instruction adherence under perturbation

### 2. Forced-Unknown Tasks
- Nonexistent entities (e.g. Atlantis, fake standards)
- Correct behavior requires explicit abstention

### 3. Reason-Required Tasks (AWR)
- Answers must include an explicit causal explanation
- Evaluates justification discipline

### 4. Clarification-First Tasks (CQS)
- Ambiguous problem statements
- Models should ask clarifying questions before answering

### 5. Premise Drift Detection
- Negation
- Quantifier shifts
- Metric substitution

### 6. Recovery After Misdirection (RRS)
- Models must ignore irrelevant or misleading instructions
- Tests goal preservation

Each task is executed across multiple perturbation levels (δ),
with fixed random seeds for reproducibility.

---

## Metrics (High-Level)

The framework reports structured metrics including:

- **CIR**: Constraint Integrity Rate  
- **CASm**: Constraint Adherence Similarity  
- **FUV**: Forced-Unknown Validity  
- **AWR**: Answer-With-Reason compliance  
- **CQS**: Clarification-Question Sensitivity  
- **RRS**: Recovery Robustness Score  

These metrics allow separation of:
- Correctness
- Abstention discipline
- Structural reasoning stability

---

## Key Observations (Qualitative)

Across the evaluated models, we observe:

- Clear **developmental stages** as parameter count increases
- Distinct differences between **dense and MoE architectures**
- Strong signatures of **training philosophy** rather than size alone
- Some models behave as “excellent students” — revealing evaluator bias
  before revealing their own weaknesses

These behaviors are consistent across repeated runs,
indicating **intrinsic model characteristics**, not sampling noise.

---

## Reproducibility

- All evaluations use fixed random seeds
- Raw JSON outputs are preserved in `logs/`
- Summaries are directly derived from raw outputs

The repository is intended to support:
- Independent re-analysis
- Metric reinterpretation
- Cross-paper comparison

---

## Intended Use

This dataset is published to support:

- Methodological discussion
- Epistemic and philosophical analysis of LLM behavior
- Comparative studies of model alignment and reasoning stability

It is **not** intended as a leaderboard.

---

## Citation

If you reference this evaluation framework or dataset, please cite:

> Chino, H. (2026).  
> *Structural Consistency and Reasoning Capability Evaluation of Large Language Models.*  
> (Manuscript under submission)

Raw data is released via GitHub.  
Zenodo archival will be provided where journal policy permits.