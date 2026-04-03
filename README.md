<div align="center">

# 🩺 CoEval

**Medical LLM Evaluation Framework**

*Part of the [Chain-of-Evidence](https://github.com/lunit-io) project by [Lunit](https://www.lunit.io)*

[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Hydra](https://img.shields.io/badge/config-Hydra-89b8cd?logo=yaml&logoColor=white)](https://hydra.cc/)
[![DeepEval](https://img.shields.io/badge/metrics-DeepEval-6c5ce7)](https://docs.confident-ai.com/)
[![HuggingFace](https://img.shields.io/badge/datasets-HuggingFace-ffd21e?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Pydantic v2](https://img.shields.io/badge/schemas-Pydantic%20v2-e92063?logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)
[![Rich](https://img.shields.io/badge/console-Rich-00b894)](https://rich.readthedocs.io/)
[![mise](https://img.shields.io/badge/toolchain-mise-5B4FC4?logo=mise&logoColor=white)](https://mise.jdx.dev/)
[![uv](https://img.shields.io/badge/pkg-uv-de5fe9?logo=uv&logoColor=white)](https://docs.astral.sh/uv/)
[![Ruff](https://img.shields.io/badge/lint-Ruff-d7ff64?logo=ruff&logoColor=black)](https://docs.astral.sh/ruff/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

[Quick Start](#-quick-start) · [Datasets](#-datasets) · [Metrics](#-metrics) · [Configuration](#%EF%B8%8F-configuration) · [Extend](#-extend)

</div>

---

## 📰 What's New

| Date | Version | Update |
|------|---------|--------|
| 2026-04-02 | **v0.1.0** | Initial release — 17 medical datasets, 9 metrics, async evaluation pipeline |

---

CoEval is an async-first evaluation framework built by Lunit's Chain-of-Evidence team for benchmarking LLMs on medical tasks. It is designed to:

- **Evaluate your own model** — Point to any OpenAI-compatible endpoint (vLLM, SGLang, OpenAI, HuggingFace TGI) and run standardized medical benchmarks.
- **Compare models fairly** — Run multiple models against the same datasets, metrics, and prompts for apples-to-apples comparison.
- **Scale easily** — Adding a new dataset is ~50 lines of Python + one YAML file. Adding a metric is even less.

It ships with **17 medical datasets**, **9 metrics** (deterministic + LLM-as-judge), and a Hydra-based config system for fully reproducible evaluations.

---

## 🚀 Quick Start

### 1. Clone and install

```bash
git clone https://github.com/lunit-io/coeval.git
cd coeval
```

```bash
mise trust          # Required on first clone — trusts mise.toml config
mise run sync       # Installs Python 3.13 + deps via mise/uv
# or
uv sync --dev       # If you already have uv and Python 3.13 (skip mise)
```

### 2. Serve your model

CoEval works with any OpenAI-compatible API:

```bash
# SGLang (recommended)
python -m sglang.launch_server --model learning-unit/your-model --port 8000

# vLLM
vllm serve learning-unit/your-model --port 8000

# ☁️ Hosted API (OpenAI, Azure, etc.)
export OPENAI_API_KEY=sk-...
```

### 3. Configure and run

`client.api_base` and `client.model` are required — pass them via CLI:

```bash
# Smoke test (5 samples)
mise run eval -- client.api_base=http://localhost:8000/v1 client.model=your-model num_samples=5

# Full benchmark (default: PubMedQA)
mise run eval -- client.api_base=http://localhost:8000/v1 client.model=your-model

# Single dataset
mise run eval -- client.api_base=http://localhost:8000/v1 client.model=your-model datasets=medqa

# All 17 datasets
mise run eval -- client.api_base=http://localhost:8000/v1 client.model=your-model datasets=all
```


### 4. (Optional) LLM-as-judge datasets

Some datasets (e.g., HealthBench) use an LLM judge to score responses instead of exact match. These require an OpenAI API key for the judge model.

Set it in `mise.toml` (recommended — keeps it out of your shell history):

```toml
# mise.toml → [env]
OPENAI_API_KEY = "sk-..."
```

Or export it directly:

```bash
export OPENAI_API_KEY=sk-...
```

Then run:

```bash
# Run HealthBench — your model generates responses, gpt-4.1 grades them
mise run eval -- datasets=healthbench_consensus
```

> **Note:** `OPENAI_API_KEY` is used for both the model server and the judge model. Most MCQ datasets (MedQA, MedMCQA, etc.) use deterministic scoring and do **not** require a judge model.

### 5. Check results

Results are saved to `evaluation_outputs/`:

```
📁 evaluation_outputs/YYYY-MM-DD/HH-MM-SS/
├── results_<dataset>.json        # Per-sample: input, prediction, scores
├── summary_<dataset>.json        # Aggregated metrics + breakdown
└── summary_combined.json         # Cross-dataset leaderboard
```

---

## 🧰 Datasets

All datasets are evaluated as **MCQ (multiple-choice question)** unless noted otherwise. The model selects an answer letter (A/B/C/D) and is scored by exact match.

### 17 datasets

| Dataset | Key | Source | Task | Metric |
|---------|-----|--------|------|--------|
| [MedQA](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options) | `medqa` | USMLE | 4-option MCQ | MCQ Accuracy |
| [MedMCQA](https://huggingface.co/datasets/lighteval/med_mcqa) | `medmcqa` | AIIMS/PGI | 4-option MCQ | MCQ Accuracy |
| [MMLU-Pro Health](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) | `mmlu_pro_health` | MMLU-Pro subset | 10-option MCQ | MCQ Accuracy |
| [HeadQA](https://huggingface.co/datasets/alesi12/head_qa_v2) | `headqa` | Spanish medical exams | 4-option MCQ | MCQ Accuracy |
| [CareQA](https://huggingface.co/datasets/HPAI-BSC/CareQA) | `careqa` | USMLE Step 1-3 | 4-option MCQ | MCQ Accuracy |
| [M-ARC](https://huggingface.co/datasets/mkieffer/M-ARC) | `m_arc` | Medical ARC | 4-option MCQ | MCQ Accuracy |
| [MetaMedQA](https://huggingface.co/datasets/maximegmd/MetaMedQA) | `metamedqa` | Meta medical eval | 4-option MCQ | MCQ Accuracy |
| [MedExQA](https://huggingface.co/datasets/bluesky333/MedExQA) | `medexqa` | Medical specialties | 4-option MCQ | MCQ Accuracy |
| [MedXpertQA](https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA) | `medxpertqa` | Expert medical QA | 4-option MCQ | MCQ Accuracy |
| [MedConceptsQA](https://huggingface.co/datasets/ofir408/MedConceptsQA) | `medconceptsqa` | Medical ontology | 3-option MCQ | MCQ Accuracy |
| [Medbullets](https://huggingface.co/datasets/mkieffer/Medbullets) | `medbullets` | Step 2 practice | 4/5-option MCQ | MCQ Accuracy |
| [KorMedMCQA](https://huggingface.co/datasets/sean0042/KorMedMCQA) | `kormedmcqa` | Korean medical exams | 5-option MCQ | MCQ Accuracy |
| [MedHallu](https://huggingface.co/datasets/UTAustin-AIHealth/MedHallu) | `medhallu` | Hallucination detection | Binary classification | Macro F1 |
| [MedCalc](https://huggingface.co/datasets/ncbi/MedCalc-Bench) | `medcalc` | Clinical calculation | Open-ended numeric | Numeric Accuracy |
| [PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA) | `pubmedqa` | 3-option MCQ (Yes/No/Maybe) | MCQ Accuracy |
| [HealthBench](https://huggingface.co/datasets/openai/HealthBench) | `healthbench_consensus` | Open-ended multi-turn | LLM-as-judge (rubric) |
| [AttributionBench](https://huggingface.co/datasets/osunlp/AttributionBench) | `attributionbench` | Binary classification (attributable/not) | Macro F1 |

```bash
mise run eval -- datasets=medqa              # Single dataset
mise run eval -- datasets=all                # All 17 datasets
mise run eval -- datasets=all num_samples=50 # Quick run, 50 samples each
```

---

## 📊 Metrics

### Deterministic (no LLM required)

| Metric | Class | Description |
|--------|-------|-------------|
| MCQ Accuracy | `MCQAccuracyMetric` | Robust answer letter extraction + exact match |
| Classification | `ClassificationMetric` | Label extraction + exact match |
| Numeric Accuracy | `NumericAccuracyMetric` | Numeric value comparison with tolerance |

### LLM-as-Judge (requires judge model)

| Metric | Class | Description |
|--------|-------|-------------|
| HealthBench Rubric | `HealthBenchRubricMetric` | Per-criterion rubric scoring |
| Faithfulness | `FaithfulnessMetric` | Is the answer grounded in context? |
| Answer Relevancy | `AnswerRelevancyMetric` | Is the answer relevant to the question? |
| Contextual Precision | `ContextualPrecisionMetric` | Does context contain relevant info? |
| Contextual Recall | `ContextualRecallMetric` | Is all relevant info retrieved? |

### Aggregation

| Strategy | Key | Use case |
|----------|-----|----------|
| Simple Average | `default` | Most MCQ datasets |
| Weighted Average | `weighted_avg` | Grouped sub-datasets (e.g., KorMedMCQA) |
| Macro F1 | `f1` | Classification tasks (e.g., AttributionBench) |

---

## ⚙️ Configuration

CoEval uses [Hydra](https://hydra.cc/) for composable configuration. Everything is overridable from the CLI.

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `OPENAI_API_KEY` | API key (model server + LLM-as-judge) | `sk-...` |

### CLI Overrides

```bash
# Override any config value
mise run eval -- datasets=medqa num_samples=100 client.temperature=0.3

# Change system prompt
mise run eval -- 'system_prompt="Answer concisely."'

# Swap judge model for HealthBench
mise run eval -- datasets=healthbench_consensus datasets/metrics/judge@healthbench_judge=gpt-4.1
```

---

## 🔧 Extend

CoEval is designed to be easily extensible:

- **[Adding a new dataset](src/coeval/datasets/README.md)** — ~50 lines of Python + one YAML config
- **[Adding a new metric](src/coeval/metrics/README.md)** — Extend `DeterministicMetric` or use DeepEval's LLM-as-judge

---

## 🏗️ Project Structure

```
src/coeval/
├── main.py                 # CLI entry point (Hydra)
├── clients/
│   └── passthrough.py      # PassthroughClient (OpenAI-compatible)
├── llm/                    # Low-level async LLM client
│   ├── client.py           # LLMClient (llama-index OpenAILike wrapper)
│   ├── config.py           # LLMConfig dataclass
│   └── exceptions.py       # Error hierarchy (timeout, rate limit, etc.)
├── core/
│   ├── runner.py           # EvalRunner — async generation + scoring pipeline
│   ├── evaluate.py         # Metric dispatch (deterministic vs LLM-as-judge)
│   ├── schema.py           # EvalResult, EvalSummary (Pydantic models)
│   └── types.py            # Centralized type aliases and deepeval re-exports
├── datasets/
│   ├── base.py             # GoldenDatasetBase, MultiTurnDatasetBase
│   ├── medqa.py            # 18 dataset loaders (one file each)
│   └── ...
├── metrics/                # 3 deterministic + 5 LLM-as-judge metrics
├── util/                   # Parsers, score aggregation, Rich console
└── conf/                   # Hydra YAML configs
    ├── config.yaml         # Root config
    ├── client/             # LLM client configs
    ├── runner/             # Runner configs
    └── datasets/           # Per-dataset configs (dataset + metric + aggregator)
```

---

## 🛠️ Development

```bash
mise run sync       # Install dependencies
mise run test       # Run unit tests
mise run lint       # Ruff linter
mise run format     # Ruff formatter
```

---

## License

Apache 2.0
