# Agentic RAG: Verify-Before-Emit Architecture — Package

## Overview

This repository contains all source code, evaluation scripts, query identifiers, and annotation guidelines required to replicate the experimental results reported in the manuscript. The system implements a **seven-tier Agentic RAG architecture** with an Autonomous Grounding Validator (Verify-Before-Emit) that enforces evidence-grounded outputs before delivery.

### Architecture Summary

```
Query → Hybrid Retrieval (BM25 + Dense) → Cross-Encoder Reranking → Generation
     → Self-Reflection (LLM-as-Judge) → Repair Orchestration → Emit / Refuse
```

**Key design features:**
- Bounded agentic loop (max 2 attempts per query)
- Deterministic repair routing (failure code → strategy mapping)
- Diagnostic refusal with audit trace
- Smart model routing (Flash → Pro escalation)

## Repository Structure

```
├── src/                        Core system modules
│   ├── agentic_loop.py         Bounded agentic controller
│   ├── agentic_rag.py          Base RAG orchestrator
│   ├── enhanced_agentic_rag.py Tier 1–6 enhanced system
│   ├── self_reflection.py      LLM-as-Judge verification
│   ├── hybrid_search.py        BM25 + Dense retrieval (RRF fusion)
│   ├── reranking.py            Cross-encoder reranker
│   ├── query_rewriting.py      Query rewrite / decompose / multi-query
│   ├── multihop_reasoning.py   Multi-hop question decomposition
│   ├── embedding_cache.py      Embedding cache layer
│   ├── citation_system.py      Citation extraction
│   ├── dataset_builder.py      Benchmark dataset construction
│   └── evaluation_framework.py Evaluation engine (RAGAS, IR, semantic)
│
├── benchmark/                  Benchmark reproduction
│   ├── run_benchmark.py        Main entry point
│   └── compute_agentic_metrics.py  Post-hoc metric computation
│
├── config/
│   └── SYSTEM_CONFIGURATION.md Full parameter reference table
│
├── data/                       Benchmark data
│   ├── query_ids_nq.txt        250 Natural Questions query IDs
│   ├── query_ids_hotpotqa.txt  250 HotpotQA query IDs
│   └── annotation_guidelines.md Human validation protocol
│
└── results/                    (empty; populated by benchmark run)
```

## Quick Start

### 1. Prerequisites

- Python 3.10+
- Google Gemini API key (for generation + embeddings)
- OpenAI API key (for RAGAS GPT-4o faithfulness evaluation only)

### 2. Installation

```bash
# Clone
git clone https://github.com/<your-username>/agentic-rag-replication.git
cd agentic-rag-replication

# Virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your GEMINI_API_KEY and OPENAI_API_KEY
```

### 3. Reproduce the Benchmark

```bash
cd benchmark
python run_benchmark.py --samples 500 --workers 1
```

This will:
1. Download NQ-Open and HotpotQA datasets (250 samples each, seed=42)
2. Configure 4 system variants (Full, No Routing, No Validation, Vanilla)
3. Run all 500 queries × 4 systems
4. Compute RAGAS faithfulness, BERTScore, IR metrics, and agentic metrics
5. Output results to `results/`

**Estimated runtime:** ~4–6 hours (500 queries × 4 systems, sequential)

## System Configuration

All system parameters are documented in [`config/SYSTEM_CONFIGURATION.md`](config/SYSTEM_CONFIGURATION.md). Key parameters:

| Parameter | Value |
|:---|:---|
| **Embedding** | `text-embedding-004` (Google) |
| **Retrieval** | Hybrid: BM25 (w=0.6) + Dense (w=0.4), RRF k=60 |
| **BM25** | k₁=1.5, b=0.75 |
| **Top-K** | 20 candidates → reranked to 8 |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **Max attempts** | 2 (Attempt-1 → Repair → Attempt-2) |
| **Verification** | LLM Judge (Flash, T=0.0, JSON) + heuristics |
| **Pass threshold** | score ≥ 0.7, no FAILED checks |
| **Generation** | Gemini 2.5 Flash (default) / Pro (escalated) |
| **Faithfulness** | RAGAS + GPT-4o judge, binarized at ≥ 0.55 |
| **Random seed** | 42 |

## Experimental Conditions

| Condition | Model | Routing | Validation | Retrieval |
|:---|:---|:---|:---|:---|
| **Governed Efficient (Ours)** | Flash | Smart routing ON | Full verification | Hybrid + Reranker |
| **Governed Frontier** | Pro (forced) | OFF | Full verification | Hybrid + Reranker |
| **No Validation** | Flash | OFF | Disabled | Hybrid + Reranker |
| **Vanilla RAG** | Flash | OFF | Disabled | Hybrid only |
| **Vanilla Pro Baseline** | Pro | OFF | Disabled | Hybrid only |

*Note: The raw datasets mathematically proving the Vanilla Pro baseline (`Vanilla_Pro_Raw_Outputs.csv`) and the Open-Weights generalizability pilot from Appendix D (`Cross_Family_Pilot_Raw_Data.csv` for Llama 3 70B and Mixtral 8x7B) are included in the repository root to ensure full empirical reproducibility.*

**Parity statement:** Retrieval index, embedding model, BM25 parameters, RRF fusion weights, Top-K, reranker model, and reranking Top-N were held identical between Governed Efficient and Governed Frontier. The sole manipulated independent variable was the generative language model.

## Faithfulness Evaluation

Faithfulness was evaluated using the RAGAS framework with GPT-4o as an independent external judge (chain-of-thought claim extraction and cross-checking). Continuous scores were binarized at ≥ 0.55 for the TOST equivalence test.

Human validation was conducted on an expanded 150-query sample, achieving Cohen's κ = 0.940 inter-rater reliability. See [`data/annotation_guidelines.md`](data/annotation_guidelines.md) for the full protocol.

*Note: The master empirical execution logs mathematically enforcing the TOST equivalence statistical findings are permanently archived in `results/Primary_500_Sample_Results.csv`. The full human validation verification grid is archived in `results/Human_Validation_150_Samples.csv`.*

## Cost Model

Token costs are computed using Gemini 2.5 pricing (as of January 2026):

| Model | Input | Output |
|:---|:---|:---|
| Gemini 2.5 Flash | $0.30 / 1M tokens | $2.50 / 1M tokens |
| Gemini 2.5 Pro | $1.25 / 1M tokens | $10.00 / 1M tokens |

Cost per query = (prompt_tokens / 1M × input_rate) + (completion_tokens / 1M × output_rate), summed across all agentic loop iterations. Retrieval and reranking compute costs are excluded as they run locally.

## Citation

```bibtex
@article{hakimi2026governing,
  title   = {Governing Generative AI for Reliable Decision Support: 
             A Seven-Tier Agentic RAG Architecture with 
             Autonomous Grounding Validation},
  author  = {Hakimi, [First Name]},
  journal = {[Target Journal]},
  year    = {2026}
}
```

## License

This project is provided for academic use and peer review purposes. See [LICENSE](LICENSE) for details.
