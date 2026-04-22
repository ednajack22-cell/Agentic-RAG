# Experimental configuration and reproducibility details
The appendix provides the configuration details required to reproduce the reported results and clarifies how retrieval and reranking parity were enforced across the two governed conditions.
Table A1. Complete system configuration parameters.

| Parameter | Value |
| --- | --- |
| Corpus | 250 NQ-Open plus 250 HotpotQA (N = 500 queries) |
| Chunking | 1,500 tokens (Google Gemini File Search, server-managed) |
| Embedding model | text-embedding-004 (Google) |
| Retrieval strategy | Hybrid: BM25 (w = 0.6) plus dense (w = 0.4), RRF k = 60 |
| BM25 parameters | k1 = 1.5; b = 0.75 |
| Top-K (initial retrieval) | 20 candidates |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 (HuggingFace) |
| Rerank keep (normal) | 8 passages |
| Rerank keep (strict repair) | 6 passages |
| Max agentic iterations | 2 |
| Verification | LLM judge (Gemini 2.5 Flash, T = 0.0, JSON) plus heuristics |
| Acceptance threshold | Score >= 0.7 and no FAILED checks |
| Generation model (efficient) | Gemini 2.5 Flash |
| Generation model (frontier) | Gemini 2.5 Pro |
| Generation temperature | Gemini API default (not overridden) |
| Judge temperature | 0.0 |
| Random seed | 42 |
| Faithfulness evaluator | RAGAS plus GPT-4o |
| Faithfulness binarization | >= 0.55 = Pass |
| Repair strategies | 7 implementation-level routes; 4 primary strategies discussed in main text |
| Cost (Flash) | 0.30 USD / 1M input tokens; 2.50 USD / 1M output tokens |
| Cost (Pro) | 1.25 USD / 1M input tokens; 10.00 USD / 1M output tokens |
| Software stack | Python 3.13.2, Google Gemini 2.5 API, RAGAS 0.2.6, HuggingFace Transformers 4.40.0 |
| Hardware | Consumer-grade Windows workstation |
| Repository | https://github.com/ednajack22-cell/Agentic-RAG |

Retrieval and reranking parity statement. The retrieval index, embedding model, BM25 parameters, RRF fusion weights, Top-K setting, reranker model, and reranking Top-N were held identical between the Governed Efficient (Gemini 2.5 Flash) and Governed Frontier (Gemini 2.5 Pro) conditions. No retrieval or reranking parameter differed across these two conditions. The sole manipulated independent variable was the generative language model. Ungoverned baselines. The Vanilla Flash and Vanilla Pro configurations used the same retrieval pipeline (identical BM25 parameters, embedding model, RRF fusion, reranker, and Top-8 passages) but omitted the agentic governance loop (verification, repair, and refusal). This ensured that performance differences between vanilla and agentic conditions within each model tier are attributable to the governance architecture rather than to retrieval variation.
Deterministic repair routing. Repair strategies were assigned deterministically from failure codes rather than selected randomly: missing evidence → Retrieve More; contradiction → Strict Rerank; incomplete response → Query Decomposition; low confidence → Multi-Query Expansion; citation error → Strict Rerank; off-topic retrieval → Query Rewrite.
Repository contents. The supplementary repository contains the evaluation scripts, benchmark configuration files, sampled query identifiers with dataset labels, the verification-judge prompt, human annotation guidelines, anonymized output files, repair logs, and cost-accounting scripts.