# System Configuration — Agentic RAG

> **Purpose**: Reproducibility appendix for IS/DSR paper submission.  
> All values extracted from source code as deployed in the January 2026 benchmark.

---

## 1. DATA / CHUNKING

```text
chunk_size:       1500 tokens (Mem0 config for benchmark ingestion)
chunk_overlap:    Managed by Gemini File Search API (server-side, not user-configurable)
preprocessing:    None (raw text ingested; no boilerplate/table/header removal)
file_types:       PDF, TXT, MD, DOCX (via Gemini File Search upload)
chunking_engine:  Google Gemini File Search API (server-managed chunking)
parent_doc_retrieval:  Available but NOT used in benchmark
                       (parent_chunk_size=800, child_chunk_size=150 — not active)
```

**Source**: [`run_evaluation_benchmarks.py:534`](../src/run_evaluation_benchmarks.py#L534), [`parent_document_retrieval.py:320`](../src/parent_document_retrieval.py#L320)

---

## 2. RETRIEVAL

```text
retriever:          Hybrid (BM25 + Dense Vector Search)
embedding_model:    models/text-embedding-004 (Google)
top_k:              20 (initial retrieval candidates)
filters:            None active in benchmark
fusion_method:      Reciprocal Rank Fusion (RRF)
rrf_k:              60 (constant in RRF formula)
bm25_weight:        0.6
dense_weight:       0.4
bm25_k1:            1.5 (term frequency saturation)
bm25_b:             0.75 (length normalization)
```

**Source**: [`agentic_loop.py:62-70`](../src/agentic_loop.py#L62-L70) (`RetrievalParams`), [`hybrid_search.py:170-184`](../src/hybrid_search.py#L170-L184) (`HybridSearchEngine.__init__`), [`run_evaluation_benchmarks.py:527-533`](../src/run_evaluation_benchmarks.py#L527-L533)

---

## 3. RERANKING

```text
reranker_model:     cross-encoder/ms-marco-MiniLM-L-6-v2 (HuggingFace)
rerank_candidates:  20 (top_k from retrieval stage)
keep_top:           8 (rerank_top_n in normal mode)
                    6 (rerank_top_n in strict_rerank repair mode)
batch_size:         8
use_cache:          True
two_stage_pipeline: TwoStageRetriever (retrieve_k=20 → rerank_k=5)
```

**Source**: [`enhanced_agentic_rag.py:99-100`](../src/enhanced_agentic_rag.py#L99-L100) (`EnhancedConfig`), [`reranking.py:55-79`](../src/reranking.py#L55-L79) (`CrossEncoderReranker.__init__`), [`agentic_loop.py:356-359`](../src/agentic_loop.py#L356-L359) (strict_rerank repair)

---

## 4. AGENTIC GOVERNANCE LOOP

```text
max_iters:              2 (Attempt-1 → self-reflection → repair → Attempt-2 → stop)
initial_confidence:     0.8
low_confidence_threshold: 0.55
acceptance_threshold:   overall_score ≥ 0.7 AND no FAILED validation checks

verify_method:          LLM-as-Judge (Gemini 2.5 Flash, temperature=0.0)
                        + Heuristic checks (term overlap, citation coverage, completeness)
verify_components:      3 checks in parallel:
                        1. Factual Consistency (LLM or heuristic)
                        2. Citation Verification (rule-based)
                        3. Completeness Check (rule-based)

refusal_rule:           If both attempts fail:
                        - MISSING_EVIDENCE or OFF_TOPIC → ask clarification question
                        - CITATION_ERROR → abstain ("can't ground this answer")
                        - LOW_CONFIDENCE + escalation enabled → escalate to Pro model
                        - CONTRADICTION → flag for manual review

repair_strategies:
  RETRIEVE_MORE:       Increase top_k to 30 (from 20)
  STRICT_RERANK:       Enable strict reranking, reduce rerank_top_n to 6, top_k=30
  QUERY_REWRITE:       LLM-based rewrite or keyword fallback
  DECOMPOSE:           Split into ≤3 subquestions, retrieve/fuse (via QueryDecomposer)
  MULTI_QUERY:         Generate 4 query variants, retrieve/fuse (via MultiQueryGenerator)
  ESCALATE_MODEL:      Switch from Flash to Pro tier
  REGENERATE_ONLY:     Re-generate without changing retrieval (ablation control)

repair_routing:         Deterministic mapping from failure_code to strategy:
  missing_evidence  → retrieve_more
  contradiction     → strict_rerank
  incomplete        → decompose
  low_confidence    → multi_query
  citation_error    → strict_rerank
  off_topic         → query_rewrite
```

**Source**: [`agentic_loop.py:73-88`](../src/agentic_loop.py#L73-L88) (`AgenticConfig`), [`agentic_loop.py:415-448`](../src/agentic_loop.py#L415-L448) (`_choose_repair_strategy`), [`self_reflection.py:596-604`](../src/self_reflection.py#L596-L604) (acceptance logic), [`self_reflection.py:706-716`](../src/self_reflection.py#L706-L716) (repair suggestion mapping)

---

## 5. LLM GENERATION SETTINGS

```text
model_flash:     gemini-2.5-flash  (base tier — used for all Attempt-1 queries)
model_pro:       gemini-2.5-pro    (escalated tier — used only on model escalation)
temperature:     Not explicitly set in generation adapter (Gemini API default)
                 Verification judge uses temperature=0.0
max_tokens:      Not explicitly set (Gemini API default)
top_p:           Not explicitly set (Gemini API default)
response_format: Plain text (generation); JSON (verification judge)

citation_requirement:  Yes (citation markers [1], [2] expected in answers)
citation_format:       Inline brackets referencing document indices

cost_tracking:
  flash_input:   $0.30 / 1M tokens ($0.00030 / 1k tokens)
  flash_output:  $2.50 / 1M tokens ($0.00250 / 1k tokens)
  pro_input:     $1.25 / 1M tokens ($0.00125 / 1k tokens)
  pro_output:    $10.00 / 1M tokens ($0.01000 / 1k tokens)

retry_policy:    Up to 8 retries with exponential backoff for 503/overload errors
                 Wait formula: 2^(attempt+2) + (attempt * 2) seconds
```

**Source**: [`enhanced_agentic_rag.py:411-417`](../src/enhanced_agentic_rag.py#L411-L417) (model selection), [`enhanced_agentic_rag.py:191-196`](../src/enhanced_agentic_rag.py#L191-L196) (cost constants), [`self_reflection.py:186-189`](../src/self_reflection.py#L186-L189) (judge config)

---

## 6. PROMPTS

### 6.1 Generation Prompt (Response Synthesis)

Used in the agentic loop's `generate_adapter` ([`enhanced_agentic_rag.py:421-430`](../src/enhanced_agentic_rag.py#L421-L430)):

```text
Question: {question}

Context:
[1] {document_1_text}
[2] {document_2_text}
...
[N] {document_N_text}
```

> **Note**: The generation prompt is minimal by design. The system relies on
> Gemini's instruction-following to synthesize answers from the numbered context
> documents. No system instruction or explicit "cite your sources" directive is
> injected at the generation stage — citation behavior is evaluated post-hoc by
> the verification judge.

### 6.2 Verification Prompt (LLM-as-Judge)

Used in `FactualConsistencyChecker._check_with_llm` ([`self_reflection.py:145-168`](../src/self_reflection.py#L145-L168)):

```text
Evaluate the quality and consistency of the RAG answer.

Question: {question}

Documents:
Document 1: {doc_text_1}
Document 2: {doc_text_2}
...

Answer:
{answer}

Evaluation criteria:
1. Factual Consistency: Are all claims in the answer supported by the documents?
2. Contradictions: Does the answer contradict the documents?
3. Answerability: Does the answer directly address the question?
   - If the answer says "I don't know", "Unable to determine", or "Context
     missing", this MUST be marked as "MISSING_INFO".
   - If the answer refuses to answer because of missing information, this is
     a FAILURE ("MISSING_INFO").
   - An answer is only "CONSISTENT" if it actually answers the question using
     the documents.

Respond with a JSON object containing:
- "status": ONE of "CONSISTENT" (Good - fully answered),
  "PARTIALLY_CONSISTENT" (Okay - partial answer),
  "INCONSISTENT" (Bad/Wrong),
  "MISSING_INFO" (Refusal/Unstated)
- "score": Float 0.0 to 1.0 (1.0 = perfect, 0.1 = I don't know)
- "issues": List of strings describing specific errors.

JSON Response:
```

**Judge Configuration**:
- Model: `gemini-2.5-flash`
- Temperature: `0.0`
- Response format: `application/json`

### 6.3 Self-Correction Prompt (Auto-Repair)

Used in `SelfReflectionSystem._attempt_correction` ([`self_reflection.py:739-752`](../src/self_reflection.py#L739-L752)):

```text
The following answer has quality issues. Please revise it to address the problems.

Question: {question}

Current answer:
{answer}

Issues found:
- {issue_1}
- {issue_2}

Suggestions:
- {suggestion_1}
- {suggestion_2}

Provide a corrected version that addresses these issues:
```

### 6.4 Refusal Template

When the system cannot answer after 2 attempts ([`agentic_loop.py:467-479`](../src/agentic_loop.py#L467-L479)):

```text
# Abstain (CITATION_ERROR):
"I can't confidently ground this answer in the retrieved evidence."

# Ask Clarification (MISSING_EVIDENCE / OFF_TOPIC):
"Can you clarify what exactly you want (scope, entity, time period),
 or provide an additional document/source?"
```

---

## 7. BENCHMARK CONFIGURATION (Jan 2026 Run)

```text
dataset:            250 Natural Questions + 250 HotpotQA (stratified)
total_queries:      500
evaluation_metrics: RAGAS (Faithfulness, Answer Relevancy, Context Recall,
                    Context Precision) + BERTScore + Semantic Sim + IR metrics

systems_compared:
  1. "Enhanced RAG (Full)"       — All features ON, model routing ON
                                   (Agentic Flash in paper)
  2. "Enhanced RAG (No Routing)" — Forced gemini-2.5-pro
                                   (Agentic Pro in paper)
  3. "RAG (No Validation)"      — No self-reflection, no pydantic
  4. "Vanilla RAG"              — Hybrid search only, no reranking,
                                   no reflection, no query rewriting
                                   (Vanilla Flash in paper)

memory_config:
  vector_store:     Qdrant (local, on-disk)
  embedding_model:  models/text-embedding-004
  llm:              gemini-flash-latest (for memory operations)
  chunk_size:       1500 tokens
```

**Source**: [`run_evaluation_benchmarks.py:501-594`](../src/run_evaluation_benchmarks.py#L501-L594) (`setup_systems_for_benchmarks`)

---

## 8. QUICK REFERENCE TABLE

| Parameter | Value |
|-----------|-------|
| **Chunking** | 1500 tokens (server-managed by Gemini File Search) |
| **Embedding** | `text-embedding-004` (Google) |
| **Retrieval** | Hybrid: BM25 (w=0.6) + Dense (w=0.4), RRF k=60 |
| **Top-K** | 20 candidates retrieved |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **Rerank Keep** | 8 (normal) / 6 (strict repair) / 5 (two-stage) |
| **Max Attempts** | 2 (Attempt-1 → Repair → Attempt-2) |
| **Verification** | LLM Judge (Flash, T=0.0) + heuristics |
| **Pass Threshold** | score ≥ 0.7, no FAILED checks |
| **Generation Model** | `gemini-2.5-flash` (base) / `gemini-2.5-pro` (escalated) |
| **Temperature** | 0.0 (judge) / default (generation) |
| **Repair Strategies** | 7 (retrieve_more, strict_rerank, query_rewrite, decompose, multi_query, escalate_model, regenerate_only) |
| **Cost (Flash)** | $0.30/1M input, $2.50/1M output |
| **Cost (Pro)** | $1.25/1M input, $10.00/1M output |

---

## 9. PAPER-READY FORMATTED VERSION

For direct inclusion in your Methods section or Appendix:

> ### System Implementation Details
>
> The system was implemented in Python using the Google Gemini 2.5 API.
> Documents were ingested via Gemini File Search with 1,500-token chunks.
> Retrieval used a hybrid strategy combining BM25 (k₁=1.5, b=0.75) and
> dense vector search (text-embedding-004), fused via Reciprocal Rank
> Fusion (RRF, k=60) with weights 0.6/0.4. The top-20 candidates were
> reranked using a cross-encoder (`ms-marco-MiniLM-L-6-v2`), retaining
> the top 8 passages.
>
> The agentic governance loop permitted a maximum of two attempts per
> query. After each attempt, the system performed three verification
> checks: (1) factual consistency via an LLM judge (Gemini 2.5 Flash,
> temperature=0.0, JSON output), (2) citation validity via rule-based
> pattern matching, and (3) completeness via keyword overlap analysis.
> An answer was accepted when the overall score exceeded 0.7 and no
> individual check returned a FAILED status.
>
> When verification failed, the system selected a repair strategy
> deterministically based on failure codes: missing evidence triggered
> expanded retrieval (top-k increased to 30), citation errors triggered
> strict reranking (top-n reduced to 6), incomplete answers triggered
> query decomposition (≤3 subquestions), and low confidence triggered
> multi-query expansion (4 variants). If both attempts failed, the
> system either requested clarification, abstained, or escalated to
> the Pro-tier model.
>
> Generation used Gemini 2.5 Flash ($0.30/1M input tokens) as the
> default model, with Gemini 2.5 Pro ($1.25/1M input) available for
> model escalation. The evaluation benchmark comprised 500 queries
> (250 Natural Questions + 250 HotpotQA) evaluated across four system
> configurations.
-----------------------------------------------------------------------
One note: the generation adapter uses a minimal prompt (just Question: ... Context: [1]...[2]...) with no explicit system instruction — the model's default behavior handles synthesis. The verification judge is the one with the detailed rubric (temperature=0.0, JSON output).