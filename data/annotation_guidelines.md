# Human Validation — Annotation Guidelines

> **Purpose:** This document specifies the protocol used for the human validation of the automated faithfulness evaluation (RAGAS + GPT-4o judge).

---

## 1. Objective

To assess the reliability of the automated GPT-4o faithfulness evaluator by computing inter-rater agreement (Cohen's κ) between the automated binary judgment and an independent human annotator on a stratified random sample.

## 2. Sample Selection

- **Sample size:** 50 queries (stratified random sample from the 500-query benchmark)
- **Stratification:** Proportional to query type (NQ-Open factual, HotpotQA multi-hop) and outcome (Pass/Fail as judged by the automated evaluator)
- **Random seed:** 42

## 3. Materials Provided to Annotators

For each query, the annotator received:

1. **Query text** — the original question
2. **Retrieved context blocks** — the exact passages fed to the generative model (numbered [1]...[N])
3. **Generated answer** — the system's output
4. **Ground truth answer** — from the benchmark dataset (for reference only; not the primary evaluation criterion)

## 4. Evaluation Criterion: Faithfulness

**Definition:** An answer is **faithful** if and only if every factual claim in the answer is **directly supported** by the retrieved context blocks. The answer need not be complete; it must simply not introduce claims unsupported by the provided context.

### Decision Rules

| Judgment | Criterion |
|:---|:---|
| **PASS (Faithful)** | All factual claims in the answer are supported by the retrieved context. Minor phrasing differences or paraphrasing are acceptable. |
| **FAIL (Unfaithful)** | The answer contains at least one factual claim that is **not supported** by the retrieved context, OR the answer **contradicts** the retrieved context, OR the answer introduces **external knowledge** not present in the context. |

### Borderline Cases

- **Refusals:** If the system refuses to answer (e.g., "I cannot determine this from the provided context"), judge this as **PASS** — the system correctly identified insufficient context.
- **Partial answers:** If the answer is incomplete but every stated claim is supported, judge as **PASS**.
- **Hedged language:** If the answer uses hedging (e.g., "Based on the context, it appears that...") and the hedged claim is supported, judge as **PASS**.

## 5. Procedure

1. The annotator reviewed each item independently without access to the automated judgment.
2. Items were presented in randomized order.
3. Each item was assigned exactly one binary label: PASS or FAIL.
4. No time limit was imposed per item.

## 6. Agreement Metric

- **Cohen's κ** was computed between the human annotator and the automated GPT-4o judge.
- **Reported result:** κ ≈ 0.912 (near-perfect agreement; Landis & Koch, 1977)

## 7. Disagreement Analysis

Disagreement cases (where human ≠ automated) were qualitatively reviewed to identify systematic bias patterns. The most common disagreement category was edge-case hedging in multi-hop answers, where the automated judge occasionally accepted slightly under-supported inferences.
