"""agentic_loop.py

Bounded Agentic RAG controller (Attempt-1 -> self-reflection -> retrieval-first repair -> Attempt-2 -> stop).

This module is designed to be **implementation-aligned** but **framework-agnostic**:
- You inject your own `retrieve_fn` and `generate_fn`.
- It uses `SelfReflectionSystem.reflect_structured(...)` (from your upgraded self_reflection module)
  to choose repair actions deterministically.

How to integrate (minimal):
- `retrieve_fn(query: str, params: RetrievalParams) -> dict` returning:
    { "documents": [ { "content": "...", "metadata": {...} }, ... ],
      "citations": [ {...}, ... ] (optional),
      "retrieval_meta": {...} (optional) }
- `generate_fn(question: str, documents: list, citations: list|None, model_tier: str, **kwargs) -> str`

"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import time

# Import from your project (replace the import path if needed)
try:
    from self_reflection import SelfReflectionSystem, FailureCode, AgenticVerdict
except ImportError:
    # Fallback or error handling
    print("Warning: Could not import self_reflection. Ensure it is in the python path.")
    SelfReflectionSystem = None
    FailureCode = None
    AgenticVerdict = None


class RepairStrategy(Enum):
    RETRIEVE_MORE = "retrieve_more"
    STRICT_RERANK = "strict_rerank"
    QUERY_REWRITE = "query_rewrite"
    DECOMPOSE = "decompose"
    MULTI_QUERY = "multi_query"
    REGENERATE_ONLY = "regenerate_only"
    ESCALATE_MODEL = "escalate_model"


class EscalationAction(Enum):
    ESCALATE_MODEL = "escalate_model"
    ASK_CLARIFICATION = "ask_clarification"
    ABSTAIN = "abstain"
    MANUAL_REVIEW = "manual_review"
    NONE = "none"


class FinalState(Enum):
    PASS_ATTEMPT1 = "pass_attempt1"
    PASS_ATTEMPT2 = "pass_attempt2"
    FAIL_ESCALATED = "fail_escalated"
    FAIL_ABSTAINED = "fail_abstained"


@dataclass
class RetrievalParams:
    top_k: int = 20  # Increased from 10 for better recall
    use_hybrid: bool = True
    strict_rerank: bool = False
    rerank_top_n: int = 8
    # Optional knobs for hybrid retrieval / filters
    hybrid_alpha: float = 0.5
    filters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgenticConfig:
    max_attempts: int = 2
    initial_confidence: float = 0.8
    low_confidence_threshold: float = 0.55
    # Attempt-2 retrieval knobs
    retrieve_more_k: int = 30
    multi_query_count: int = 4
    decompose_max_subquestions: int = 3
    # Model routing
    base_model_tier: str = "base"
    escalated_model_tier: str = "pro"
    allow_model_escalation: bool = True
    forced_model_tier: Optional[str] = None  # When set, ALWAYS use this tier (no escalation)
    # Baseline control (for experiments)
    attempt2_override: Optional[RepairStrategy] = None  # e.g., REGENERATE_ONLY


@dataclass
class StepLog:
    attempt: int
    query: str
    repair_strategy: Optional[str]
    model_tier: str
    latency_ms: float
    verdict: Dict[str, Any]
    retrieval_params: Dict[str, Any] = field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgenticResponse:
    answer: str
    documents: List[Dict[str, Any]]
    citations: List[Dict[str, Any]]
    attempts: int
    final_state: str
    failure_code: Optional[str]
    repair_strategy: Optional[str]
    escalation_action: str
    followup_question: Optional[str]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost: float = 0.0
    step_logs: List[StepLog] = field(default_factory=list)


class AgenticRAGController:
    """Bounded agentic controller with retrieval-first repair."""

    def __init__(
        self,
        reflection: SelfReflectionSystem,
        retrieve_fn: Callable[[str, RetrievalParams], Dict[str, Any]],
        generate_fn: Callable[..., str],
        rewrite_query_fn: Optional[Callable[[str], str]] = None,
        decompose_fn: Optional[Callable[[str, int], List[str]]] = None,
        multi_query_fn: Optional[Callable[[str, int], List[str]]] = None,
        logger_fn: Optional[Callable[[StepLog], None]] = None,
        config: Optional[AgenticConfig] = None,
    ):
        self.reflection = reflection
        self.retrieve_fn = retrieve_fn
        self.generate_fn = generate_fn
        self.rewrite_query_fn = rewrite_query_fn
        self.decompose_fn = decompose_fn
        self.multi_query_fn = multi_query_fn
        self.logger_fn = logger_fn
        self.config = config or AgenticConfig()

    def answer(self, question: str, retrieval_params: Optional[RetrievalParams] = None) -> AgenticResponse:
        params = retrieval_params or RetrievalParams()
        logs: List[StepLog] = []

        # Determine model tier (forced overrides base)
        model_tier = self.config.forced_model_tier or self.config.base_model_tier

        # -------------------------
        # Attempt 1
        # -------------------------
        a1 = self._run_attempt(
            attempt=1,
            question=question,
            query=question,
            params=params,
            model_tier=model_tier,
            repair_strategy=None,
        )
        logs.append(a1["log"])
        if self.logger_fn:
            self.logger_fn(a1["log"])

        if a1["verdict"].passed:
            return AgenticResponse(
                answer=a1["answer"],
                documents=a1["documents"],
                citations=a1["citations"],
                attempts=1,
                final_state=FinalState.PASS_ATTEMPT1.value,
                failure_code=None,
                repair_strategy=None,
                escalation_action=EscalationAction.NONE.value,
                followup_question=None,
                prompt_tokens=a1["log"].prompt_tokens,
                completion_tokens=a1["log"].completion_tokens,
                total_cost=a1["log"].cost,
                step_logs=logs,
            )

        # -------------------------
        # Attempt 2 (repair)
        # -------------------------
        strategy = self._choose_repair_strategy(a1["verdict"])
        if self.config.attempt2_override is not None:
            strategy = self.config.attempt2_override

        a2 = self._run_repaired_attempt(
            attempt=2,
            question=question,
            prev_verdict=a1["verdict"],
            strategy=strategy,
            base_params=params,
        )
        logs.append(a2["log"])
        if self.logger_fn:
            self.logger_fn(a2["log"])

        # -------------------------
        # Choose best output deterministically
        # -------------------------
        best = self._choose_best(a1, a2)

        # If best passed, return
        if best["verdict"].passed:
            return AgenticResponse(
                answer=best["answer"],
                documents=best["documents"],
                citations=best["citations"],
                attempts=2,
                final_state=FinalState.PASS_ATTEMPT2.value,
                failure_code=None,
                repair_strategy=strategy.value,
                escalation_action=EscalationAction.NONE.value,
                followup_question=None,
                prompt_tokens=sum(l.prompt_tokens for l in logs),
                completion_tokens=sum(l.completion_tokens for l in logs),
                total_cost=sum(l.cost for l in logs),
                step_logs=logs,
            )

        # -------------------------
        # If still failing after Attempt-2: decide escalation action
        # -------------------------
        final_failure = best["verdict"].failure_code
        escalation_action, followup = self._choose_escalation_action(final_failure)

        final_state = FinalState.FAIL_ESCALATED.value if escalation_action != EscalationAction.ABSTAIN else FinalState.FAIL_ABSTAINED.value

        return AgenticResponse(
            answer=best["answer"],
            documents=best["documents"],
            citations=best["citations"],
            attempts=2,
            final_state=final_state,
            failure_code=final_failure if final_failure else None,
            repair_strategy=strategy.value,
            escalation_action=escalation_action.value,
            followup_question=followup,
            prompt_tokens=sum(l.prompt_tokens for l in logs),
            completion_tokens=sum(l.completion_tokens for l in logs),
            total_cost=sum(l.cost for l in logs),
            step_logs=logs,
        )

    # -------------------------
    # Internals
    # -------------------------

    def _run_attempt(
        self,
        attempt: int,
        question: str,
        query: str,
        params: RetrievalParams,
        model_tier: str,
        repair_strategy: Optional[str],
    ) -> Dict[str, Any]:
        t0 = time.time()
        retrieved = self.retrieve_fn(query, params) or {}
        documents = retrieved.get("documents", []) or []
        citations = retrieved.get("citations", []) or []

        raw_output = self.generate_fn(
            question=question,
            documents=documents,
            citations=citations,
            model_tier=model_tier,
        )
        t1 = time.time()

        # Handle tuple return (answer, metadata) or just string
        usage_meta = {}
        if isinstance(raw_output, tuple):
            answer = raw_output[0]
            usage_meta = raw_output[1] if len(raw_output) > 1 else {}
        else:
            answer = raw_output
            
        if isinstance(answer, tuple):
             answer = str(answer[0]) if len(answer) > 0 else ""

        # Extract usage
        p_tokens = usage_meta.get('prompt_tokens', 0)
        c_tokens = usage_meta.get('completion_tokens', 0)
        cost = usage_meta.get('cost', 0.0)

        # Reflect
        verdict = self.reflection.reflect_structured(
            question=question,
            answer=answer,
            documents=documents,
            citations=citations,
            initial_confidence=self.config.initial_confidence,
        )

        latency_ms = (time.time() - t0) * 1000.0

        log = StepLog(
            attempt=attempt,
            query=query,
            repair_strategy=repair_strategy,
            model_tier=model_tier,
            latency_ms=latency_ms,
            verdict=verdict.to_dict(),
            retrieval_params={
                "top_k": params.top_k,
                "use_hybrid": params.use_hybrid,
                "strict_rerank": params.strict_rerank,
                "rerank_top_n": params.rerank_top_n,
                "hybrid_alpha": params.hybrid_alpha,
                "filters": dict(params.filters),
            },
            prompt_tokens=p_tokens,
            completion_tokens=c_tokens,
            cost=cost,
            meta=retrieved.get("retrieval_meta", {}) or {},
        )

        return {
            "answer": answer,
            "documents": documents,
            "citations": citations,
            "verdict": verdict,
            "log": log,
        }

    def _run_repaired_attempt(
        self,
        attempt: int,
        question: str,
        prev_verdict: AgenticVerdict,
        strategy: RepairStrategy,
        base_params: RetrievalParams,
    ) -> Dict[str, Any]:
        model_tier = self.config.forced_model_tier or self.config.base_model_tier
        query = question
        params = RetrievalParams(**base_params.__dict__)  # shallow copy

        # Apply deterministic repair deltas
        if strategy == RepairStrategy.REGENERATE_ONLY:
            # no retrieval changes
            pass

        elif strategy == RepairStrategy.RETRIEVE_MORE:
            params.top_k = max(params.top_k, self.config.retrieve_more_k)

        elif strategy == RepairStrategy.STRICT_RERANK:
            params.strict_rerank = True
            params.rerank_top_n = min(params.rerank_top_n, 6)
            params.top_k = max(params.top_k, 30)

        elif strategy == RepairStrategy.QUERY_REWRITE:
            query = self._rewrite_query(question)

        elif strategy == RepairStrategy.DECOMPOSE:
            # Decompose into subquestions; retrieve/fuse
            subqs = self._decompose(question)
            docs, cits = self._retrieve_fused(subqs, params)
            # Generate directly from fused evidence
            t0 = time.time()
            answer = self.generate_fn(question=question, documents=docs, citations=cits, model_tier=model_tier)
            verdict = self.reflection.reflect_structured(question, answer, docs, cits, initial_confidence=self.config.initial_confidence)
            latency_ms = (time.time() - t0) * 1000.0
            log = StepLog(
                attempt=attempt,
                query=" | ".join(subqs),
                repair_strategy=strategy.value,
                model_tier=model_tier,
                latency_ms=latency_ms,
                verdict=verdict.to_dict(),
                retrieval_params={"fused": True, **params.__dict__},
            )
            return {"answer": answer, "documents": docs, "citations": cits, "verdict": verdict, "log": log}

        elif strategy == RepairStrategy.MULTI_QUERY:
            variants = self._multi_query(question)
            docs, cits = self._retrieve_fused(variants, params)
            t0 = time.time()
            answer = self.generate_fn(question=question, documents=docs, citations=cits, model_tier=model_tier)
            verdict = self.reflection.reflect_structured(question, answer, docs, cits, initial_confidence=self.config.initial_confidence)
            latency_ms = (time.time() - t0) * 1000.0
            log = StepLog(
                attempt=attempt,
                query=" | ".join(variants),
                repair_strategy=strategy.value,
                model_tier=model_tier,
                latency_ms=latency_ms,
                verdict=verdict.to_dict(),
                retrieval_params={"fused": True, **params.__dict__},
            )
            return {"answer": answer, "documents": docs, "citations": cits, "verdict": verdict, "log": log}

        elif strategy == RepairStrategy.ESCALATE_MODEL and self.config.allow_model_escalation and not self.config.forced_model_tier:
            model_tier = self.config.escalated_model_tier

        # Default path: normal retrieve+generate
        return self._run_attempt(
            attempt=attempt,
            question=question,
            query=query,
            params=params,
            model_tier=model_tier,
            repair_strategy=strategy.value,
        )

    def _choose_repair_strategy(self, verdict: AgenticVerdict) -> RepairStrategy:
        # Prefer the verdict suggestion if present
        repair_str = verdict.suggested_repair
        if isinstance(repair_str, tuple):
             # Handle weird Pydantic/Gemini tuple return: (value, meta) or just values
             repair_str = repair_str[0] if len(repair_str) > 0 else ""
             
        tag = (str(repair_str) or "").strip().lower()
        if tag == "retrieve_more":
            return RepairStrategy.RETRIEVE_MORE
        if tag == "strict_rerank":
            return RepairStrategy.STRICT_RERANK
        if tag == "query_rewrite":
            return RepairStrategy.QUERY_REWRITE
        if tag == "decompose":
            return RepairStrategy.DECOMPOSE
        if tag == "multi_query":
            return RepairStrategy.MULTI_QUERY

        # Otherwise map failure code
        fc = verdict.failure_code
        if fc == FailureCode.CITATION_ERROR.value:
            return RepairStrategy.STRICT_RERANK
        if fc == FailureCode.MISSING_EVIDENCE.value:
            return RepairStrategy.QUERY_REWRITE
        if fc == FailureCode.INCOMPLETE.value:
            return RepairStrategy.DECOMPOSE
        if fc == FailureCode.LOW_CONFIDENCE.value:
            return RepairStrategy.QUERY_REWRITE
        if fc == FailureCode.OFF_TOPIC.value:
            return RepairStrategy.QUERY_REWRITE
        if fc == FailureCode.CONTRADICTION.value:
            return RepairStrategy.MULTI_QUERY
        return RepairStrategy.RETRIEVE_MORE

    def _choose_best(self, a1: Dict[str, Any], a2: Dict[str, Any]) -> Dict[str, Any]:
        v1: AgenticVerdict = a1["verdict"]
        v2: AgenticVerdict = a2["verdict"]

        # PASS beats FAIL
        if v1.passed and not v2.passed:
            return a1
        if v2.passed and not v1.passed:
            return a2

        # Otherwise use overall_score if available
        s1 = float((v1.metadata or {}).get("overall_score", 0.0))
        s2 = float((v2.metadata or {}).get("overall_score", 0.0))
        if s2 > s1:
            return a2
        return a1

    def _choose_escalation_action(self, failure_code: Optional[FailureCode]) -> Tuple[EscalationAction, Optional[str]]:
        if failure_code in (FailureCode.MISSING_EVIDENCE, FailureCode.OFF_TOPIC):
            return EscalationAction.ASK_CLARIFICATION, self._clarifying_question()
        if failure_code == FailureCode.CITATION_ERROR:
            return EscalationAction.ABSTAIN, "I can’t confidently ground this answer in the retrieved evidence."
        if failure_code == FailureCode.LOW_CONFIDENCE and self.config.allow_model_escalation:
            return EscalationAction.ESCALATE_MODEL, None
        if failure_code == FailureCode.CONTRADICTION:
            return EscalationAction.MANUAL_REVIEW, None
        return EscalationAction.ABSTAIN, None

    def _clarifying_question(self) -> str:
        return "Can you clarify what exactly you want (scope, entity, time period), or provide an additional document/source?"

    # -------------------------
    # Repair utilities (pluggable)
    # -------------------------

    def _rewrite_query(self, question: str) -> str:
        if self.rewrite_query_fn:
            try:
                return self.rewrite_query_fn(question)
            except Exception:
                pass
        # Simple rule-based rewrite: keep keywords, add intent terms
        words = [w for w in question.split() if len(w) > 3]
        return " ".join(words + ["evidence", "source", "policy", "definition"])

    def _decompose(self, question: str) -> List[str]:
        if self.decompose_fn:
            try:
                return self.decompose_fn(question, self.config.decompose_max_subquestions)
            except Exception:
                pass
        # Minimal fallback: no decomposition
        return [question]

    def _multi_query(self, question: str) -> List[str]:
        if self.multi_query_fn:
            try:
                qs = self.multi_query_fn(question, self.config.multi_query_count)
                return qs[: self.config.multi_query_count] if qs else [question]
            except Exception:
                pass
        # Simple variants
        base = question.strip()
        return list(dict.fromkeys([
            base,
            base + " background",
            base + " key points",
            base + " evidence",
        ]))[: self.config.multi_query_count]

    def _retrieve_fused(self, queries: Sequence[str], params: RetrievalParams) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        # Naive fusion: concatenate and deduplicate by content hash
        all_docs: List[Dict[str, Any]] = []
        all_cits: List[Dict[str, Any]] = []
        seen = set()
        for q in queries:
            r = self.retrieve_fn(q, params) or {}
            for d in (r.get("documents", []) or []):
                key = (d.get("content", "")[:300]).strip()
                if key and key not in seen:
                    seen.add(key)
                    all_docs.append(d)
            for c in (r.get("citations", []) or []):
                all_cits.append(c)
        return all_docs, all_cits


def run_agentic_rag(
    question: str,
    retrieve_fn: Callable[[str, RetrievalParams], Dict[str, Any]],
    generate_fn: Callable[..., str],
    reflection: Optional[SelfReflectionSystem] = None,
    config: Optional[AgenticConfig] = None,
) -> AgenticResponse:
    """Convenience wrapper."""

    reflection = reflection or SelfReflectionSystem(use_llm=False, auto_correct=False, validator_mode="heuristics")
    controller = AgenticRAGController(
        reflection=reflection,
        retrieve_fn=retrieve_fn,
        generate_fn=generate_fn,
        config=config or AgenticConfig(),
    )
    return controller.answer(question)
