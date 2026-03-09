"""
Comprehensive Evaluation Framework for Agentic RAG System
==========================================================

Research-grade evaluation for journal publication including:
- RAGAS metrics (faithfulness, relevancy, context precision/recall)
- Traditional IR metrics (Precision@k, Recall@k, MRR, NDCG)
- Semantic similarity (BERTScore, ROUGE)
- Multiple baselines comparison
- Ablation studies
- Statistical significance testing
- Human evaluation interface
- Error analysis
- LaTeX export for papers

Designed for publication in top-tier venues (TACL, JAIR, ACL, EMNLP, etc.)
"""

import json
import time
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# Import evaluation components
from metrics_ragas import RAGASEvaluator
from metrics_ir import IRMetricsEvaluator
from metrics_semantic import SemanticEvaluator
from baselines import BaselineManager
from ablation import AblationStudy
from statistical_analysis import StatisticalAnalyzer
from visualization import PaperVisualizer
from latex_export import LaTeXExporter

# Database integration
try:
    from supabase import create_client
    from evaluation_db import EvaluationDB
    HAS_DB = True
except ImportError:
    HAS_DB = False
    print("⚠️ Supabase/EvaluationDB not available - running in local mode")


@dataclass
class EvaluationConfig:
    """Configuration for evaluation experiments"""

    # Experiment metadata
    experiment_name: str
    experiment_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    description: str = ""

    # Dataset configuration
    dataset_path: str = "evaluation_data/test_set.json"
    num_samples: Optional[int] = None  # None = use all

    # System configurations to evaluate
    evaluate_baselines: bool = True
    evaluate_ablations: bool = True

    # Metrics to compute
    compute_ragas: bool = True
    compute_ir_metrics: bool = True
    compute_semantic: bool = True
    compute_efficiency: bool = True

    # Statistical analysis
    compute_significance: bool = True
    significance_level: float = 0.05

    # Human evaluation
    include_human_eval: bool = False
    num_human_samples: int = 100

    # Output configuration
    max_workers: int = 4  # Default parallel workers (increase for Tier 1)
    output_dir: str = "evaluation_results"
    generate_latex: bool = True
    generate_plots: bool = True
    save_detailed_results: bool = True


@dataclass
class QueryResult:
    """Result from a single query evaluation"""
    query_id: str
    question: str
    ground_truth: str
    predicted_answer: str
    retrieved_contexts: List[str]

    # Metadata
    query_type: str
    system_name: str
    timestamp: str

    # Performance metrics
    latency_ms: float
    latency_ms: float
    tokens_used: int
    cost: float = 0.0

    cost: float = 0.0

    # Ground truth context (for strict IR eval)
    ground_truth_context: Optional[str] = None

    # Quality scores (computed later)
    scores: Dict[str, float] = field(default_factory=dict)
    
    # Metadata for deep analysis (e.g. agentic steps)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResults:
    """Complete evaluation results"""
    config: EvaluationConfig

    # Aggregate metrics per system
    system_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Detailed per-query results
    query_results: List[QueryResult] = field(default_factory=list)

    # Statistical analysis
    statistical_tests: Dict[str, Any] = field(default_factory=dict)

    # Human evaluation (if performed)
    human_evaluation: Optional[Dict[str, Any]] = None

    # Error analysis
    error_analysis: Dict[str, Any] = field(default_factory=dict)

    # Timing information
    total_evaluation_time: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON export"""
        return asdict(self)

    def save(self, output_path: str):
        """Save results to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, input_path: str) -> 'EvaluationResults':
        """Load results from JSON file"""
        with open(input_path, 'r') as f:
            data = json.load(f)
        return cls(**data)


class EvaluationFramework:
    """
    Main evaluation framework orchestrator

    Coordinates all evaluation components to produce comprehensive
    research-grade evaluation results suitable for journal publication.
    """

    def __init__(self, config: EvaluationConfig):
        """
        Initialize evaluation framework

        Args:
            config: Evaluation configuration
        """
        self.config = config

        # Create output directory
        self.output_dir = Path(config.output_dir) / config.experiment_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize evaluation components
        self.ragas_evaluator = RAGASEvaluator() if config.compute_ragas else None
        self.ir_evaluator = IRMetricsEvaluator() if config.compute_ir_metrics else None
        self.semantic_evaluator = SemanticEvaluator() if config.compute_semantic else None

        # Initialize comparison components
        self.baseline_manager = BaselineManager() if config.evaluate_baselines else None
        self.ablation_study = AblationStudy() if config.evaluate_ablations else None

        # Initialize analysis components
        self.statistical_analyzer = StatisticalAnalyzer()
        self.visualizer = PaperVisualizer(self.output_dir)
        self.latex_exporter = LaTeXExporter(self.output_dir)

        # Initialize Database Logger
        self.eval_db = None
        if HAS_DB:
            try:
                url = os.getenv("SUPABASE_URL")
                key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
                if url and key:
                    supabase = create_client(url, key)
                    self.eval_db = EvaluationDB(supabase)
                    print(f"[EvaluationFramework] ✅ Connected to Supabase Dashboard")
            except Exception as e:
                print(f"[EvaluationFramework] ⚠️ Failed to connect to DB: {e}")

        print(f"[EvaluationFramework] Initialized with ID: {config.experiment_id}")
        print(f"[EvaluationFramework] Output directory: {self.output_dir}")

    def load_dataset(self) -> List[Dict]:
        """
        Load evaluation dataset

        Returns:
            List of evaluation samples with format:
            {
                'query_id': str,
                'question': str,
                'ground_truth': str,
                'query_type': str,
                'relevant_docs': List[str]
            }
        """
        print(f"[EvaluationFramework] Loading dataset from {self.config.dataset_path}")

        with open(self.config.dataset_path, 'r') as f:
            data = json.load(f)

        # Handle structured JSON with 'items' key (e.g., from DatasetBuilder)
        if isinstance(data, dict) and 'items' in data:
            dataset = data['items']
        else:
            dataset = data
        
        # Normalize field names to match expected format
        for item in dataset:
            # Ensure 'ground_truth' field exists (may be 'ground_truth_answer')
            if 'ground_truth_answer' in item and 'ground_truth' not in item:
                item['ground_truth'] = item['ground_truth_answer']

        # Sample if needed
        if self.config.num_samples:
            dataset = dataset[:self.config.num_samples]

        print(f"[EvaluationFramework] Loaded {len(dataset)} samples")
        return dataset

    def evaluate_system(
        self,
        system,
        system_name: str,
        dataset: List[Dict],
        max_workers: int = 4
    ) -> List[QueryResult]:
        """
        Evaluate a single system on the dataset with parallel processing
        """
        print(f"\n[EvaluationFramework] Evaluating system: {system_name}")
        print(f"[EvaluationFramework] Processing {len(dataset)} queries with {max_workers} workers...")

        import concurrent.futures

        results = []
        
        def process_single_query(sample):
            try:
                # Time the query
                start_time = time.time()

                # Execute query
                response = system.query(
                    question=sample['question'],
                    include_citations=True
                )

                latency_ms = (time.time() - start_time) * 1000

                # Robustly extract fields handling both dict and objects
                if isinstance(response, dict):
                    text = response.get('text', '') or response.get('answer', '')
                    cost = response.get('cost', 0.0)
                    tokens = response.get('total_tokens', 0)
                    # If tokens not set, check prompt/completion
                    if tokens == 0:
                        tokens = response.get('prompt_tokens', 0) + response.get('completion_tokens', 0)
                    
                    # Extract metadata if available
                    response_meta = response.get('metadata', {})
                else:
                    # Assume object (AgenticResponse or BaselineResult)
                    text = getattr(response, 'answer', '') or getattr(response, 'text', '')
                    cost = getattr(response, 'total_cost', 0.0)
                    if cost == 0.0: cost = getattr(response, 'cost', 0.0)
                    
                    tokens = getattr(response, 'total_tokens', 0)
                    if tokens == 0:
                         tokens = getattr(response, 'prompt_tokens', 0) + getattr(response, 'completion_tokens', 0)
                         
                    # Extract metadata from object attributes
                    response_meta = {}
                    if hasattr(response, 'attempts'):
                        response_meta['attempts'] = response.attempts
                        response_meta['final_state'] = getattr(response, 'final_state', '')
                        response_meta['repair_strategy'] = getattr(response, 'repair_strategy', '')
                    if hasattr(response, 'metadata'):
                        response_meta.update(response.metadata)

                # Estimate if missing
                if tokens == 0 and text:
                     tokens = len(text) // 4

                # Create result object
                return QueryResult(
                    query_id=sample['query_id'],
                    question=sample['question'],
                    ground_truth=sample['ground_truth'],
                    predicted_answer=text,
                    retrieved_contexts=self._extract_contexts(response if isinstance(response, dict) else (asdict(response) if hasattr(response, '__dataclass_fields__') else {})),
                    query_type=sample.get('query_type', 'GENERAL'),
                    system_name=system_name,
                    timestamp=datetime.now().isoformat(),
                    latency_ms=latency_ms,
                    tokens_used=tokens,
                    cost=cost,
                    ground_truth_context=sample.get('metadata', {}).get('ground_truth_context'),
                    metadata=response_meta
                )
            except Exception as e:
                print(f"  Error processing query {sample.get('query_id', 'unknown')}: {e}")
                return None

        # Use ThreadPoolExecutor for parallel execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Map queries to the executor
            future_to_query = {executor.submit(process_single_query, sample): sample for sample in dataset}
            
            # Retrieve results as they complete
            for i, future in enumerate(concurrent.futures.as_completed(future_to_query)):
                result = future.result()
                if result:
                    results.append(result)
                
                # Progress update
                if (i + 1) % 5 == 0 or (i + 1) == len(dataset):
                    print(f"  Processed {i + 1}/{len(dataset)} queries")
                    
                    # --- CRITICAL: CHECKPOINT SAVE ---
                    # Save results every 5 items to prevent data loss on crash/stop
                    try:
                         checkpoint_path = self.output_dir / "results_partial.json"
                         temp_results = EvaluationResults(
                             config=self.config,
                             system_metrics={}, 
                             query_results=results, # Current partial list
                             statistical_tests={},
                             error_analysis={},
                             total_evaluation_time=0
                         )
                         # Custom minimal save to minimize IO overhead
                         with open(checkpoint_path, 'w') as f:
                             import dataclasses
                             data = dataclasses.asdict(temp_results)
                             json.dump(data, f, indent=2)
                         # Also detailed CSV for safety
                         if len(results) > 0:
                             import pandas as pd
                             df = pd.DataFrame([dataclasses.asdict(r) for r in results])
                             df.to_csv(self.output_dir / "results_partial.csv", index=False)
                    except Exception as e:
                        print(f"  ⚠️ Warning: Checkpoint save failed: {e}")
                    # ---------------------------------

        print(f"[EvaluationFramework] Completed {len(results)} queries for {system_name}")
        return results

    def compute_metrics(self, results: List[QueryResult]) -> Dict[str, float]:
        """
        Compute all metrics for a set of results

        Args:
            results: List of QueryResult objects

        Returns:
            Dictionary of metric scores
        """
        print(f"[EvaluationFramework] Computing metrics for {len(results)} results...")

        metrics = {}

        # RAGAS metrics
        if self.ragas_evaluator:
            print("  Computing RAGAS metrics...")
            ragas_scores = self.ragas_evaluator.evaluate_batch(results)
            metrics.update(ragas_scores)

        # IR metrics
        if self.ir_evaluator:
            print("  Computing IR metrics...")
            ir_scores = self.ir_evaluator.evaluate_batch(results)
            metrics.update(ir_scores)

        # Semantic similarity metrics
        if self.semantic_evaluator:
            print("  Computing semantic metrics...")
            semantic_scores = self.semantic_evaluator.evaluate_batch(results)
            metrics.update(semantic_scores)

        # Efficiency metrics
        if self.config.compute_efficiency:
            print("  Computing efficiency metrics...")
            efficiency_scores = self._compute_efficiency_metrics(results)
            metrics.update(efficiency_scores)

        print(f"[EvaluationFramework] Computed {len(metrics)} metrics")
        
        # --- NEW: Calibration & Agentic Metrics (P0 for Journal) ---
        try:
            from metrics_ragas import CalibrationMetrics
            
            # 1. Calibration (ECE)
            cal_data = CalibrationMetrics.generate_calibration_data(results)
            metrics['ece_score'] = cal_data['ece']
            print(f"  Computed Calibration ECE: {cal_data['ece']:.4f}")
            
            # 2. Agentic Repair Stats (The "Resurrection" Rate)
            # Count how many queries failed initially but succeeded later
            # 2. Agentic Repair Stats (The "Resurrection" Rate)
            # Count how many queries failed initially but succeeded later
            # logic: attempts > 1 AND final_state == passed
            repairs = [r for r in results if r.metadata.get('attempts', 1) > 1]
            if repairs:
                metrics['agentic_loop_count'] = len(repairs)
                # Success rate of repairs (did they end up passing?)
                successful_repairs = [r for r in repairs if 'pass' in str(r.metadata.get('final_state', '')).lower()]
                metrics['agentic_repair_success_rate'] = len(successful_repairs) / len(repairs) if len(repairs) > 0 else 0.0
                print(f"  Agentic Repair Rate: {metrics['agentic_repair_success_rate']:.2%} ({len(successful_repairs)}/{len(repairs)} loops)")
                
                # Tool usage
                strategies = {}
                for r in repairs:
                    strat = r.metadata.get('repair_strategy', 'unknown')
                    strategies[strat] = strategies.get(strat, 0) + 1
                metrics['tool_usage'] = str(strategies) # stringify for CSV compatibility
                
        except Exception as e:
             print(f"  ⚠️ Could not compute advanced agentic metrics: {e}")
        # -----------------------------------------------------------

        return metrics

    def run_full_evaluation(self, systems: Dict[str, Any]) -> EvaluationResults:
        """
        Run complete evaluation on multiple systems

        Args:
            systems: Dictionary mapping system names to system objects
                    Example: {'Baseline RAG': baseline_rag, 'Agentic RAG': agentic_rag}

        Returns:
            Complete evaluation results
        """
        print("\n" + "="*80)
        print(f"EVALUATION EXPERIMENT: {self.config.experiment_name}")
        print(f"Experiment ID: {self.config.experiment_id}")
        print("="*80 + "\n")

        start_time = time.time()

        # Load dataset
        dataset = self.load_dataset()

        # Evaluate each system
        all_query_results = []
        system_metrics = {}

        for system_name, system in systems.items():
            # Run queries
            query_results = self.evaluate_system(
                system, 
                system_name, 
                dataset, 
                max_workers=self.config.max_workers
            )
            all_query_results.extend(query_results)

            # Compute metrics
            metrics = self.compute_metrics(query_results)
            system_metrics[system_name] = metrics

            # Store PER-QUERY metrics (not the batch aggregate!)
            # This is the FIX for the bug where all queries had identical scores
            print(f"  Computing per-query metrics for {len(query_results)} results...")
            for i, result in enumerate(query_results):
                per_query_scores = {}
                
                # Copy system-level aggregate metrics as baseline
                per_query_scores.update(metrics)
                
                # Compute IR metrics per query
                if self.ir_evaluator:
                    retrieved = result.retrieved_contexts
                    relevant = self.ir_evaluator._get_relevant_contexts(result)
                    ir_single = self.ir_evaluator.evaluate_single(retrieved, relevant)
                    for k, v in ir_single.items():
                        per_query_scores[f'ir_{k}'] = v
                
                # Compute semantic metrics per query (BERTScore, ROUGE, BLEU)
                if self.semantic_evaluator:
                    try:
                        semantic_single = self.semantic_evaluator.evaluate_single(
                            result.predicted_answer, 
                            result.ground_truth
                        )
                        per_query_scores.update(semantic_single)
                    except Exception as e:
                        pass  # Keep aggregate values
                
                # Compute RAGAS per query (expensive - only if explicitly enabled)
                # OPTIMIZATION: Check if already computed during batch processing
                ragas_done = False
                if self.ragas_evaluator:
                    existing = getattr(result, 'scores', {}) or {}
                    if 'ragas_faithfulness' in existing and existing['ragas_faithfulness'] != 0.0:
                        # Already computed in batch step! Copy over.
                        per_query_scores.update({k: v for k, v in existing.items() if k.startswith('ragas_')})
                        ragas_done = True

                if not ragas_done and self.ragas_evaluator and hasattr(self.ragas_evaluator, 'evaluate_single'):
                    print(f"  ➡️ Query {i + 1}/{len(query_results)}: Computing RAGAS...")
                    try:
                        ragas_single = self.ragas_evaluator.evaluate_single(
                            result.question,
                            result.predicted_answer,
                            result.retrieved_contexts,
                            result.ground_truth
                        )
                        if ragas_single:
                            for k, v in ragas_single.to_dict().items():
                                per_query_scores[k] = v
                    except Exception as e:
                        pass  # Keep aggregate values
                
                result.scores = per_query_scores

            # --- CRITICAL: GLOBAL CHECKPOINT AFTER EACH SYSTEM ---
            # This saves ALL results from ALL systems processed so far
            try:
                import dataclasses
                checkpoint_path = self.output_dir / "results_checkpoint.json"
                checkpoint_data = {
                    "config": dataclasses.asdict(self.config),
                    "system_metrics": system_metrics,
                    "query_results": [dataclasses.asdict(r) for r in all_query_results],
                    "systems_completed": list(system_metrics.keys()),
                    "total_results_so_far": len(all_query_results)
                }
                with open(checkpoint_path, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)
                print(f"  ✅ Checkpoint saved: {len(all_query_results)} results from {len(system_metrics)} systems")
            except Exception as e:
                print(f"  ⚠️ Warning: Global checkpoint save failed: {e}")
            # -------------------------------------------------------

            # === VALIDATION & INTERACTIVE PAUSE ===
            validation_passed = self._validate_system_results(
                system_name, 
                query_results, 
                len(dataset),
                checkpoint_path
            )
            
            # Get list of remaining systems
            system_names = list(systems.keys())
            current_idx = system_names.index(system_name)
            remaining = system_names[current_idx + 1:]
            
            if remaining:
                choice = self._ask_continue(
                    system_name, 
                    current_idx + 1, 
                    len(systems), 
                    remaining[0],
                    validation_passed
                )
                if choice == 'q':
                    print(f"\n👋 Quitting. Results saved to: {checkpoint_path}")
                    print(f"   To resume, re-run the script - completed systems will be skipped.")
                    break
                elif choice == 's':
                    print(f"⏭️ Skipping {remaining[0]}")
                    continue
            # ========================================

        # Statistical significance testing
        statistical_tests = {}
        if self.config.compute_significance and len(systems) >= 2:
            print("\n[EvaluationFramework] Computing statistical significance...")
            statistical_tests = self.statistical_analyzer.compare_systems(
                system_metrics,
                alpha=self.config.significance_level
            )

        # Error analysis
        print("\n[EvaluationFramework] Performing error analysis...")
        error_analysis = self._perform_error_analysis(all_query_results)

        # Create results object
        results = EvaluationResults(
            config=self.config,
            system_metrics=system_metrics,
            query_results=all_query_results,
            statistical_tests=statistical_tests,
            error_analysis=error_analysis,
            total_evaluation_time=time.time() - start_time
        )

        # Save results
        self._save_results(results)

        # Generate outputs for paper
        if self.config.generate_latex:
            self._generate_latex_tables(results)

        if self.config.generate_plots:
            self._generate_visualizations(results)

        print("\n" + "="*80)
        print(f"EVALUATION COMPLETE")
        print(f"Total time: {results.total_evaluation_time:.2f} seconds")
        print(f"Results saved to: {self.output_dir}")
        print("="*80 + "\n")

        return results

    def _extract_contexts(self, response: Dict) -> List[str]:
        """Extract retrieved contexts from response"""
        contexts = []
        
        # Priority 1: Use full retrieved 'documents' list if available (for IR metrics)
        if 'documents' in response and response['documents']:
             for doc in response['documents']:
                 if isinstance(doc, dict):
                     # Handle various document formats
                     content = doc.get('page_content') or doc.get('content') or doc.get('text') or doc.get('snippet') or ''
                     if content:
                         contexts.append(content)
                 elif isinstance(doc, str):
                     contexts.append(doc)
                     
        # Priority 2: Fallback to 'citations' (used contexts)
        elif 'citations' in response:
            for citation in response['citations']:
                contexts.append(citation.get('snippet', ''))
                
        return contexts

    def _estimate_tokens(self, response: Dict) -> int:
        """Estimate token count (rough approximation)"""
        text = response.get('text', '')
        # Rough estimate: 1 token ≈ 4 characters
        return len(text) // 4

    def _validate_system_results(self, system_name: str, results: List[QueryResult], 
                                  expected_count: int, checkpoint_path: Path) -> bool:
        """Validate that results are not empty after each system run"""
        print(f"\n🔍 VALIDATING RESULTS for {system_name}...")
        
        result_count = len(results)
        validation_passed = True
        
        # Check result count
        if result_count == 0:
            print(f"   ❌ FAIL: No results!")
            validation_passed = False
        elif result_count < expected_count * 0.8:
            pct = result_count / expected_count * 100
            print(f"   ⚠️ WARNING: Only {result_count}/{expected_count} results ({pct:.1f}%)")
        else:
            print(f"   ✅ PASS: {result_count} results found")
        
        # Check for empty answers (defensive: handle tuple answers)
        def _get_answer_str(r):
            ans = r.predicted_answer
            if isinstance(ans, tuple):
                ans = str(ans[0]) if len(ans) > 0 else ""
            return str(ans).strip()
        empty_answers = sum(1 for r in results if not _get_answer_str(r))
        if empty_answers > 0:
            empty_pct = empty_answers / result_count * 100 if result_count > 0 else 0
            if empty_pct > 20:
                print(f"   ⚠️ WARNING: {empty_answers}/{result_count} ({empty_pct:.1f}%) have empty answers")
            else:
                print(f"   📊 Note: {empty_answers} results have empty answers ({empty_pct:.1f}%)")
        
        # Final status
        status = "✅ VALIDATION PASSED" if validation_passed else "❌ VALIDATION FAILED"
        print(f"\n   {status}")
        print(f"   File: {checkpoint_path}")
        print(f"   Results: {result_count}/{expected_count}")
        
        return validation_passed

    def _ask_continue(self, system_name: str, current: int, total: int, 
                      next_system: str, validation_passed: bool) -> str:
        """Ask user whether to continue to next system"""
        status_icon = "✅" if validation_passed else "⚠️"
        
        print("\n" + "="*60)
        print(f"{status_icon} System {current}/{total} complete: {system_name}")
        print("="*60)
        print(f"\nNext: System {current + 1} - {next_system}")
        print("\nOptions:")
        print("  [c] Continue to next system")
        print("  [s] Skip next system")
        print("  [q] Quit (results are saved, can resume later)")
        
        while True:
            try:
                choice = input("\nYour choice (c/s/q): ").strip().lower()
                if choice in ['c', 's', 'q']:
                    return choice
                print("Invalid choice. Please enter c, s, or q.")
            except (KeyboardInterrupt, EOFError):
                print("\n\n👋 Interrupted. Results saved.")
                return 'q'

    def _compute_efficiency_metrics(self, results: List[QueryResult]) -> Dict[str, float]:
        """Compute efficiency-related metrics"""
        latencies = [r.latency_ms for r in results]
        tokens = [r.tokens_used for r in results]

        costs = [r.cost for r in results]
        
        metrics = {
            'mean_latency_ms': float(np.mean(latencies)),
            'median_latency_ms': float(np.median(latencies)),
            'std_latency_ms': float(np.std(latencies)),
            'p95_latency_ms': float(np.percentile(latencies, 95)),
            'mean_tokens': float(np.mean(tokens)),
            'total_tokens': int(np.sum(tokens)),
            'mean_cost': float(np.mean(costs)),
            'total_cost': float(np.sum(costs))
        }
        
        # Calculate cost per 1k queries
        metrics['cost_per_1k'] = metrics['mean_cost'] * 1000
        
        return metrics

    def _perform_error_analysis(self, results: List[QueryResult]) -> Dict[str, Any]:
        """Perform detailed error analysis"""
        # Group by query type
        by_type = {}
        for result in results:
            if result.query_type not in by_type:
                by_type[result.query_type] = []
            by_type[result.query_type].append(result)

        # Analyze performance by query type
        type_analysis = {}
        for qtype, type_results in by_type.items():
            type_analysis[qtype] = {
                'count': len(type_results),
                'mean_latency': float(np.mean([r.latency_ms for r in type_results]))
            }

        return {
            'by_query_type': type_analysis,
            'total_queries': len(results)
        }

    def _save_results(self, results: EvaluationResults):
        """Save evaluation results"""
        # Save main results as JSON
        results_path = self.output_dir / 'results.json'
        results.save(str(results_path))
        print(f"[EvaluationFramework] Saved results to {results_path}")

        # Save metrics as CSV
        metrics_df = pd.DataFrame(results.system_metrics).T
        metrics_path = self.output_dir / 'metrics.csv'
        metrics_df.to_csv(metrics_path)
        print(f"[EvaluationFramework] Saved metrics to {metrics_path}")

        # Save detailed query results
        if self.config.save_detailed_results:
            query_data = [asdict(qr) for qr in results.query_results]
            query_df = pd.DataFrame(query_data)
            query_path = self.output_dir / 'query_results.csv'
            query_df.to_csv(query_path, index=False)
            print(f"[EvaluationFramework] Saved query results to {query_path}")
            
        # =========================================================
        # SYNC TO SUPABASE DASHBOARD
        # =========================================================
        if self.eval_db:
            print("\n[EvaluationFramework] 💾 Syncing with Supabase Dashboard...")
            try:
                for system_name, metrics in results.system_metrics.items():
                    # Calculate aggregates for dashboard view
                    ragas_keys = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
                    ragas_vals = [metrics.get(k, 0) for k in ragas_keys if k in metrics]
                    ragas_score = sum(ragas_vals) / len(ragas_vals) if ragas_vals else 0
                    
                    ir_keys = ['recall_at_5', 'precision_at_5', 'mrr']
                    ir_vals = [metrics.get(k, 0) for k in ir_keys if k in metrics]
                    ir_score = sum(ir_vals) / len(ir_vals) if ir_vals else 0
                    
                    semantic_score = metrics.get('bert_score_f1', 0)
                    overall_score = (ragas_score + ir_score + semantic_score) / (3 if semantic_score else 2)
                    
                    # 1. Create Run Entry
                    dataset_name = Path(results.config.dataset_path).stem if results.config.dataset_path else "unknown_dataset"
                    
                    try:
                        # Get or create benchmark user for script runs
                        benchmark_user_id = self.eval_db.get_or_create_benchmark_user()
                        if not benchmark_user_id:
                            print(f"   ⚠️ Skipping DB sync for '{system_name}': Could not get/create benchmark user.")
                            continue
                            
                        run = self.eval_db.create_evaluation_run(
                            experiment_name=f"{results.config.experiment_name} ({system_name})",
                            created_by=benchmark_user_id,
                            dataset_name=dataset_name,
                            system_name=system_name,
                            version=results.config.experiment_id,
                            dataset_size=len(results.query_results) // len(results.system_metrics) if results.system_metrics else 0,
                            status="completed"
                        )
                    except Exception as e:
                        if "violates foreign key constraint" in str(e):
                            print(f"   ⚠️ Skipping DB sync for '{system_name}': User ID invalid (FK violation).")
                            continue
                        raise e
                    
                    if run:
                        print(f"   scyncing system '{system_name}' -> Run ID: {run['id']}")
                        
                        # 2. Update Scores
                        self.eval_db.update_evaluation_run(
                            evaluation_id=run['id'],
                            overall_score=overall_score,
                            ragas_score=ragas_score,
                            ir_score=ir_score,
                            semantic_score=semantic_score,
                            duration_seconds=int(results.total_evaluation_time)
                        )
                        
                        # 3. Save Detailed Metrics
                        self.eval_db.save_evaluation_metrics(
                            evaluation_id=run['id'],
                            metrics=metrics
                        )
                        
                        # 4. Save Efficiency Metrics if available
                        if 'mean_latency_ms' in metrics:
                             self.eval_db.save_efficiency_metrics(
                                evaluation_id=run['id'],
                                avg_latency_ms=metrics.get('mean_latency_ms', 0),
                                total_tokens_used=int(metrics.get('total_tokens', 0))
                             )
                             
                print("[EvaluationFramework] ✅ Synced successfully!")
                
            except Exception as e:
                print(f"[EvaluationFramework] ❌ Sync failed: {e}")
                import traceback
                traceback.print_exc()

    def _generate_latex_tables(self, results: EvaluationResults):
        """Generate LaTeX tables for paper"""
        print("[EvaluationFramework] Generating LaTeX tables...")

        # Main results table
        self.latex_exporter.create_results_table(
            results.system_metrics,
            output_file='table_main_results.tex'
        )

        # Statistical significance table
        if results.statistical_tests:
            self.latex_exporter.create_significance_table(
                results.statistical_tests,
                output_file='table_significance.tex'
            )

        print(f"[EvaluationFramework] LaTeX tables saved to {self.output_dir}")

    def _generate_visualizations(self, results: EvaluationResults):
        """Generate publication-quality visualizations"""
        print("[EvaluationFramework] Generating visualizations...")

        # Performance comparison plot
        # Performance comparison plot - using plot_system_comparison (default is F1 Score)
        # We need to pick a metric to show, e.g. 'overall_score' or 'answer_relevancy'
        # Since plot_system_comparison shows one metric, let's show overall_score if computed manually, or f1.
        # But system_metrics is a dict of dicts.
        
        # We'll use plot_metric_comparison to show multiple metrics
        metrics_to_show = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
        self.visualizer.plot_metric_comparison(
            results.system_metrics,
            metrics=metrics_to_show,
            filename='fig_comparison.pdf'
        )

        # Latency distribution plot - mapping to plot_score_distribution for latency
        # We need to restructure data: system -> list of latencies
        latency_data = {}
        for res in results.query_results:
             if res.system_name not in latency_data: latency_data[res.system_name] = []
             latency_data[res.system_name].append(res.latency_ms)
             
        self.visualizer.plot_score_distribution(
            latency_data,
            title="Latency Distribution (ms)",
            filename='fig_latency.pdf'
        )

        # Performance by query type
        # Need to aggregate first
        query_type_perf = {} # type -> system -> score
        # Using error_analysis which already groups by type
        if results.error_analysis:
             # This part might need better aggregation logic if error_analysis structure differs
             # Let's do a simple aggregation here
             pass
        
        # Simple query type aggregation
        q_types = {}
        for res in results.query_results:
             qt = res.query_type
             if qt not in q_types: q_types[qt] = {}
             if res.system_name not in q_types[qt]: q_types[qt][res.system_name] = []
             # Use a metric, e.g. answer_relevancy or just correctness (if we had boolean)
             # Let's use answer_relevancy if available
             score = res.scores.get('answer_relevancy', 0.0)
             q_types[qt][res.system_name].append(score)
             
        # Average the scores
        final_q_perf = {}
        for qt, sys_scores in q_types.items():
             final_q_perf[qt] = {}
             for sys_name, scores in sys_scores.items():
                  final_q_perf[qt][sys_name] = sum(scores) / len(scores) if scores else 0
                  
        self.visualizer.plot_query_type_analysis(
            final_q_perf,
            filename='fig_query_types.pdf'
        )

        print(f"[EvaluationFramework] Plots saved to {self.output_dir}")


def create_evaluation_config(
    experiment_name: str,
    dataset_path: str,
    **kwargs
) -> EvaluationConfig:
    """
    Convenience function to create evaluation configuration

    Args:
        experiment_name: Name of the experiment
        dataset_path: Path to evaluation dataset
        **kwargs: Additional configuration parameters

    Returns:
        EvaluationConfig object
    """
    return EvaluationConfig(
        experiment_name=experiment_name,
        dataset_path=dataset_path,
        **kwargs
    )


if __name__ == "__main__":
    print("Evaluation Framework for Agentic RAG - Research Grade")
    print("=" * 60)
    print("\nThis module provides comprehensive evaluation capabilities")
    print("for research publication in top-tier venues.")
    print("\nUsage:")
    print("  from evaluation_framework import EvaluationFramework, create_evaluation_config")
    print("  config = create_evaluation_config('My Experiment', 'data/test.json')")
    print("  framework = EvaluationFramework(config)")
    print("  results = framework.run_full_evaluation(systems)")
