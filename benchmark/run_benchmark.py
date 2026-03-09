"""
run_benchmark.py — Main Benchmark Reproduction Script
=====================================================

Reproduces the full 500-query evaluation benchmark:
  - 250 Natural Questions (NQ-Open) + 250 HotpotQA
  - 4 system configurations (Full, No Routing, No Validation, Vanilla)
  - RAGAS faithfulness, BERTScore, IR metrics, agentic metrics

Usage:
    python run_benchmark.py --samples 500 --workers 1

Requirements:
    - GEMINI_API_KEY in .env
    - OPENAI_API_KEY in .env (for RAGAS GPT-4o evaluation)
    - Dependencies from requirements.txt installed

Output:
    Results saved to ../results/
"""

import os
import sys
import json
import argparse
import warnings
import logging
from datetime import datetime
from pathlib import Path

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Suppress noisy warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Fix SSL
try:
    import certifi
    os.environ['SSL_CERT_FILE'] = certifi.where()
except ImportError:
    pass

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

from enhanced_agentic_rag import EnhancedAgenticRAG, EnhancedConfig
from dataset_builder import DatasetBuilder, GroundTruthItem
from evaluation_framework import EvaluationFramework, EvaluationConfig


GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# ============================================================================
# Dataset Downloads
# ============================================================================

def get_cache_dir():
    """Get preferred HuggingFace cache directory."""
    cache_dir = os.environ.get('HF_DATASETS_CACHE')
    if not cache_dir and Path("F:/huggingface_cache").exists():
        cache_dir = "F:/huggingface_cache"
    return cache_dir


def download_hotpotqa(num_samples: int = 250):
    """Download and sample HotpotQA benchmark (seed=42)."""
    print(f"Downloading HotpotQA (sampling {num_samples} questions)...")

    cache_path = Path("../data/hotpotqa_eval.json")
    if cache_path.exists():
        try:
            builder = DatasetBuilder()
            dataset = builder.load_dataset_from_json(str(cache_path))
            print(f"  Found cached HotpotQA with {len(dataset)} items")
            return dataset
        except Exception:
            pass

    from datasets import load_dataset as hf_load
    cache_dir = get_cache_dir()
    dataset = hf_load('hotpot_qa', 'fullwiki', split='validation', cache_dir=cache_dir)
    sampled = dataset.shuffle(seed=42).select(range(num_samples))

    builder = DatasetBuilder()
    items = []

    for i, example in enumerate(sampled):
        context_text = ""
        titles = example['context']['title']
        sentences = example['context']['sentences']
        for title, sent_list in zip(titles, sentences):
            context_text += f"\n=== {title} ===\n"
            context_text += " ".join(sent_list) + "\n"

        items.append(GroundTruthItem(
            query_id=f"hotpotqa_{i+1:04d}",
            question=example['question'],
            ground_truth_answer=example['answer'],
            query_type="MULTI_HOP",
            difficulty="hard",
            domain="general",
            metadata={'dataset': 'hotpotqa', 'ground_truth_context': context_text}
        ))

    print(f"  Selected {len(items)} HotpotQA questions")
    eval_dataset = builder.create_dataset("hotpotqa_eval", f"HotpotQA ({num_samples})", items)
    builder.save_dataset(eval_dataset, str(cache_path))
    return eval_dataset


def download_natural_questions(num_samples: int = 250):
    """Download and sample Natural Questions benchmark (seed=42)."""
    print(f"Downloading Natural Questions (sampling {num_samples} questions)...")

    cache_path = Path("../data/natural_questions_eval.json")
    if cache_path.exists():
        try:
            builder = DatasetBuilder()
            dataset = builder.load_dataset_from_json(str(cache_path))
            items = dataset.items if hasattr(dataset, 'items') else dataset.get('items', [])
            if len(items) > 0:
                print(f"  Found cached NQ with {len(items)} items")
                return dataset
        except Exception:
            pass

    from datasets import load_dataset as hf_load
    cache_dir = get_cache_dir()

    kwargs = {'split': 'validation', 'streaming': False, 'cache_dir': cache_dir} if cache_dir else \
             {'split': 'train', 'streaming': True}

    dataset = hf_load('natural_questions', **kwargs)

    builder = DatasetBuilder()
    items = []
    scanned = 0

    for example in dataset:
        if len(items) >= num_samples:
            break
        scanned += 1

        try:
            anns = example.get('annotations', {})
            if not anns:
                continue

            short_answers_list = anns.get('short_answers', [])
            long_answer_list = anns.get('long_answer', [])
            if not short_answers_list or not long_answer_list:
                continue

            short_ans = short_answers_list[0]
            long_ans = long_answer_list[0]

            has_short = bool(short_ans and short_ans.get('text') and len(short_ans['text']) > 0)
            has_long = long_ans.get('candidate_index', -1) != -1
            if not has_short or not has_long:
                continue

            tokens = example['document']['tokens']['token']
            answer_text = short_ans['text'][0]
            l_start, l_end = long_ans['start_token'], long_ans['end_token']
            context_text = " ".join(tokens[l_start:l_end])
            context_text = context_text.replace('<P>', '').replace('</P>', '')

            items.append(GroundTruthItem(
                query_id=f"nq_{len(items)+1:04d}",
                question=example['question']['text'],
                ground_truth_answer=answer_text,
                query_type="FACTUAL",
                difficulty="medium",
                domain="general",
                metadata={'dataset': 'natural_questions', 'ground_truth_context': context_text}
            ))
        except Exception:
            continue

    print(f"  Selected {len(items)} NQ questions (scanned {scanned})")
    eval_dataset = builder.create_dataset("nq_eval", f"NQ ({len(items)})", items)
    builder.save_dataset(eval_dataset, str(cache_path))
    return eval_dataset


# ============================================================================
# System Configuration
# ============================================================================

def setup_systems():
    """Configure the 4 experimental conditions."""
    systems = {}

    memory_config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {"path": "../data/qdrant_data", "on_disk": True}
        },
        "llm": {
            "provider": "gemini",
            "config": {"model": "gemini-flash-latest", "api_key": GEMINI_API_KEY}
        },
        "embedder": {
            "provider": "gemini",
            "config": {"model": "models/text-embedding-004", "api_key": GEMINI_API_KEY}
        },
        "chunk_size": 1500,
        "version": "v1.1"
    }

    # 1. Full system (Governed Efficient)
    config_full = EnhancedConfig()
    config_full.use_pydantic = True
    config_full.use_structured_output = True
    config_full.use_model_routing = True
    config_full.use_hybrid_search = True
    config_full.use_reranking = True
    config_full.use_multihop = True
    config_full.use_self_reflection = True

    systems['Enhanced RAG (Full)'] = EnhancedAgenticRAG(
        api_key=GEMINI_API_KEY,
        config=config_full,
        memory_config=memory_config
    )

    # 2. No routing — forced Pro (Governed Frontier)
    config_no_routing = EnhancedConfig()
    config_no_routing.use_pydantic = True
    config_no_routing.use_structured_output = True
    config_no_routing.use_model_routing = False
    config_no_routing.forced_model = "gemini-2.5-pro"

    systems['Enhanced RAG (No Routing)'] = EnhancedAgenticRAG(
        api_key=GEMINI_API_KEY,
        config=config_no_routing,
        enable_memory=False
    )

    # 3. No Validation ablation
    config_no_struct = EnhancedConfig()
    config_no_struct.use_pydantic = False
    config_no_struct.use_self_reflection = False

    systems['RAG (No Validation)'] = EnhancedAgenticRAG(
        api_key=GEMINI_API_KEY,
        config=config_no_struct,
        enable_memory=False
    )

    # 4. Vanilla RAG baseline
    config_vanilla = EnhancedConfig()
    config_vanilla.use_hybrid_search = True
    config_vanilla.use_reranking = False
    config_vanilla.use_pydantic = False
    config_vanilla.use_multihop = False
    config_vanilla.use_self_reflection = False
    config_vanilla.use_query_rewriting = False

    systems['Vanilla RAG'] = EnhancedAgenticRAG(
        api_key=GEMINI_API_KEY,
        config=config_vanilla,
        enable_memory=False
    )

    print(f"Configured {len(systems)} systems for comparison")
    return systems


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run benchmark evaluation")
    parser.add_argument("--samples", type=int, default=500,
                        help="Total samples (split 50/50 NQ + HotpotQA)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers (default: 1)")
    args = parser.parse_args()

    print("=" * 60)
    print("AGENTIC RAG BENCHMARK EVALUATION")
    print("=" * 60)

    per_dataset = args.samples // 2

    # 1. Download datasets
    hotpotqa = download_hotpotqa(per_dataset)
    nq = download_natural_questions(per_dataset)

    # 2. Combine with seed=42 shuffle
    import random
    random.seed(42)

    builder = DatasetBuilder()

    def _get_items(ds):
        if not ds:
            return []
        items = ds.items if hasattr(ds, 'items') else ds.get('items', [])
        return [GroundTruthItem.from_dict(i) if isinstance(i, dict) else i for i in items]

    all_items = _get_items(hotpotqa) + _get_items(nq)
    random.shuffle(all_items)

    combined = builder.create_dataset(
        "combined_benchmark",
        f"Combined ({len(all_items)} queries)",
        all_items
    )

    print(f"\nBenchmark: {len(all_items)} queries "
          f"({len(_get_items(hotpotqa))} HotpotQA + {len(_get_items(nq))} NQ)")

    # 3. Setup systems
    systems = setup_systems()

    # 4. Run evaluation
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("../results") / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save runtime dataset slice
    import dataclasses
    slice_path = Path("../data/runtime_eval_slice.json")
    items_to_save = combined.items if hasattr(combined, 'items') else combined.get('items', [])
    json_items = [dataclasses.asdict(i) if dataclasses.is_dataclass(i) else i for i in items_to_save]
    with open(slice_path, 'w', encoding='utf-8') as f:
        json.dump(json_items, f, indent=2)

    config = EvaluationConfig(
        experiment_name="Agentic RAG Benchmark",
        experiment_id=experiment_id,
        dataset_path=str(slice_path),
        num_samples=None,
        evaluate_baselines=True,
        evaluate_ablations=False,
        compute_ragas=True,
        compute_ir_metrics=True,
        compute_semantic=True,
        compute_significance=True,
        significance_level=0.05,
        generate_latex=True,
        generate_plots=True,
        max_workers=args.workers,
        output_dir=str(output_dir.parent)
    )

    # Setup logging
    log_file = output_dir / "execution.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    logging.getLogger().setLevel(logging.INFO)

    framework = EvaluationFramework(config)

    print(f"\nRunning {len(json_items)} queries x {len(systems)} systems...")
    print(f"Output: {output_dir}")
    print(f"Log: {log_file}\n")

    results = framework.run_full_evaluation(systems)

    # 5. Compute agentic metrics
    from compute_agentic_metrics import compute_agentic_metrics
    results_json = output_dir / "results.json"
    if results_json.exists():
        agentic = compute_agentic_metrics(str(results_json))
        with open(output_dir / "agentic_metrics.json", 'w') as f:
            json.dump(agentic, f, indent=2)
        print(f"\nAgentic metrics saved to {output_dir / 'agentic_metrics.json'}")

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
