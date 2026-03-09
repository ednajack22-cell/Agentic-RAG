"""
compute_agentic_metrics.py — Post-Hoc Agentic Metric Computation
================================================================

Parses benchmark results.json to extract agentic-specific metrics:
  - Repair Rate (% queries requiring > 1 attempt)
  - Repair Success Rate
  - Model Tier Usage (Flash vs Pro)
  - Failure Code Distribution
  - Repair Strategy Distribution
  - Per-Tier Cost Breakdown

Usage:
    python compute_agentic_metrics.py <path_to_results.json>
"""

import json
import re
import sys
from collections import Counter
from typing import Dict


def compute_agentic_metrics(results_json_path: str) -> Dict:
    """
    Compute agentic-specific metrics from StepLog data.

    Args:
        results_json_path: Path to the benchmark results.json file

    Returns:
        Dict with agentic metrics
    """
    try:
        with open(results_json_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
    except Exception as e:
        print(f"Could not load results.json: {e}")
        return {}

    total_queries = 0
    queries_with_repair = 0
    total_attempts = 0
    model_tier_counts = Counter()
    failure_code_counts = Counter()
    repair_strategy_counts = Counter()

    flash_cost = 0.0
    pro_cost = 0.0
    repairs_attempted = 0
    repairs_successful = 0

    query_results = results.get('query_results', results.get('results', []))

    for result in query_results:
        total_queries += 1

        agentic_log = result.get('agentic_log') or result.get('metadata', {})

        if isinstance(agentic_log, str):
            try:
                import ast
                agentic_log = ast.literal_eval(agentic_log)
            except Exception:
                agentic_log = {}

        attempts = agentic_log.get('attempts', 1)
        total_attempts += attempts

        if attempts > 1:
            queries_with_repair += 1

        step_logs = agentic_log.get('step_logs', [])
        prev_passed = None

        for i, step_log in enumerate(step_logs):
            if isinstance(step_log, str):
                # Parse string-encoded StepLog
                tier_match = re.search(r"model_tier='(\w+)'", step_log)
                tier = tier_match.group(1) if tier_match else 'unknown'
                model_tier_counts[tier] += 1

                cost_match = re.search(r"cost=([0-9.e-]+)", step_log)
                step_cost = float(cost_match.group(1)) if cost_match else 0.0
                if tier == 'base':
                    flash_cost += step_cost
                elif tier == 'pro':
                    pro_cost += step_cost

                fc_match = re.search(r"'failure_code': (\w+|None|'[^']+')", step_log)
                if fc_match:
                    fc_val = fc_match.group(1).strip("'")
                    if fc_val != 'None':
                        failure_code_counts[fc_val] += 1

                rs_match = re.search(r"repair_strategy=(\w+|None|'[^']+')", step_log)
                if rs_match:
                    rs_val = rs_match.group(1).strip("'")
                    if rs_val != 'None':
                        repair_strategy_counts[rs_val] += 1

                passed_match = re.search(r"'passed': (True|False)", step_log)
                current_passed = passed_match.group(1) == 'True' if passed_match else None
                if i > 0 and prev_passed is False and current_passed is True:
                    repairs_successful += 1
                if i > 0:
                    repairs_attempted += 1
                prev_passed = current_passed

            else:
                # Already a dict
                tier = step_log.get('model_tier', 'unknown')
                model_tier_counts[tier] += 1
                step_cost = step_log.get('cost', 0.0)
                if tier == 'base':
                    flash_cost += step_cost
                elif tier == 'pro':
                    pro_cost += step_cost

                fc = step_log.get('verdict', {}).get('failure_code')
                if fc:
                    failure_code_counts[fc] += 1

                rs = step_log.get('repair_strategy')
                if rs:
                    repair_strategy_counts[rs] += 1

                current_passed = step_log.get('verdict', {}).get('passed')
                if i > 0 and prev_passed is False and current_passed is True:
                    repairs_successful += 1
                if i > 0:
                    repairs_attempted += 1
                prev_passed = current_passed

    # Compute summary
    agentic_metrics = {
        'total_queries': total_queries,
        'repair_rate': round((queries_with_repair / total_queries * 100), 1) if total_queries > 0 else 0,
        'avg_loop_iterations': round(total_attempts / total_queries, 2) if total_queries > 0 else 1,
        'model_tier_usage': dict(model_tier_counts),
        'failure_code_distribution': dict(failure_code_counts),
        'repair_strategy_distribution': dict(repair_strategy_counts),
        'repairs_attempted': repairs_attempted,
        'repairs_successful': repairs_successful,
        'repair_success_rate': round((repairs_successful / repairs_attempted * 100), 1) if repairs_attempted > 0 else 0.0,
    }

    total_calls = sum(model_tier_counts.values())
    if total_calls > 0:
        agentic_metrics['flash_pct'] = round(model_tier_counts.get('base', 0) / total_calls * 100, 1)
        agentic_metrics['pro_pct'] = round(model_tier_counts.get('pro', 0) / total_calls * 100, 1)

    agentic_metrics['flash_total_cost'] = round(flash_cost, 6)
    agentic_metrics['pro_total_cost'] = round(pro_cost, 6)
    agentic_metrics['total_cost'] = round(flash_cost + pro_cost, 6)

    if total_queries > 0:
        agentic_metrics['flash_cost_per_1k'] = round((flash_cost / total_queries) * 1000, 4)
        agentic_metrics['pro_cost_per_1k'] = round((pro_cost / total_queries) * 1000, 4)

    return agentic_metrics


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compute_agentic_metrics.py <results.json>")
        sys.exit(1)

    metrics = compute_agentic_metrics(sys.argv[1])

    if metrics:
        print(json.dumps(metrics, indent=2))
    else:
        print("No metrics computed.")
