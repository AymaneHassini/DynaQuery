import pandas as pd
import time
import json
from tqdm import tqdm
import re
import sys
import os
import argparse
# This line adds the parent directory (dynaquery-impl) to Python's path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dynaquery.chains.mmp import invoke_mmp

def calculate_metrics(ground_truth_pids, system_pids):
    """Calculates precision, recall, and F1-score."""
    gt_set = set(pid.strip() for pid in ground_truth_pids if pid.strip())
    sys_set = set(pid.strip() for pid in system_pids if pid.strip())

    if not gt_set and not sys_set:
        return 1.0, 1.0, 1.0, set(), set(), set()

    true_positives_set = gt_set.intersection(sys_set)
    false_positives_set = sys_set - gt_set
    false_negatives_set = gt_set - sys_set
    
    true_positives = len(true_positives_set)
    
    precision = true_positives / len(sys_set) if sys_set else 0.0
    recall = true_positives / len(gt_set) if gt_set else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1_score, true_positives_set, false_positives_set, false_negatives_set

def run_mmp_benchmark(benchmark_file: str, results_file: str, classifier_type: str):   
    """
    Runs the full RQ3 benchmark suite for the MMP pipeline.
    """
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    print(f"Loading benchmark suite from: {benchmark_file}")
    try:
        suite_df = pd.read_csv(benchmark_file, sep=',', quotechar='"', skipinitialspace=True)
    except FileNotFoundError:
        print(f"ERROR: Benchmark file not found at '{benchmark_file}'.")
        return
    except Exception as e:
        print(f"ERROR: Failed to parse CSV file. Check formatting. Details: {e}")
        return

    results = []
    
    for index, row in tqdm(suite_df.iterrows(), total=suite_df.shape[0], desc=f"Running Benchmark ({classifier_type.upper()})"):
        query_id = row['query_id']
        query_text = row['query_text']
        ground_truth_pids = row['ground_truth_pids'].split(';') if pd.notna(row['ground_truth_pids']) else []
        
        print(f"\n\n{'='*80}")
        print(f"--- Running Query ID: {query_id} (Hardness: {row['hardness']}) ---")
        print(f"Query: {query_text}")
        print(f"Ground Truth PIDs ({len(ground_truth_pids)}): {ground_truth_pids}")
        print(f"{'-'*80}")

        start_time = time.time()
        
        try:
            response_obj = invoke_mmp(query_text, messages=[], classifier_type=classifier_type)
            system_response_pids = response_obj.get("accepted_pids", [])
        except Exception as e:
            print(f"FATAL ERROR during invoke_mmp for {query_id}: {e}")
            system_response_pids = [] # Treat as failure

        end_time = time.time()
        latency = end_time - start_time
        
        (precision, recall, f1_score, 
         tp_set, fp_set, fn_set) = calculate_metrics(ground_truth_pids, system_response_pids)
        
        # --- detailed DEBUG output ---
        print(f"\n--- Analysis for {query_id} ---")
        print(f"System Response PIDs ({len(system_response_pids)}): {system_response_pids}")
        print(f"  - True Positives ({len(tp_set)}): {list(tp_set)}")
        print(f"  - False Positives ({len(fp_set)}): {list(fp_set)}")
        print(f"  - False Negatives ({len(fn_set)}): {list(fn_set)}")
        print(f"  - Precision: {precision:.2f}, Recall: {recall:.2f}")
        print(f"  - F1-Score: {f1_score:.2f}, Latency: {latency:.2f}s")
        print(f"{'='*80}\n")

        results.append({
            'query_id': query_id,
            'hardness': row['hardness'],
            'query_text': query_text,
            'ground_truth_count': len(ground_truth_pids),
            'system_response_count': len(system_response_pids),
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'latency_seconds': latency,
            'ground_truth_pids': ';'.join(ground_truth_pids), 
            'system_response_pids': ';'.join(system_response_pids)
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(results_file, index=False)
    
    print(f"\n--- Benchmark Complete for {classifier_type.upper()} ---")
    print(f"Results saved to: {results_file}")
    
    print("\n--- Overall Summary ---")
    print(f"Average F1-Score: {results_df['f1_score'].mean():.3f}")
    print(f"Average Precision: {results_df['precision'].mean():.3f}")
    print(f"Average Recall: {results_df['recall'].mean():.3f}")
    print(f"Average Latency: {results_df['latency_seconds'].mean():.2f} seconds")

    print("\n--- Summary by Hardness ---")
    summary = results_df.groupby('hardness')[['f1_score', 'precision', 'recall']].mean()
    print(summary)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark_file', type=str, required=True, help="Path to the benchmark suite CSV file.")
    parser.add_argument('--results_file', type=str, required=True, help="Path to save the output results CSV file.")
    parser.add_argument('--classifier', type=str, default="llm", choices=["llm", "bert"], help="Classifier to use within the MMP.")
    args = parser.parse_args()
    
    run_mmp_benchmark(
        benchmark_file=args.benchmark_file, 
        results_file=args.results_file,
        classifier_type=args.classifier
    )