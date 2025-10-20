import pandas as pd
import time
from tqdm import tqdm
import sys
import os
import ast # Used for safely evaluating string representations of results
import argparse
# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dynaquery.chains.sqp import invoke_sqp
from dynaquery.data.db_connector import get_langchain_db

def are_results_equivalent(result1, result2):
    """
    Compares two SQL query results for equivalence.
    Handles strings, lists of tuples, etc.
    """
    def to_set_of_tuples(res):
        # Safely evaluate string representations
        if isinstance(res, str):
            try:
                res = ast.literal_eval(res)
            except (ValueError, SyntaxError):
                return { (res,) }
        
        if not isinstance(res, list):
            return { (res,) }
        
        # Convert list of single-item lists/tuples to list of values
        if all(isinstance(item, (list, tuple)) and len(item) == 1 for item in res):
            res = [item[0] for item in res]

        # Convert list of values to a set of tuples for comparison
        return set(tuple(item) if isinstance(item, list) else (item,) for item in res)

    return to_set_of_tuples(result1) == to_set_of_tuples(result2)

def run_sqp_benchmark(benchmark_file: str, results_file: str):
    """
    Runs the full benchmark suite for the SQP pipeline.
    """
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    print(f"Loading benchmark suite from: {benchmark_file}")
    suite_df = pd.read_csv(benchmark_file)

    results = []
    db = get_langchain_db()
    
    for index, row in tqdm(suite_df.iterrows(), total=suite_df.shape[0], desc="Running SQP Benchmark"):
        query_id = row['query_id']
        query_text = row['query_text']
        ground_truth_sql = row['ground_truth_sql']
        
        print(f"\n--- Running Query ID: {query_id} (Hardness: {row['hardness']}) ---")
        print(f"Query: {query_text}")

        start_time = time.time()
        
        # Call with return_dict=True to get all the data
        response_obj = invoke_sqp(query_text, messages=[], return_dict=True)
        
        generated_sql = response_obj.get("cleaned_query", "ERROR")
        system_result = response_obj.get("result", "")
        
        end_time = time.time()
        latency = end_time - start_time
        
        try:
            ground_truth_result = db.run(ground_truth_sql)
        except Exception as e:
            print(f"ERROR: Ground truth SQL for {query_id} failed: {e}")
            ground_truth_result = "GT_SQL_ERROR"

        is_correct = are_results_equivalent(ground_truth_result, system_result)
        
        results.append({
            'query_id': query_id,
            'hardness': row['hardness'],
            'execution_accuracy': 1 if is_correct else 0,
            'latency_seconds': latency,
            'generated_sql': generated_sql,
        })
        
        print(f"Result for {query_id}: Correct = {is_correct}, Latency = {latency:.2f}s")
        if not is_correct:
            print(f"  - System Result: {system_result}")
            print(f"  - Ground Truth Result: {ground_truth_result}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(results_file, index=False)
    
    print(f"\n--- SQP Benchmark Complete ---")
    print(f"Results saved to: {results_file}")
    
    print("\n--- Overall Summary ---")
    print(f"Average Execution Accuracy: {results_df['execution_accuracy'].mean():.3f}")
    
    print("\n--- Summary by Hardness ---")
    summary = results_df.groupby('hardness')['execution_accuracy'].mean()
    print(summary)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark_file', type=str, required=True, help="Path to the benchmark suite CSV file.")
    parser.add_argument('--results_file', type=str, required=True, help="Path to save the output results CSV file.")
    args = parser.parse_args()
    
    run_sqp_benchmark(benchmark_file=args.benchmark_file, results_file=args.results_file)