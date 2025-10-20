# calculate_true_accuracy_with_failure_reports.py
import json
import os
from collections import defaultdict

# --- Configuration ---
BIRD_SAMPLE_PATH = "data_samples/bird_dev_sample_500.json"
BIRD_DEV_FULL_PATH = "external_data/bird/dev.json" 
PREDICTED_SQL_DYNAQUERY = "outputs/bird_dynaquery/predict_dev.json"
PREDICTED_SQL_RAG = "outputs/bird_rag/predict_dev.json"

RESULTS_FILE_DYNAQUERY = "outputs/bird/dynaquery_bird_results_ea.json"
RESULTS_FILE_RAG = "outputs/bird/dynaquery_rag_results_ea.json"


def calculate_scores(model_name: str, results_file_path: str, predicted_sql_path: str):
    """
    Loads full evaluation results, calculates accuracy on the sample,
    and generates a single, consolidated JSON report for all failed queries.
    """
    print(f"\n--- Analyzing Model: {model_name.upper()} ---")
    print(f"--- Using Results From: {os.path.basename(results_file_path)} ---")

    # 1. Load data sources
    try:
        with open(BIRD_SAMPLE_PATH, 'r', encoding='utf-8') as f:
            sample_data = json.load(f)
        with open(BIRD_DEV_FULL_PATH, 'r', encoding='utf-8') as f:
            full_dev_data = json.load(f)
        with open(results_file_path, 'r', encoding='utf-8') as f:
            eval_results = json.load(f)
        with open(predicted_sql_path, 'r', encoding='utf-8') as f:
            predicted_sqls_raw = json.load(f)
    except FileNotFoundError as e:
        print(f"ERROR: Could not find a required file: {e.filename}")
        return

    # 2. Prepare lookups for efficient processing
    sample_info = {item['question_id']: item['difficulty'] for item in sample_data}
    sample_ids = set(sample_info.keys())
    
    index_to_dev_item = {i: item for i, item in enumerate(full_dev_data)}

    predicted_sqls = {}
    for key, value in predicted_sqls_raw.items():
        if isinstance(value, str):
            predicted_sqls[int(key)] = value.split('\t----- bird -----\t')[0].strip()
        else:
            predicted_sqls[int(key)] = "SKIPPED_OR_INVALID_PREDICTION"

    # 3. Filter results, aggregate scores, and collect failures
    correct_by_difficulty = defaultdict(int)
    total_by_difficulty = defaultdict(int)
    
    # Create a list to hold all failure reports for this model
    all_failures = []

    for result in eval_results:
        idx = result['sql_idx']
        dev_item = index_to_dev_item.get(idx)
        
        if not dev_item:
            continue
        
        question_id = dev_item['question_id']
        
        if question_id in sample_ids:
            difficulty = sample_info[question_id]
            total_by_difficulty[difficulty] += 1
            
            if result['res'] == 1:
                correct_by_difficulty[difficulty] += 1
            else:
                # This is a failure, so we construct a report dictionary
                failure_report = {
                    "question_id": question_id,
                    "difficulty": difficulty,
                    "db_id": dev_item.get("db_id", "N/A"),
                    "question": dev_item.get("question", "N/A"),
                    "evidence": dev_item.get("evidence", ""),
                    "ground_truth_sql": dev_item.get("SQL", "N/A"),
                    "predicted_sql": predicted_sqls.get(idx, "PREDICTION NOT FOUND")
                }
                all_failures.append(failure_report)

    output_dir = os.path.dirname(results_file_path)
    report_path = os.path.join(output_dir, f"failures_report_{model_name}.json")
    print(f"Consolidated failure report saved to: '{report_path}'")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(all_failures, f, indent=4)


    # 4. Calculate and print the final, true accuracies
    difficulties = ['simple', 'moderate', 'challenging']
    
    print(f"\n{'Difficulty':<15} | {'Correct':>10} | {'Total in Sample':>15} | {'Accuracy':>10}")
    print("-" * 65)

    total_correct = 0
    total_count = 0

    for diff in difficulties:
        correct = correct_by_difficulty.get(diff, 0)
        total = total_by_difficulty.get(diff, 0)
        accuracy = (correct / total * 100) if total > 0 else 0.0
        print(f"{diff:<15} | {correct:>10} | {total:>15} | {accuracy:>9.2f}%")
        total_correct += correct
        total_count += total
    
    print("-" * 65)
    overall_accuracy = (total_correct / total_count * 100) if total_count > 0 else 0.0
    print(f"{'Overall':<15} | {total_correct:>10} | {total_count:>15} | {overall_accuracy:>9.2f}%")


if __name__ == "__main__":
    calculate_scores(
        model_name="dynaquery",
        results_file_path=RESULTS_FILE_DYNAQUERY,
        predicted_sql_path=PREDICTED_SQL_DYNAQUERY
    )
    
    calculate_scores(
        model_name="rag",
        results_file_path=RESULTS_FILE_RAG,
        predicted_sql_path=PREDICTED_SQL_RAG
    )