# calculate_true_ves.py
import json
import os
import math
from collections import defaultdict

# --- Configuration ---
BIRD_SAMPLE_PATH = "data_samples/bird_dev_sample_500.json"
BIRD_DEV_FULL_PATH = "external_data/bird/dev.json"

RESULTS_FILE_DYNAQUERY = "outputs/bird/dynaquery_bird_results_ves.json"
RESULTS_FILE_RAG = "outputs/bird/rag_bird_results_ves.json"

def calculate_ves_scores(model_name: str, results_file_path: str):
    """
    Loads full VES evaluation results and calculates the true VES
    based only on the items in our stratified sample.
    """
    print(f"\n--- Calculating True VES for Model: {model_name.upper()} ---")

    # 1. Load sample and dev data for mapping
    try:
        with open(BIRD_SAMPLE_PATH, 'r', encoding='utf-8') as f:
            sample_data = json.load(f)
        with open(BIRD_DEV_FULL_PATH, 'r', encoding='utf-8') as f:
            full_dev_data = json.load(f)
        with open(results_file_path, 'r', encoding='utf-8') as f:
            full_results = json.load(f)
    except FileNotFoundError as e:
        print(f"ERROR: Could not find a required file: {e.filename}")
        return

    # 2. Prepare lookups
    sample_info = {item['question_id']: item['difficulty'] for item in sample_data}
    sample_ids = set(sample_info.keys())
    index_to_question_id = {i: item['question_id'] for i, item in enumerate(full_dev_data)}

    # 3. Filter results and aggregate scores by difficulty
    scores_by_difficulty = defaultdict(list)
    total_by_difficulty = defaultdict(int)

    for result in full_results:
        idx = result['sql_idx']
        question_id = index_to_question_id.get(idx)
        
        if question_id in sample_ids:
            difficulty = sample_info[question_id]
            total_by_difficulty[difficulty] += 1
            # Add the raw time_ratio to the list for that difficulty
            scores_by_difficulty[difficulty].append(result.get('time_ratio', 0))

    # 4. Calculate and print the final, true VES scores
    difficulties = ['simple', 'moderate', 'challenging']
    
    print(f"\n{'Difficulty':<15} | {'Total in Sample':>15} | {'VES Score':>10}")
    print("-" * 50)

    all_scores = []
    total_count = 0

    for diff in difficulties:
        scores = scores_by_difficulty.get(diff, [])
        total = total_by_difficulty.get(diff, 0)
        
        # Calculate VES: sum of sqrt(ratio) / total_count
        ves_sum = sum(math.sqrt(ratio) for ratio in scores if ratio > 0)
        ves_score = (ves_sum / total * 100) if total > 0 else 0.0
        
        print(f"{diff:<15} | {total:>15} | {ves_score:>9.2f}")
        all_scores.extend(scores)
        total_count += total
    
    print("-" * 50)
    overall_ves_sum = sum(math.sqrt(ratio) for ratio in all_scores if ratio > 0)
    overall_ves = (overall_ves_sum / total_count * 100) if total_count > 0 else 0.0
    print(f"{'Overall':<15} | {total_count:>15} | {overall_ves:>9.2f}")


if __name__ == "__main__":
    calculate_ves_scores(
        model_name="dynaquery",
        results_file_path=RESULTS_FILE_DYNAQUERY
    )
    
    calculate_ves_scores(
        model_name="rag",
        results_file_path=RESULTS_FILE_RAG
    )