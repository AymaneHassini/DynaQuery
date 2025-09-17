# analyze_hardness.py (Final Version with Heuristic Classifier and Direct Execution)

import json
import argparse
import os
import sqlite3
from tqdm import tqdm

# --- Step 1: Our New, Robust Hardness Classifier ---
# This section replaces the dependency on the broken Spider parser.
# It works directly on SQL query strings.

def classify_hardness_by_keywords(sql_str: str) -> str:
    """
    Classifies SQL query hardness using robust keyword counting.
    This is a heuristic that approximates the original Spider logic without brittle parsing.
    """
    sql_lower = sql_str.lower()

    # Heuristic for Component 2: Nested queries
    count_comp2 = sql_lower.count('intersect') + sql_lower.count('union') + sql_lower.count('except')

    # Heuristic for Component 1: Keywords indicating complexity
    count_comp1 = 0
    # We count 'group by' first to avoid double-counting 'by'
    comp1_keywords = ['group by', 'order by', 'where', 'limit', 'join', 'or', 'like']
    for key in comp1_keywords:
        count_comp1 += sql_lower.count(key)

    # Heuristic for "Others": Multiple aggregations or multiple select columns
    count_others = 0
    agg_ops = ['count(', 'avg(', 'sum(', 'min(', 'max(']
    agg_count = sum(sql_lower.count(op) for op in agg_ops)
    if agg_count > 1:
        count_others += 1
    
    # Count select columns (approximate by counting commas in the SELECT clause)
    # This is a bit tricky, but we can look for the first 'select' and the first 'from'
    try:
        select_clause = sql_lower.split('select', 1)[1].split('from', 1)[0]
        if ',' in select_clause:
            count_others += 1
    except IndexError:
        pass # Not a valid SELECT statement, will be 'easy'

    # Apply a simplified version of the original hardness rules
    if count_comp2 > 0 or (count_comp1 > 2 and count_others > 2):
        return "extra"
    elif count_comp1 > 2 or count_others > 1:
        return "hard"
    elif count_comp1 > 1 or count_others > 0:
        return "medium"
    else:
        return "easy"

# --- Step 2: Our Robust Execution Evaluation Function ---
def eval_exec_match(db_path, p_str, g_str):
    """
    Executes two SQL queries and compares their results.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute(p_str)
        p_res = cursor.fetchall()

        cursor.execute(g_str)
        g_res = cursor.fetchall()
        
        conn.close()

        has_order_by = 'order by' in g_str.lower()
        
        if has_order_by:
            return p_res == g_res
        else:
            return set(map(tuple, p_res)) == set(map(tuple, g_res))
    except Exception:
        # If any part of the execution fails, it's not a match.
        if 'conn' in locals() and conn:
            conn.close()
        return False

# --- Step 3: The Main Orchestration Logic ---
def main():
    parser = argparse.ArgumentParser(description="Run hardness analysis for Spider benchmark results.")
    parser.add_argument('--pred_file', type=str, required=True, help="Path to the prediction .sql file.")
    parser.add_argument('--gold_file', type=str, required=True, help="Path to the corresponding gold .sql file.")
    parser.add_argument('--db_dir', type=str, required=True, help="Path to the database directory.")
    # table_file is no longer needed as we don't use the parser
    # parser.add_argument('--table_file', type=str, required=True, help="Path to the official tables.json file.")
    args = parser.parse_args()

    # --- Load Data ---
    print("Loading gold queries and predictions...")
    with open(args.gold_file, 'r', encoding='utf-8') as f:
        glist = [line.strip().split('\t') for line in f if line.strip()]

    # Robust parsing for multi-line prediction files
    plist = []
    with open(args.pred_file, 'r', encoding='utf-8') as f:
        current_query_lines = []
        for line in f:
            line_stripped = line.strip()
            if not line_stripped: continue
            if '\t' in line:
                query_part, db_name = line.split('\t', 1)
                current_query_lines.append(query_part)
                full_query = " ".join(current_query_lines).replace('\n', ' ').strip()
                plist.append({'query': full_query, 'db': db_name.strip()})
                current_query_lines = []
            else:
                current_query_lines.append(line_stripped)
    
    # --- Initialize Score Counters ---
    scores = {
        'easy': {'total': 0, 'exec': 0}, 'medium': {'total': 0, 'exec': 0},
        'hard': {'total': 0, 'exec': 0}, 'extra': {'total': 0, 'exec': 0},
        'all': {'total': 0, 'exec': 0}
    }
    
    print(f"Analyzing EA breakdown for {len(plist)} predictions...")
    
    # --- Main Loop ---
    for i, p_item in enumerate(tqdm(plist, desc="Evaluating EA by Hardness")):
        if i >= len(glist):
            break
            
        g_str, g_db_id = glist[i]
        p_str, p_db_id = p_item['query'], p_item['db']
        
        # Classify hardness using our robust, parser-free method
        hardness = classify_hardness_by_keywords(g_str)

        scores[hardness]['total'] += 1
        scores['all']['total'] += 1
        
        db_path = os.path.join(args.db_dir, g_db_id, g_db_id + ".sqlite")
        
        # Evaluate using our robust direct execution method
        is_correct = eval_exec_match(db_path, p_str, g_str)
        
        if is_correct:
            scores[hardness]['exec'] += 1
            scores['all']['exec'] += 1

    # --- Print Results ---
    print(f"\n--- Execution Accuracy Breakdown for: {os.path.basename(args.pred_file)} ---")
    print(f"{'Difficulty':<10} | {'Correct':<10} | {'Total':<10} | {'Accuracy':<10}")
    print("-----------------------------------------------------")
    for level in ['easy', 'medium', 'hard', 'extra', 'all']:
        if level in scores and scores[level]['total'] > 0:
            level_scores = scores[level]
            accuracy = (level_scores['exec'] / level_scores['total']) * 100
            print(f"{level:<10} | {level_scores['exec']:<10} | {level_scores['total']:<10} | {accuracy:.1f}%")
    print("-----------------------------------------------------")

if __name__ == "__main__":
    main()

