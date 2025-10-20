# analyze_rq1_bird_failures.py
import json
import os
import sqlite3
from collections import defaultdict
import sqlglot
from sqlglot import exp

# --- CONFIGURATION ---
FAILURE_REPORT_PATHS = [
    "outputs/bird/failures_report_dynaquery.json",
    "outputs/bird/failures_report_rag.json"
]
BIRD_DB_ROOT = "external_data/bird/dev_databases" 

# Cache to store actual database schemas to avoid redundant lookups
SCHEMA_CACHE = {}

def get_actual_schema(db_id: str):
    if db_id in SCHEMA_CACHE:
        return SCHEMA_CACHE[db_id]

    db_path = os.path.join(BIRD_DB_ROOT, db_id, f"{db_id}.sqlite")
    if not os.path.exists(db_path):
        return None

    try:
        con = sqlite3.connect(db_path)
        cursor = con.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {row[0].lower() for row in cursor.fetchall()}

        columns = set()
        for table in tables:
            cursor.execute(f"PRAGMA table_info(`{table}`)")
            columns.update({row[1].lower() for row in cursor.fetchall()})
        con.close()

        schema = {"tables": tables, "columns": columns}
        SCHEMA_CACHE[db_id] = schema
        return schema
    except Exception as e:
        print(f"Warning: Could not connect to or parse DB: {db_id}. Error: {e}")
        return None


def classify_error(pred_sql: str, true_sql: str, db_id: str) -> str:
    if not pred_sql.lower().strip().startswith(('select', 'with')):
        return "REFUSAL_TO_ANSWER"

    try:
        pred_ast = sqlglot.parse_one(pred_sql, read="mysql")
    except sqlglot.errors.ParseError:
        return "SYNTAX_ERROR"

    actual_schema = get_actual_schema(db_id)
    if actual_schema:
        predicted_tables = {t.name.lower() for t in pred_ast.find_all(exp.Table)}
        predicted_columns = {c.name.lower() for c in pred_ast.find_all(exp.Column)}
        if (predicted_tables - actual_schema["tables"]) or (predicted_columns - actual_schema["columns"]):
            return "SCHEMA_HALLUCINATION"

    try:
        true_ast = sqlglot.parse_one(true_sql, read="sqlite")
    except sqlglot.errors.ParseError:
        return "GROUND_TRUTH_PARSE_ERROR"

    true_tables = {t.name.lower() for t in true_ast.find_all(exp.Table)}
    predicted_tables = {t.name.lower() for t in pred_ast.find_all(exp.Table)}
    if true_tables != predicted_tables:
        return "JOIN_TABLE_MISMATCH"

    true_selects = {str(s).lower() for s in true_ast.find(exp.Select).expressions}
    pred_selects = {str(s).lower() for s in pred_ast.find(exp.Select).expressions}
    if true_selects != pred_selects:
        true_aggs = {type(agg) for agg in true_ast.find_all(exp.AggFunc)}
        pred_aggs = {type(agg) for agg in pred_ast.find_all(exp.AggFunc)}
        if true_aggs != pred_aggs:
            return "SELECT_AGGREGATION_MISMATCH"
        return "SELECT_COLUMN_MISMATCH"

    if true_ast.find(exp.Where) and not pred_ast.find(exp.Where):
        return "WHERE_CLAUSE_MISSING"

    return "WHERE_OR_LOGIC_ERROR"


def analyze_report(path: str):
    if not os.path.exists(path):
        print(f"ERROR: Failure report not found at '{path}'")
        return

    if not os.path.exists(BIRD_DB_ROOT):
        print(f"ERROR: BIRD database root not found at '{BIRD_DB_ROOT}'")
        return

    with open(path, 'r', encoding='utf-8') as f:
        failures = json.load(f)

    error_counts = defaultdict(int)
    analyzed_failures = []

    print(f"\nAnalyzing {len(failures)} failures from '{path}'...")

    for failure in failures:
        error_category = classify_error(
            failure["predicted_sql"],
            failure["ground_truth_sql"],
            failure["db_id"]
        )
        error_counts[error_category] += 1
        failure_with_analysis = failure.copy()
        failure_with_analysis["error_category"] = error_category
        analyzed_failures.append(failure_with_analysis)

    total_failures = len(failures)
    print("\n--- Programmatic Failure Analysis Report ---")
    print(f"{'Error Category':<30} | {'Count':>10} | {'Percentage':>12}")
    print("-" * 60)
    sorted_errors = sorted(error_counts.items(), key=lambda item: item[1], reverse=True)
    for category, count in sorted_errors:
        percentage = (count / total_failures * 100) if total_failures > 0 else 0
        print(f"{category:<30} | {count:>10} | {percentage:>11.2f}%")
    print("-" * 60)
    print(f"{'Total Analyzed':<30} | {total_failures:>10} | {'100.00%':>12}")

    output_path = path.replace('.json', '_analyzed.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analyzed_failures, f, indent=4)
    print(f"\nDetailed analysis saved to: '{output_path}'")


def main():
    for path in FAILURE_REPORT_PATHS:
        analyze_report(path)


if __name__ == "__main__":
    main()
