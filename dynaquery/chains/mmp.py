# chains/mmp.py
"""
Final, correct, and robust MMP implementation with enhanced debugging.
This version implements the 'Filtered Join' architecture for scalability and
uses a true two-step "Reason -> Classify" architecture.
"""
from tqdm import tqdm
from sqlalchemy import inspect, types
import torch
import torch.nn.functional as F
from dynaquery.data.db_connector import get_langchain_db, get_query_tool
from dynaquery.data.schema_utils import table_chain, get_table_details, filter_schema_for_tables
from dynaquery.models.llm import load_llm
from dynaquery.models.classifier import load_classifier_bert
from dynaquery.preprocessing.image import preprocess_and_check_image
from dynaquery.prompts.templates import REASONING_GUIDELINES
from dynaquery.chains.llm_classifier import get_llm_native_classifier_chain, classify_with_llm
from dynaquery.utils.join import  generate_join_clauses
from dynaquery.chains.answer_chain import format_answer
from dynaquery.utils.where_clause import create_where_clause_chain
from dynaquery.utils.sql import clean_sql_query

def process_candidate_row(
    row: tuple,
    columns: list[str],
    multimodal_indices: dict,
    user_prompt: str
) -> str:
    llm = load_llm()
    field_lines = "\n".join(f"- {columns[i]}: {row[i]!r}" for i in range(len(columns)))
    prompt_text = f"""
    User's question: "{user_prompt}"
    Here is one candidate record with all its fields:
    {field_lines}
    {REASONING_GUIDELINES}
    """
    print("\n" + "="*50)
    print("      STEP 4: RATIONALE GENERATION (API Call #1)")
    print("="*50)
    print("📝 [DEBUG] Full text prompt for Rationale Generation:\n", prompt_text)
    llm_content_list = []
    for idx in multimodal_indices.get("image", []):
        if row[idx] is not None:
            try:
                image = preprocess_and_check_image(row[idx])
                llm_content_list.append(image)
            except Exception as e:
                print(f"WARNING ▶ Could not process image from column '{columns[idx]}': {e}")
    for idx in multimodal_indices.get("document", []):
        if row[idx] is not None:
            llm_content_list.append(str(row[idx]))
    llm_content_list.append(prompt_text)
    resp = llm.generate_content(llm_content_list)
    print("📬 [DEBUG] LLM Rationale Generated:", resp.text)
    return resp.text


def invoke_mmp(user_query: str, messages: list[dict], classifier_type: str = "bert") -> str:
    """
    Invoke the MMP with the scalable Filtered Join architecture.
    """
    
    # STEP 1: Get the query plan from the shared SILE.
    print("\n" + "="*50)
    print("      STEP 1: Generating Query Plan (Shared SILE)")
    print("="*50)
    query_plan_result = table_chain.invoke({"input": user_query})
    if not query_plan_result: return {
        "final_answer_string": "I could not devise a plan to answer that query.",
        "accepted_pids": [],
        "status": "SILE_FAILURE"
    }
    query_plan = query_plan_result[0]
    base_table = query_plan.base_table
    all_tables = [base_table] + query_plan.join_tables
    filtered_schema = filter_schema_for_tables(all_tables)
    print(f"✅ [DEBUG] Query Plan Generated. Base Table: '{base_table}', Join Tables: {query_plan.join_tables}")

    # STEP 2: Generate the structured WHERE clause.
    print("\n" + "="*50)
    print("      STEP 2: Generating Structured Filters (API Call)")
    print("="*50)
    where_clause_chain = create_where_clause_chain()
    raw_structured_conditions = where_clause_chain.invoke({
        "schema": filtered_schema,
        "input": user_query,
        "base_table": query_plan.base_table,
        "join_tables": ", ".join(query_plan.join_tables) 
    })
    structured_conditions = clean_sql_query(raw_structured_conditions)

    print(f"🔍 [DEBUG] Generated WHERE conditions: {structured_conditions}")

    # --- STEP 2.5: DYNAMICALLY DISCOVER MULTIMODAL COLUMNS ---
    print("\n" + "="*50)
    print("      STEP 2.5: Discovering Multimodal Columns")
    print("="*50)
    db = get_langchain_db()
    insp = inspect(db._engine)
    all_multimodal_fqns = []
    potential_mm_columns = []
    for table in all_tables:
        try:
            for col_info in insp.get_columns(table):
                if isinstance(col_info["type"], (types.String, types.Text, types.VARCHAR, types.LargeBinary)):
                    potential_mm_columns.append({"name": col_info["name"], "table": table})
        except Exception as e:
            print(f"WARNING ▶ Could not inspect columns for table '{table}': {e}")

    if potential_mm_columns:
        quote_char = db._engine.dialect.identifier_preparer.quote
        select_clauses = [f'(SELECT {quote_char(col["name"])} FROM {quote_char(col["table"])} WHERE {quote_char(col["name"])} IS NOT NULL LIMIT 1) AS {quote_char(col["name"])}' for col in potential_mm_columns]
        sniffing_sql = f"SELECT {', '.join(select_clauses)};"
        print(f" sniffing_sql: {sniffing_sql}")
        conn = None
        try:
            conn = db._engine.raw_connection()
            sniff_cursor = conn.cursor()
            sniff_cursor.execute(sniffing_sql)
            first_non_null_values = sniff_cursor.fetchone()
            sniff_cursor.close()
            if first_non_null_values:
                for i, col in enumerate(potential_mm_columns):
                    sample_value = first_non_null_values[i]
                    if isinstance(sample_value, str):
                        low = sample_value.lower()
                        if low.endswith((".jpg", ".jpeg", ".png", ".pdf", ".doc", ".docx", ".ppt", ".pptx")):
                            all_multimodal_fqns.append(f"{col['table']}.{col['name']}")
        except Exception as e:
            print(f"WARNING ▶ Content sniffing query failed: {e}.")
        finally:
            if conn: conn.close()
    print(f"[DEBUG] Discovered all multimodal columns (fully qualified): {all_multimodal_fqns}")

    # STEP 3: Build the final "Filtered Join" SQL query.
    print("\n" + "="*50)
    print("      STEP 3: Building Filtered Candidate SQL")
    print("="*50)
    raw_from_and_joins = generate_join_clauses(filtered_schema, base_table, query_plan.join_tables)
    cleaned_from_and_joins = clean_sql_query(raw_from_and_joins)
    candidate_sql = f"SELECT * {cleaned_from_and_joins}"
    
    final_where_conditions = []
    if "NO_CONDITIONS" not in structured_conditions:
        final_where_conditions.append(f"({structured_conditions})")
    
    if all_multimodal_fqns:
        is_not_null_checks = " OR ".join([f"{col} IS NOT NULL" for col in all_multimodal_fqns])
        final_where_conditions.append(f"({is_not_null_checks})")
    
    if final_where_conditions:
        candidate_sql += " WHERE " + " AND ".join(final_where_conditions)
    
    candidate_sql += ";"
    print(f"📝 [DEBUG] Final Candidate SQL to be executed:\n{candidate_sql}")

    # STEP 4: Execute the query and fetch the small candidate set.
    db = get_langchain_db()
    
    try:
        conn = db._engine.raw_connection()
        cursor = conn.cursor()
        cursor.execute(candidate_sql)
        rows = cursor.fetchall()
        columns = [d[0] for d in cursor.description]
    except Exception as e:
        print(f"❌ [ERROR] The generated SQL failed to execute.")
        print(f"Error details: {e}")
        # Close the connection on failure
        if 'cursor' in locals() and cursor: cursor.close()
        if 'conn' in locals() and conn: conn.close()
        return {
        "final_answer_string": f"I'm sorry, the generated SQL was invalid: {e}",
        "accepted_pids": [],
        "status": "SQL_EXECUTION_ERROR"
    }

    print(f"✅ [DEBUG] SQL executed successfully. Fetched {len(rows)} candidate rows.")

    if not rows:
        return {
            "final_answer_string": "I found no products that match the structured part of your query.",
            "accepted_pids": [],
            "status": "OK" 
        }

    # STEP 5: Robust, Multi-Column Multimodal Content Detection
    insp = inspect(db._engine)   
    quote_char = db._engine.dialect.identifier_preparer.quote
    multimodal_indices = {"image": [], "document": []}
    for fqn in all_multimodal_fqns:
        simple_name = fqn.split('.')[-1]
        try:
            idx = columns.index(simple_name)
            # Re-check the file extension to correctly categorize
            for r in rows:
                if r[idx] and isinstance(r[idx], str):
                    low = r[idx].lower()
                    if low.endswith((".jpg", ".jpeg", ".png")):
                        if idx not in multimodal_indices["image"]: multimodal_indices["image"].append(idx)
                    elif low.endswith((".pdf", ".doc", ".docx", ".ppt", ".pptx")):
                        if idx not in multimodal_indices["document"]: multimodal_indices["document"].append(idx)
                    break
        except ValueError:
            print(f"WARNING ▶ Discovered multimodal column '{simple_name}' not in final query result.")



    # STEP 6: Primary Key Detection
    pk_info = insp.get_pk_constraint(base_table)
    if pk_info and pk_info.get("constrained_columns"):
        pk_col = pk_info["constrained_columns"][0]
    else:
        pk_col = insp.get_columns(base_table)[0]['name']

    # STEP 7: Per-row CoT + Pluggable Classification
    accepted = []
    
    if classifier_type == "bert":
        print("\nDEBUG ▶ Initializing BERT classifier.")
        trainer, tokenizer = load_classifier_bert()
    elif classifier_type == "llm":
        print("\nDEBUG ▶ Initializing LLM-native classifier chain.")
        llm_classifier_chain = get_llm_native_classifier_chain()
    else:
        raise ValueError("Invalid classifier_type specified. Must be 'bert' or 'llm'.")

    for row_idx, row in enumerate(tqdm(rows, desc=f"Processing with {classifier_type.upper()} classifier")):

        
        print("\n" + "-"*20 + f" Processing Row #{row_idx+1} " + "-"*20)
        llm_rationale = process_candidate_row(row, columns, multimodal_indices, user_query)
                
        print("\n" + "="*50)
        print(f"      STEP 5: CLASSIFICATION ({classifier_type.upper()})")
        print("="*50)
        
        predicted_class = -1
        if classifier_type == "bert":
            bert_input_text = f"Question: {user_query} Answer: {llm_rationale}"
            inputs = tokenizer(bert_input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
            inputs = {k: v.to(trainer.model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = trainer.model(**inputs)
                logits = outputs.logits
            probs = F.softmax(logits, dim=-1).squeeze().tolist()
            predicted_class = probs.index(max(probs))
            class_map = {0: "ACCEPT", 1: "RECOMMEND", 2: "REJECT"}
            print(f"🧠 [DEBUG] BERT Input Text:\n{bert_input_text}")
            print(f"📈 [DEBUG] BERT Logits: {[f'{p:.3f}' for p in logits.squeeze().tolist()]}")
            print(f"📊 [DEBUG] BERT Probabilities: [ACCEPT: {probs[0]:.3f}, RECOMMEND: {probs[1]:.3f}, REJECT: {probs[2]:.3f}]")
            print(f"✅ [DEBUG] BERT Final Decision: Class {predicted_class} ({class_map.get(predicted_class, 'UNKNOWN')})")

        elif classifier_type == "llm":
            print("🧠 [DEBUG] LLM Classifier is being invoked (API Call #2)...")
            response_obj = classify_with_llm(user_query, llm_rationale, llm_classifier_chain)
            
            if response_obj:
                label_map = {"ACCEPT": 0, "RECOMMEND": 1, "REJECT": 2}
                predicted_label_str = response_obj.label.upper()
                predicted_class = label_map.get(predicted_label_str, 2)
                
                print(f"💬 [DEBUG] LLM Classifier Explanation: \"{response_obj.explanation}\"")
                print(f"✅ [DEBUG] LLM Final Decision: Class {predicted_class} ({predicted_label_str})")
            else:
                predicted_class = 2
                print("❌ [DEBUG] LLM Classifier failed. Defaulting to REJECT.")

        if predicted_class == 0: # ACCEPT
            pk_idx = columns.index(pk_col)
            accepted.append(row[pk_idx])
            print(f"👍 [DEBUG] Row ACCEPTED. PK: {row[pk_idx]}")
        else:
            pk_idx = columns.index(pk_col)
            print(f"👎 [DEBUG] Row REJECTED. PK: {row[pk_idx]}")

    # STEP 8: Final Answer Generation
    if not accepted:
        return {
            "final_answer_string": "I found products that matched your criteria, but none passed the final visual check.",
            "accepted_pids": []
        }

    final_sql = f"SELECT * FROM {base_table} WHERE {quote_char(pk_col)} IN ({', '.join(map(repr, accepted))});"
    
    print("\n" + "="*50)
    print("      FINAL STEP: Generating Final Answer")
    print("="*50)
    print(f"🔍 [DEBUG] Final Generated SQL:\n{final_sql}")
    
    tool = get_query_tool()
    final_rows = tool.run(final_sql)
    final_str = "\n".join(map(str, final_rows))
    final_answer = format_answer(user_query, final_sql, final_str, filtered_schema)

    # Close the final database connection
    if 'cursor' in locals() and cursor: cursor.close()
    if 'conn' in locals() and conn: conn.close()

    return {
    "final_answer_string": final_answer,
    "accepted_pids": accepted,
    "status": "OK"
    }