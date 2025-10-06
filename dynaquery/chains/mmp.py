# chains/mmp.py
"""
Final, correct, and robust MMP implementation with enhanced debugging.
This version uses a true two-step "Reason -> Classify" architecture for both
BERT and LLM-native classifiers, ensuring a fair and direct comparison.
"""
from tqdm import tqdm
from sqlalchemy import inspect, types
import torch
import torch.nn.functional as F
from data.db_connector import get_langchain_db, get_query_tool
from data.schema_utils import table_chain, get_table_details, filter_schema_for_tables
from models.llm import load_llm
from models.classifier import load_classifier_bert
from preprocessing.image import preprocess_and_check_image
from prompts.templates import REASONING_GUIDELINES
from chains.llm_classifier import get_llm_native_classifier_chain, classify_with_llm
from utils.join import generate_left_join_query
from chains.answer_chain import format_answer

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
    print("      STEP 1: RATIONALE GENERATION (API Call #1)")
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
    Invoke the advanced integrated pipeline with a specified classifier.
    """
    
    # 1) Get the explicit query plan
    full_schema = get_table_details()
    query_plan_result = table_chain.invoke({"input": user_query})
    if not query_plan_result: return "I could not devise a plan to answer that query."
    query_plan = query_plan_result[0]
    base_table = query_plan.base_table
    all_tables = [base_table] + query_plan.join_tables
    filtered_schema = filter_schema_for_tables(all_tables)

    # 2) Build candidate_sql
    candidate_sql = generate_left_join_query(filtered_schema, base_table, query_plan.join_tables)

    # 3) Execute & fetch
    db = get_langchain_db()
    quote_char = db._engine.dialect.identifier_preparer.quote
    tool = get_query_tool()
    conn = db._engine.raw_connection()
    cursor = conn.cursor()
    cursor.execute(candidate_sql)
    rows = cursor.fetchall()
    columns = [d[0] for d in cursor.description]

    # 4) Robust, Multi-Column Multimodal Content Detection
    insp = inspect(db._engine)
    multimodal_indices = {"image": [], "document": []}
    potential_mm_columns = []
    for i, col_name in enumerate(columns):
        original_table = None
        for table in all_tables:
            try:
                table_cols = [c['name'] for c in insp.get_columns(table)]
                if col_name in table_cols:
                    original_table = table
                    break
            except Exception: continue
        if not original_table: continue
        col_info = next((c for c in insp.get_columns(original_table) if c["name"] == col_name), None)
        if not col_info: continue
        col_type = col_info["type"]
        if isinstance(col_type, (types.String, types.Text, types.VARCHAR, types.LargeBinary)):
            potential_mm_columns.append({"name": col_name, "index": i, "table": original_table})
    if potential_mm_columns:
        select_clauses = [f'(SELECT {quote_char(col["name"])} FROM {quote_char(col["table"])} WHERE {quote_char(col["name"])} IS NOT NULL LIMIT 1) AS {quote_char(col["name"])}' for col in potential_mm_columns]
        sniffing_sql = f"SELECT {', '.join(select_clauses)};"
        try:
            sniff_cursor = conn.cursor()
            sniff_cursor.execute(sniffing_sql)
            first_non_null_values = sniff_cursor.fetchone()
            sniff_cursor.close()
            if first_non_null_values:
                for i, col in enumerate(potential_mm_columns):
                    sample_value = first_non_null_values[i]
                    if isinstance(sample_value, str):
                        low = sample_value.lower()
                        if low.endswith((".jpg", ".jpeg", ".png")):
                            multimodal_indices["image"].append(col["index"])
                        elif low.endswith((".pdf", ".doc", ".docx", ".ppt", ".pptx")):
                            multimodal_indices["document"].append(col["index"])
        except Exception as e:
            print(f"WARNING ▶ Content sniffing query failed: {e}.")

    # 5) Primary key detection
    pk_info = insp.get_pk_constraint(base_table)
    if pk_info and pk_info.get("constrained_columns"):
        pk_col = pk_info["constrained_columns"][0]
    else:
        pk_col = insp.get_columns(base_table)[0]['name']

    # 6) Per-row CoT + Pluggable Classification
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
        print(f"      STEP 2: CLASSIFICATION ({classifier_type.upper()})")
        print("="*50)
        
        predicted_class = -1 # Default to an invalid class

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
            # --- MODIFICATION: Call the updated function ---
            response_obj = classify_with_llm(user_query, llm_rationale, llm_classifier_chain)
            
            if response_obj:
                # --- NEW: Unpack the object and print details ---
                label_map = {"ACCEPT": 0, "RECOMMEND": 1, "REJECT": 2}
                predicted_label_str = response_obj.label.upper()
                predicted_class = label_map.get(predicted_label_str, 2) # Default to REJECT if label is weird
                
                print(f"💬 [DEBUG] LLM Classifier Explanation: \"{response_obj.explanation}\"")
                print(f"✅ [DEBUG] LLM Final Decision: Class {predicted_class} ({predicted_label_str})")
            else:
                # Handle the case where the classifier returned None (an error)
                predicted_class = 2 # Default to REJECT with print statement
                print("❌ [DEBUG] LLM Classifier failed. Defaulting to REJECT.")

        if predicted_class == 0: # ACCEPT
            pk_idx = columns.index(pk_col)
            accepted.append(row[pk_idx])
            print(f"👍 [DEBUG] Row ACCEPTED. PK: {row[pk_idx]}")
        else:
            pk_idx = columns.index(pk_col)
            print(f"👎 [DEBUG] Row REJECTED. PK: {row[pk_idx]}")

    # 7) Final filtered SQL on base table
    if not accepted:
        return "None of the candidate rows passed the advanced reasoning filter."

    base_query = candidate_sql.strip().rstrip(';')
    qualified_pk_col = f"{base_table}.{pk_col}"
    pk_values_str = ', '.join(map(repr, accepted))
    final_sql = f"{base_query} WHERE {qualified_pk_col} IN ({pk_values_str});"
    
    # SQL Debug Print
    print("\n" + "="*50)
    print("      FINAL STEP: SQL GENERATION")
    print("="*50)
    print(f"🔍 [DEBUG] Final Generated SQL:\n{final_sql}")
    
    # 8) Execute final query and format answer
    final_tool = get_query_tool()
    final_rows = final_tool.run(final_sql)
    final_str = "\n".join(map(str, final_rows))
    final_answer = format_answer(user_query, final_sql, final_str, filtered_schema)

    return final_answer