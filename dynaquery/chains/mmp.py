# chains/mmp.py

"""
Advanced integrated pipeline implementation with row-level reasoning.
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
from preprocessing.text import tokenize
from prompts.templates import REASONING_GUIDELINES
from utils.join import generate_left_join_query
from utils.sql import build_where_clause
from chains.answer_chain import format_answer

def process_candidate_row(
    row: tuple,
    columns: list[str],
    image_idx: int = None,
    doc_idx: int = None,
    user_prompt: str = ""
) -> str:
    """
    Process a single candidate row with the LLM.
    
    Args:
        row: Database row as a tuple
        columns: Column names
        image_idx: Index of image column (if any)
        doc_idx: Index of document column (if any)
        user_prompt: Original user question
        
    Returns:
        str: LLM reasoning about the row
    """
    # Get LLM
    llm = load_llm()
    
    # 1) Render all column:value pairs
    field_lines = "\n".join(f"- {columns[i]}: {row[i]!r}" for i in range(len(columns)))
    print("🔍 [DEBUG] Field lines for this row:\n", field_lines)

    # 2) Build chain-of-thought prompt
    prompt = f"""
    User's question: "{user_prompt}"
    Here is one candidate record with all its fields:
    {field_lines}
    {REASONING_GUIDELINES}
    """
    print("📝 [DEBUG] Full prompt sent to LLM:\n", prompt)

    # 3) Dispatch to correct modality
    if image_idx is not None and row[image_idx] is not None:
        # Preprocess the image
        image = preprocess_and_check_image(row[image_idx])
        # Feed the image and prompt to the LLM
        resp = llm.generate_content([image, "\n\n", prompt])
    elif doc_idx is not None and row[doc_idx] is not None:
        # Feed the document and prompt to the LLM
        document = row[doc_idx]
        resp = llm.generate_content([document, prompt])
    else:
        # Text-only prompt
        resp = llm.generate_content([prompt])
        
    print("📬 [DEBUG] LLM response:", resp.text)
    return resp.text

def invoke_mmp(user_query: str, messages: list[dict]) -> str:
    """Invoke the advanced integrated pipeline."""
    
    # 1) Get the explicit query plan from the upgraded SILE
    full_schema = get_table_details()
    query_plan_result = table_chain.invoke({
        "input": user_query 
    })
    
    if not query_plan_result:
        return "I could not devise a plan to answer that query."
    
    # The result is a list, we take the first plan
    query_plan = query_plan_result[0]
    base_table = query_plan.base_table
    all_tables = [base_table] + query_plan.join_tables
    
    print(f"DEBUG ▶ Query Plan: Base='{base_table}', Joins={query_plan.join_tables}")
    
    filtered_schema = filter_schema_for_tables(all_tables)

    # 2) Build candidate_sql using the explicit query plan
    candidate_sql = generate_left_join_query(filtered_schema, base_table, query_plan.join_tables)
    print("DEBUG ▶ Advanced candidate_sql:", candidate_sql)

    # 3) Execute & fetch
    db = get_langchain_db()
    tool = get_query_tool()
    conn = db._engine.raw_connection()
    cursor = conn.cursor()
    cursor.execute(candidate_sql)
    rows = cursor.fetchall()
    columns = [d[0] for d in cursor.description]
    print("DEBUG ▶ Columns:", columns, "| #rows:", len(rows))

    # 4) Dynamic detection of image/doc columns
    insp = inspect(db._engine)
    cols_info = insp.get_columns(base_table)
    sample_row = rows[0] if rows else None

    image_idx = None
    doc_idx = None
    
    for i, col_name in enumerate(columns):
        info = next((c for c in cols_info if c["name"] == col_name), None)
        if not info:
            continue
        col_type = info["type"]

        if isinstance(col_type, types.LargeBinary):
            if any(tok in col_name.lower() for tok in ("img", "photo", "picture")):
                image_idx = i
            else:
                doc_idx = i
        elif isinstance(col_type, (types.String, types.Text, types.VARCHAR)) and sample_row:
            val = sample_row[i]
            if isinstance(val, str):
                low = val.lower()
                if low.endswith((".jpg", ".jpeg", ".png")):
                    image_idx = i
                elif low.endswith((".pdf", ".doc", ".docx", ".ppt", ".pptx")):
                    doc_idx = i

    print(f"DEBUG ▶ image_idx={image_idx}, doc_idx={doc_idx}")

    # 5) Primary key detection
    insp = inspect(db._engine)
    pk_info = insp.get_pk_constraint(base_table)
    
    if pk_info and pk_info.get("constrained_columns"):
        pk_col = pk_info["constrained_columns"][0]
    else:
        # Fallback to the first column of the explicit base table
        pk_col = insp.get_columns(base_table)[0]['name']

    print(f"DEBUG ▶ Detected Primary Key: '{pk_col}' from base table: '{base_table}'")

    # 6) Per-row CoT + BERT classification (Optimized Single-Call Version)
    trainer, tokenizer = load_classifier_bert()
    accepted = []
    
    for row in tqdm(rows, desc="Processing candidate rows"):
        # Get the full LLM output which includes reasoning and the summary line
        llm_full_output = process_candidate_row(row, columns, image_idx, doc_idx, user_query)
        print("🔖 [DEBUG] Full LLM output received:\n", llm_full_output)
        
        # Parse the output to find the specific line for BERT
        formatted_for_bert = ""
        for line in llm_full_output.splitlines():
            # Use the prefix from your updated REASONING_GUIDELINES
            cleaned_line = line.strip().replace('*', '')
            if cleaned_line.startswith("BERT_SUMMARY:"):
                formatted_for_bert = cleaned_line.replace("BERT_SUMMARY:", "").strip()
                break
        
        # Fallback if the LLM didn't produce the required line
        if not formatted_for_bert:
            print("⚠️ [WARNING] Could not find 'BERT_SUMMARY:' line in LLM output. Skipping row.")
            continue

        # Prepare the final input for BERT using the parsed line
        bert_in = f"Question: {user_query} {formatted_for_bert}"
        print("📝 [DEBUG] Final BERT input:", bert_in)
        
        # Tokenize and classify
        ds = tokenize(bert_in, tokenizer)
        pred = trainer.predict(ds)
        
        # Get probabilities and prediction
        probs = F.softmax(torch.tensor(pred.predictions), dim=-1)
        cls = int(probs.argmax().item())
        print("📊 [DEBUG] BERT probs:", probs.tolist(), "→ class", cls)
        
        # If classified as positive, add to accepted rows
        if cls == 0:
            idx = columns.index(pk_col)
            accepted.append(row[idx])


    if not accepted:
        return "None of the candidate rows passed the advanced reasoning filter."

    # 7) Final filtered SQL on base table
    base_query = candidate_sql.strip().rstrip(';')
    qualified_pk_col = f"{base_table}.{pk_col}"
    pk_values_str = ', '.join(map(str, accepted))
    final_sql = f"{base_query} WHERE {qualified_pk_col} IN ({pk_values_str});"
    print("DEBUG ▶ Final SQL:", final_sql)
    
    # 8) Execute final query and format answer
    final_tool = get_query_tool()
    final_rows = final_tool.run(final_sql)
    final_str = "\n".join(map(str, final_rows))
    final_answer = format_answer(user_query, final_sql, final_str, filtered_schema)

    return final_answer