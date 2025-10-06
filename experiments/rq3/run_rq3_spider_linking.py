# run_rq3_spider_linking.py
"""
FINAL, SELF-CONTAINED evaluation script for Research Question 3 (RQ3).
"""
import json
import random,os
import re
from tqdm import tqdm
from langchain.prompts import ChatPromptTemplate
# --- DYNAMIC FACTORIES AND MODELS (Self-Contained in this script) ---
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.openai_tools import create_extraction_chain_pydantic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.pydantic_v1 import BaseModel, Field
from dynaquery.config.settings import LLM_MODEL

class QueryPlan(BaseModel):
    """A plan for executing a query, including the base table and any tables to join."""
    base_table: str = Field(description="The single table that contains the primary entity the user is asking for.")
    join_tables: list[str] = Field(description="A list of any other tables that need to be joined to the base table to answer the query.")

print("Initializing LLM and Embedding Model...")
llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/llm-embedder")
print("Models initialized.")

def create_dynamic_sile_chain(schema_str: str):
    """
    Creates a SILE chain dynamically for a specific database schema using the
    modern .with_structured_output() method.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are an expert at creating query plans.
Your goal is to return a query plan to answer the user's question based on the provided schema.
The plan MUST identify the single `base_table` that contains the primary entity the user is asking about, and a list of `join_tables` needed for filtering or additional information.
Example: For "Find specs for Apple laptops", the plan is: base_table="specifications", join_tables=["products"].

The available tables are:
{schema_str}
"""),
        ("human", "{input}")
    ])
    
    structured_llm = llm.with_structured_output(QueryPlan)
    return prompt | structured_llm

def create_dynamic_rag_retriever(schema_str: str):
    """Builds a RAG retriever for a specific database schema string."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
    schema_chunks = text_splitter.split_text(schema_str)
    vector_store = Chroma.from_texts(texts=schema_chunks, embedding=embedding_model)
    return vector_store.as_retriever(search_kwargs={"k": 4})

# --- DATASET CONFIGURATION AND PARSING ---
FULL_DATASET_PATH = "external_data/spider-schema-linking-dataset-main/data/schema-linking/dev.jsonl"
SAMPLE_PATH = "outputs/spider/spider_linking_sample_500.jsonl"
SAMPLE_SIZE = 500

def parse_entry_with_schema(entry: dict) -> dict:
    """
    Parses a single JSON object to extract the question, ground-truth tables,
    and the formatted schema string for that specific entry.
    """
    question = entry["text"]
    meta = entry["meta"]
    labels = entry["labels"]
    
    ground_truth_tables = set()
    table_id_to_name = {key: value for key, value in meta.items() if key.startswith("001_table")}
            
    for label in labels:
        entity_id = label[2]
        if entity_id in table_id_to_name:
            ground_truth_tables.add(table_id_to_name[entity_id])

    schema_str = ""
    table_columns = {name: [] for name in table_id_to_name.values()}
    for key, value in meta.items():
        if key.startswith("002_col"):
            try:
                table_name, col_name = value.split('.', 1)
                if table_name in table_columns:
                    table_columns[table_name].append(col_name)
            except ValueError:
                continue
    
    for table_name in sorted(table_columns.keys()):
        schema_str += f"Table Name: {table_name}\nColumns:\n"
        for col_name in table_columns[table_name]:
            schema_str += f" - {col_name}\n"
        schema_str += "\n"
            
    return {
        "question": question,
        "tables": list(ground_truth_tables),
        "schema": schema_str.strip()
    }


def create_dataset_sample():
    """Creates a reproducible random sample from the full dataset."""
    print(f"Creating a random sample of {SAMPLE_SIZE} from {FULL_DATASET_PATH}...")
    try:
        with open(FULL_DATASET_PATH, 'r', encoding='utf-8') as f:
            full_data = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"ERROR: The full dataset was not found at '{FULL_DATASET_PATH}'")
        print("Please run download_data.sh at root of the repository")
        return False

    random.seed(42) # Fixed seed for reproducibility
    sample_data = random.sample(full_data, SAMPLE_SIZE)
    output_dir = os.path.dirname(SAMPLE_PATH)
    os.makedirs(output_dir, exist_ok=True)
    with open(SAMPLE_PATH, 'w', encoding='utf-8') as f:
        for item in sample_data:
            f.write(json.dumps(item) + '\n')

    print(f"Successfully created sample file at {SAMPLE_PATH}")
    return True

def calculate_metrics(predicted: list[str], ground_truth: list[str]) -> dict:
    """Calculates Precision, Recall, and F1 Score."""
    pred_set = set(predicted)
    gt_set = set(ground_truth)
    
    if not gt_set and not pred_set: return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not gt_set: return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    tp = len(pred_set.intersection(gt_set))
    precision = tp / len(pred_set) if len(pred_set) > 0 else 0.0
    recall = tp / len(gt_set) if len(gt_set) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {"precision": precision, "recall": recall, "f1": f1}

# --- MAIN EXPERIMENT SCRIPT ---

def run_schema_linking_experiment():
    """Runs the full, dynamic head-to-head comparison."""
    
    try:
        with open(SAMPLE_PATH, 'r', encoding='utf-8') as f:
            dataset_sample = [json.loads(line) for line in f]
    except FileNotFoundError:
        if not create_dataset_sample(): return
        with open(SAMPLE_PATH, 'r', encoding='utf-8') as f:
            dataset_sample = [json.loads(line) for line in f]

    sile_results = []
    rag_results = []

    print(f"\nRunning DYNAMIC evaluation on {len(dataset_sample)} samples...")
    for entry in tqdm(dataset_sample, desc="Evaluating Schema Linking"):
        parsed_data = parse_entry_with_schema(entry)
        question = parsed_data["question"]
        ground_truth_tables = parsed_data["tables"]
        current_schema = parsed_data["schema"]
        
        # 1. Evaluate SILE (Our Method)
        sile_chain = create_dynamic_sile_chain(current_schema)
        try:
            sile_plan = sile_chain.invoke({"input": question})
            sile_predicted_tables = list(set([sile_plan.base_table] + sile_plan.join_tables))
        except Exception as e:
            print(f"Warning: SILE chain failed for a sample with error: {e}")
            sile_predicted_tables = []
        sile_results.append(calculate_metrics(sile_predicted_tables, ground_truth_tables))
        
        # 2. Evaluate RAG Baseline
        rag_retriever = create_dynamic_rag_retriever(current_schema)
        docs = rag_retriever.invoke(question)
        retrieved_context = "\n\n".join(doc.page_content for doc in docs)
        rag_predicted_tables = list(set(re.findall(r"Table Name: (\w+)", retrieved_context)))
        rag_results.append(calculate_metrics(rag_predicted_tables, ground_truth_tables))

    # 3. Aggregate and report results
    def aggregate_scores(results: list[dict]) -> dict:
        """Averages the precision, recall, and f1 scores from a list of results."""
        if not results:
            return {"Precision": 0.0, "Recall": 0.0, "F1-Score": 0.0}
        
        avg_p = sum(r['precision'] for r in results) / len(results)
        avg_r = sum(r['recall'] for r in results) / len(results)
        avg_f1 = sum(r['f1'] for r in results) / len(results)
        return {"Precision": avg_p, "Recall": avg_r, "F1-Score": avg_f1}

    sile_scores = aggregate_scores(sile_results)
    rag_scores = aggregate_scores(rag_results)

    print("\n--- Schema Linking Performance ---")
    print(f"Method                  | Precision | Recall    | F1-Score")
    print("------------------------|-----------|-----------|-----------")
    print(f"RAG Baseline            | {rag_scores['Precision']:.4f}    | {rag_scores['Recall']:.4f}   | {rag_scores['F1-Score']:.4f}")
    print(f"DynaQuery (SILE)        | {sile_scores['Precision']:.4f}    | {sile_scores['Recall']:.4f}   | {sile_scores['F1-Score']:.4f}")
    print("----------------------------------------------------")

if __name__ == "__main__":
    run_schema_linking_experiment()