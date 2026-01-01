# experiments/run_rq2_classifier_comparison.py
import csv
import re
import os,json
import argparse
from tqdm import tqdm
from sklearn.metrics import classification_report
from dynaquery.config.settings import LLM_MODEL
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

BENCHMARK_FILE_PATH = "external_data/dynaquery_eval_5k_benchmark/test_split.csv"
LLM_MODEL_NAME = LLM_MODEL

# --- Step 1: Define the Structured Output Schema ---
class ClassificationDecision(BaseModel):
    """A structured object containing the classification decision."""
    explanation: str = Field(description="A single-sentence explanation for the classification decision.")
    label: str = Field(description="The final classification label, must be one of ACCEPT, RECOMMEND, or REJECT.")


def get_llm_native_classifier_chain():
    """Initializes an LLM chain that produces structured, parsable output."""
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, temperature=0)
    
    parser = PydanticOutputParser(pydantic_object=ClassificationDecision)

    prompt_template = """
You are a meticulous classification expert. Your task is to analyze a user's question and a reasoning text to classify the outcome according to a strict rubric.
---
**CLASSIFICATION RUBRIC:**

- **ACCEPT** → ALL conditions from the user's question are satisfied.  
- **RECOMMEND** → SOME (at least one) conditions are satisfied, but not all. Treat this as a "partial match" similar to a recommender system.  
- **REJECT** → NONE of the conditions are satisfied.  

⚠️ Important: If even ONE condition is satisfied, but others fail, you MUST classify as **RECOMMEND**, not REJECT. Only assign REJECT when ZERO conditions match.
---

**User's Question:** "{question}"

**Reasoning Text:**
{reasoning_text}

---
---
Important Note:
For borderline or minimal items that still meet the core functional requirements for casual or basic use, treat them as fulfilling the logical conditions. Only label as REJECT if none of the required functionality is met or the item is clearly non-functional.

--
Based on your analysis, provide your response in the requested format. Do not include any other text before or after the JSON object.
{format_instructions}
    """
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["question", "reasoning_text"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    chain = prompt | llm | parser
    return chain


def run_evaluation(test_run_limit: int = None, debug_mode: bool = False):
    """
    Loads the benchmark data, runs the LLM-native classifier with structured output,
    and reports the final performance metrics.
    """
    print(f"Loading benchmark data from: {BENCHMARK_FILE_PATH}")
    try:
        with open(BENCHMARK_FILE_PATH, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            
            # Robustly handle the header row
            first_line = next(reader)
            if 'text' in first_line and 'labels' in first_line:
                print("Header found and skipped.")
            else:
                print("No header row found. Processing all lines as data.")
                all_lines = list(reader)
                all_lines.insert(0, first_line)
                reader = all_lines
            
            benchmark_data = list(reader)

    except FileNotFoundError:
        print(f"ERROR: Benchmark file not found at '{BENCHMARK_FILE_PATH}'")
        print("Please ensure the manually cleaned file 'rq2_llm_native_benchmark.csv' exists in 'data_samples/'.")
        return

    if test_run_limit:
        print(f"--- !!! RUNNING IN TEST MODE: PROCESSING ONLY {test_run_limit} SAMPLES !!! ---")
        benchmark_data = benchmark_data[:test_run_limit]

    # Initialize the structured output pipeline
    classifier_chain = get_llm_native_classifier_chain()
    
    ground_truths = []
    predictions = []
    error_count = 0
    analysis = []
    print(f"Running LLM-native classification on {len(benchmark_data)} samples...")
    for i, row in enumerate(tqdm(benchmark_data, desc="Evaluating RQ2")):
        if len(row) != 2: 
            continue
        
        text, label_str = row
        
        try:
            # Step 1: Parse the input text from the CSV
            try:
                question_part, reasoning_part = text.split(" Answer: ", 1)
                question = question_part.replace("Question: ", "", 1).strip().strip('"')
                reasoning_text = reasoning_part.strip().strip('"')
            except ValueError:
                question = "N/A"
                reasoning_text = text

            # Step 2: Run the structured output classification chain
            # The chain now takes a dictionary and returns a Pydantic object
            response_obj = classifier_chain.invoke({
                "question": question,
                "reasoning_text": reasoning_text
            })
            
            # Step 4: Convert the label from the parsed object to an integer and store results
            label_str_pred = response_obj.label.upper()
            if label_str_pred == "ACCEPT":
                predicted_label = 0
            elif label_str_pred == "RECOMMEND":
                predicted_label = 1
            else:
                predicted_label = 2
            
            if debug_mode:
                print(f"\n--- DEBUG: Sample #{i+1} ---")
                print(f"Ground Truth Label: {label_str}")
                print(f"Final Predicted Label: {predicted_label}")
                print("--- PARSED LLM Response Object ---")
                print(response_obj)
                print("--------------------------------")
            predictions.append(predicted_label)
            ground_truth_label = int(label_str) # Convert to int once

            ground_truths.append(ground_truth_label)
            if predicted_label != label_str:
                analysis.append({
                    "sample_index": i,
                    "question": question,
                    "reasoning_text": reasoning_text,
                    "ground_truth_label": ground_truth_label,
                    "predicted_label": predicted_label,
                    "llm_explanation": response_obj.explanation
                })
        except Exception as e:
            error_count += 1
            print(f"\nAPI or PARSING error on a sample, skipping. Error: {e}")
            
    # --- Calculate and Print Final Results ---
    print("\n--- RQ2: LLM-Native Classifier Performance ---")
    
    if error_count > 0:
        print(f"WARNING: Skipped {error_count} samples due to API or parsing errors.")
    
    print(f"Evaluating on {len(predictions)} successfully processed samples.")
    
    target_names = ['ACCEPT (Class 0)', 'RECOMMEND (Class 1)', 'REJECT (Class 2)']
    labels_to_report= [0,1,2]
    # Generate the classification report from sklearn
    report = classification_report(
        ground_truths, 
        predictions, 
        labels=labels_to_report, 
        target_names=target_names,
        digits=3,
        zero_division=0
    )
    
    print(report)
    if analysis:
        output_dir = "outputs/rq2"
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, "rq2_analysis_report.json")  # Note: Change filename if testing a different prompt.
        print(f"\Analyzed {len(analysis)} entry. Saving detailed report to: {report_path}")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=4)
    else:
        print("\nProblem occured !")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-run', type=int, nargs='?', const=5, default=None)
    parser.add_argument('--debug', action='store_true', help="Print parsed LLM response objects for debugging.")
    args = parser.parse_args()
    
    run_evaluation(test_run_limit=args.test_run, debug_mode=args.debug)