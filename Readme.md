# DynaQuery: A Self-Adapting Framework for Querying Structured and Multimodal Data

This repository contains the source code and experimental artifacts for the paper:

**"DynaQuery: A Self-Adapting Framework for Querying Structured and Multimodal Data"**

The contents of this artifact support full reproduction of the main experimental results presented in the paper. The contents support full reproduction of the main experimental results for Research Questions 1, 2, and 3.

## Repository Structure

-   `/dynaquery`: Core source code for the DynaQuery framework.
-   `/experiments`: Self-contained Python scripts used to generate and evaluate all experimental results.
-   `/setup`: Files required for the initial experimental setup (e.g., database dump).
-   `/artifacts`: Static, verifiable artifacts (e.g., execution logs, ground truth) that support the qualitative analyses in the paper.
-   `download_data.sh`: Script to automatically download external datasets.
-   `requirements.txt`: Python package dependencies required to run the code.
-   `LICENSE`: License for this repository.
-   `/LICENSES`: Licenses for all third-party code and data used in this work.

## 1. Setup Instructions

**Prerequisites:**

-   A Unix-like environment (Linux, macOS, WSL) with `bash`, `wget`/`curl`, `unzip`, and `git`.
-   Python 3.9+
-   `gdown` for downloading from Google Drive (`pip install gdown`).
-   A Kaggle account and the `kaggle` CLI package.
-   A running MySQL server instance (required for the RQ2 case study and live demo).

---

### Step 1. Clone the Repository

```bash
git clone https://github.com/aymanehassini/DynaQuery.git
cd DynaQuery
```

### Step 2. Create a Python Environment and Install Dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
### Step 3. Configure API Keys and Database Credentials

Create a `.env` file by copying the provided template.

```bash
cp .env.example .env
```
Then, edit the .env file and add your GOOGLE_API_KEY and database credentials.

### Kaggle API Key (Required for Dataset Download)

Our primary training dataset is hosted on **Kaggle**. The automated download script requires the Kaggle API to be configured.

1. Log in to your [Kaggle account](https://www.kaggle.com/settings).  
2. Go to your **Account Settings** page and click **"Create New API Token"**.  
   This will download a `kaggle.json` file.  
3. Place this file in a `.kaggle` directory in your home folder:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```
### Step 4. Download Pre-trained Model Checkpoint

Our experiments rely on a fine-tuned BERT model. For convenience and reproducibility, we provide the exact model checkpoint used in our paper.

1- Navigate to the **Releases** page of this GitHub repository and download the `bert-checkpoint-rq2.zip` file.  
2- Unzip the file to a stable location on your machine.
3- Open `dynaquery/config/settings.py` and update the `CHECKPOINT_PATH` variable to point to the location of the unzipped directory.  
We strongly recommend using an **absolute path**.

### Step 5. Download Benchmark Data

This script will download all large external datasets (Spider, BIRD, and our Annotated Rationale Dataset) into a new  `external_data/` directory.

```bash
bash download_data.sh
```

## 2. Reproducing Paper Results

This section details how to reproduce the key results from the paper.

**IMPORTANT:** All commands must be run from the root directory of the DynaQuery repository.

### 2.1: RQ1: Spider and BIRD Evaluation

### 2.1.1: Spider Evaluation

The following three-stage process reproduces our results on the Spider benchmark, as presented in Tables [2] and [3] of the paper.

#### Stage 1: Direct Schema Linking Performance 

This first experiment evaluates the component-level performance (Precision, Recall, and F1-Score) of our SILE linker against the RAG baseline on a reproducible 500-entry sample.

**Command:**
```bash
python -m experiments.rq1.run_rq3_spider_linking
```
## What this script does
**Step 1 (Data Preparation):**  
The script first checks for the existence of the full Spider Schema Linking dataset in `external_data/`. If found, it programmatically creates a reproducible 500-entry random sample and saves it to `outputs/spider/spider_linking_sample_500.jsonl`.
**Step 2 (Evaluation):**  
It then iterates through this newly created 500-entry sample, running both the DynaQuery (SILE) and RAG Baseline linkers for each question.
**Step 3 (Reporting):**  
Finally, it aggregates the results and prints the final performance scores to the console.
**Expected Output:**
The script will run for approximately 45 minutes, depending on API latency. Upon completion, it will print a formatted table to the console with the final scores. The F1-Score for DynaQuery (SILE) should be approximately **0.77** for DynaQuery (SILE) and **0.34** for the RAG Baseline.

#### Stage 2: End-to-End Prediction Generation

This script uses the sample file created in Stage 1 to run the full DynaQuery and RAG pipelines, generating the SQL queries needed for the final evaluation.

**Command:**
```bash
python -m experiments.rq1.run_rq1_spider_e2e
```
**Expected Output:**
The script will run for a significant amount of time (approx. 2-3 hours). It will create three new files in the `outputs/spider/` directory:
- `predictions_dynaquery-schema-linking.sql`
- `predictions_rag-schema-linking.sql`
- `dev_gold_sample-schema-linking.sql`

#### Stage 3: Execution Accuracy and Hardness Analysis 

This final stage uses our robust, self-contained analysis script to calculate the end-to-end Execution Accuracy and break down the results by query difficulty. This script uses our direct-execution protocol and a heuristic-based classifier modeled on the official Spider hardness rules.


##### **Analyze DynaQuery (SILE)**

```bash
python -m experiments.rq1.analyze_hardness \
    --pred_file outputs/spider/predictions_dynaquery-schema-linking.sql \
    --gold_file outputs/spider/dev_gold_sample-schema-linking.sql \
    --db_dir external_data/spider/database/
```
##### **Analyze RAG**
```bash
python -m experiments.rq1.analyze_hardness \
    --pred_file outputs/spider/predictions_rag-schema-linking.sql \
    --gold_file outputs/spider/dev_gold_sample-schema-linking.sql \
    --db_dir external_data/spider/database/
```
**Expected Output:**
Each command will run quickly and print a formatted table breaking down the Execution Accuracy by difficulty. The overall EA for DynaQuery should be **80.0%** and for the RAG baseline should be **57.1%**.

### 2.3.3 BIRD Generalization Evaluation

This four-stage process reproduces our results on the BIRD benchmark, as presented in Tables [4] and [5] of the paper.

#### Stage 1: Generate Prediction Files and Sample Data

This first script is the main entry point for the BIRD experiment. It performs two key tasks:
1. It creates a reproducible, 500-entry stratified random sample of the BIRD development set.
2. It runs the full DynaQuery and RAG pipelines on this sample, generating "sparse" prediction files that are compatible with the official BIRD evaluation tools.

**Command:**
```bash
python -m experiments.rq1.run_rq3_bird_e2e
```
**Output:**
- `data_samples/bird_dev_sample_500.json (The stratified sample file)`
- `outputs/bird_dynaquery/predict_dev.json (DynaQuery's (SILE) predictions)`
- `outputs/bird_rag/predict_dev.json (RAG's predictions)`
#### Stage 2: Generate Raw Performance Reports

This stage uses our modified versions of the official BIRD evaluation scripts to execute all queries and produce raw, per-query JSON reports for both Execution Accuracy (EA) and Valid Efficiency Score (VES).

**Commands for DynaQuery:**

- **Get Raw EA Report**
```bash
python -m vendor.evaluation \
    --predicted_sql_path outputs/bird_dynaquery/ \
    --ground_truth_path external_data/bird/ \
    --data_mode dev \
    --db_root_path external_data/bird/dev_databases/ \
    --diff_json_path external_data/bird/dev.json \
    --output_path outputs/bird/dynaquery_bird_results_ea.json
```
- **Get Raw VES Report**
```bash
python -m vendor.evaluation_ves \
    --predicted_sql_path outputs/bird_dynaquery/ \
    --ground_truth_path external_data/bird/ \
    --data_mode dev \
    --db_root_path external_data/bird/dev_databases/ \
    --diff_json_path external_data/bird/dev.json \
    --output_path outputs/bird/dynaquery_bird_results_ves.json
```
**Commands for RAG Baseline:**

- **Get Raw EA Report**
```bash
python -m vendor.evaluation \
    --predicted_sql_path outputs/bird_rag/ \
    --ground_truth_path external_data/bird/ \
    --data_mode dev \
    --db_root_path external_data/bird/dev_databases/ \
    --diff_json_path external_data/bird/dev.json \
    --output_path outputs/bird/rag_bird_results_ea.json
```
- **Get Raw VES Report**
```bash
python -m vendor.evaluation_ves \
    --predicted_sql_path outputs/bird_rag/ \
    --ground_truth_path external_data/bird/ \
    --data_mode dev \
    --db_root_path external_data/bird/dev_databases/ \
    --diff_json_path external_data/bird/dev.json \
    --output_path outputs/bird/rag_bird_results_ves.json
```
**Output:**
- `outputs/bird/dynaquery_bird_results_ea.json`
- `outputs/bird/dynaquery_bird_results_ves.json`
- `outputs/bird/dynaquery_rag_results_ea.json`
- `outputs/bird/rag_bird_results_ves.json`
#### Stage 3: Post-Process Reports to Calculate Final Scores

This stage uses our custom analysis scripts to parse the raw reports from Stage 2 and calculate the true, final scores for our 500-item sample.

**Step 3a: Calculate Final Execution Accuracy (Table [6]) and Generate Failure Reports**
```bash
python -m experiments.rq1.calculate_true_ves
```
**Output:**
- Prints the final, stratified VES breakdown tables for both models.

#### Stage 4: Programmatically Analyze Failures (Table [5])

This final script takes the failure reports generated in Stage 3a and uses `sqlglot` to automatically categorize every error, producing the final analysis for the paper.

**Command:**
```bash
python -m experiments.rq3.analyze_rq1_bird_failures
```
**Output:**
- Prints the two failure analysis summary tables (one for DynaQuery, one for RAG).  
- Creates detailed `..._analyzed.json` reports in the `outputs/bird/` directory.

## 2.2 RQ2: Classifier Trade-off Analysis 

### 2.2.1 Setup for the OOD Case Study

The OOD case study requires a small, persistent **MySQL database**.

#### Create the Database
First, log in to your MySQL server and create a new database:

```sql
-- (In your MySQL client)
CREATE DATABASE dynaquery_ood;
```
#### Load the Data

From your terminal (not the MySQL client), run the following command to load the schema and data.  
Replace `[username]` with your MySQL username. You will be prompted for your password.

```bash
mysql -u [username] -p dynaquery_ood < setup/ood_database.sql
```
**Update Credentials:**  
Ensure your `.env` file is configured with the correct credentials to access the `dynaquery_ood` database.

### 2.2.2 Reproducing the In-Distribution (IID) Results (Table 2)

This experiment evaluates the three classifier architectures on our **1,000-sample IID test set**.  
The required `train_split.csv` (4,000 samples) and `test_split.csv` (1,000 samples) files are included in the **DynaQuery-Eval-5K** benchmark, which is downloaded into the `external_data/` directory during setup.

#### 1. Evaluate the Fine-Tuned BERT Specialist

This command re-trains the **BERT model** from scratch using the provided training split and evaluates it on the test split.  
The command includes the exact hyperparameters we used in training.

```bash
python -m experiments.rq2.bert.train \
    --train_file external_data/dynaquery_eval_5k_benchmark/train_split.csv \
    --test_file external_data/dynaquery_eval_5k_benchmark/test_split.csv \
    --epochs 4 \
    --learning_rate 2e-5 \
    --batch_size 32 \
    --output_dir outputs/rq2/bert_retrained_checkpoint
```
**Expected Output:**  
The script will train the model and then print a final evaluation report.  
The `eval_F1_macro` score should be approximately **0.991**.

#### 2. Evaluate the LLM Generalists

**LLM (Rule-Based Prompt):**  
This is the default behavior of the script and represents our final, recommended architecture.
```bash
python -m experiments.rq2.run_rq2_classifier_comparison 
```
**Expected Output:**  
The macro F1-score should be approximately **78.0%**.
**(Optional) LLM (Descriptive Prompt):**  
To reproduce the score for the descriptive prompt, you must manually edit the `dynaquery/chains/llm_classifier.py` file.  
Inside the `get_llm_native_classifier_chain` function, perform the following two steps:

1. Comment out the active **"Rule-Based Prompt"** string.  
2. Uncomment the inactive **"Descriptive Prompt"** string provided in the comments.

After saving the change, run the evaluation command again from from the project root directory:

```bash
python -m experiments.rq2.run_rq2_classifier_comparison 
```
**Expected Output:**  
With the descriptive prompt active, the macro `F1_macro` should be approximately 94.7%.

**Artifact Note:**  
For full transparency, we also provide the script used to generate the data splits at `experiments/rq2/bert/split.py`.  
Running this script is **not necessary** to reproduce the results, as the output files are already provided in the Kaggle download.

### 2.2.3 Verifying the Out-of-Distribution (OOD) Case Study (Table 3 & Figure 2)

The analysis in the paper is based on static, verifiable execution logs from our **OOD experiment**, which are provided as research artifacts.
- **Verifiable Execution Logs:** The complete console output for each pipeline is located in:
  - `artifacts/rq2/bert_run_log.txt`
  - `artifacts/rq2/llm_descriptive_run_log.txt`
  - `artifacts/rq2/llm_rule_based_run_log.txt`

- **Human-Annotated Ground Truth:** The expected correct classification and justification for each query is provided in the subdirectory:
  - `artifacts/rq2/ground_truth/`

To verify our claims, inspect these execution logs. For instance, the BERT failure in Figure 2(a) can be confirmed by examining the `"Question 2"` block in `artifacts/rq2/bert_run_log.txt`.
## 2.3: RQ3: End-to-End System Generalization
This final evaluation tests the **end-to-end generalization** of the full **DynaQuery framework** on our novel **Olist Multimodal Benchmark**.
### 2.3.1 Setup for the Olist Benchmark

The Olist benchmark requires a dedicated **MySQL database**.

1. **Create the Database:**
   ```sql
   CREATE DATABASE dynaquery_olist;
    ```
2. **Load the Data:**
   ```bash
   mysql -u [username] -p dynaquery_olist < setup/olist_multimodal_benchmark.sql
    ```
3. **Update Credentials:**
   Ensure your `.env` file is configured to access the `dynaquery_olist` database.
### 2.3.2 Running the Evaluation

The evaluation consists of running our two main pipelines against their respective benchmark query suites (provided in `artifacts/rq3/benchmark_suites/`).  
The experiment is run in two modes: a baseline **"schema-aware"** mode and a **"semantics-aware"** mode, controlled by the presence of the `dynaquery/config/schema_comments.json` file.
1. **Run the Baseline (Schema-Aware) Evaluation:**  
First, ensure the schema comments file does **not exist** by temporarily renaming it:

```bash
mv dynaquery/config/schema_comments.json dynaquery/config/schema_comments.json.bak
```
Now, run the benchmark scripts:

```bash
# Run SQP Baseline
python -m experiments.rq3.run_sqp_benchmark \
    --benchmark_file artifacts/rq3/benchmark_suites/sqp_benchmark_suite.csv \
    --results_file outputs/rq3/sqp_baseline_results.csv

# Run MMP Baseline
python -m experiments.rq3.run_mmp_benchmark \
    --benchmark_file artifacts/rq3/benchmark_suites/mmp_benchmark_suite.csv \
    --results_file outputs/rq3/mmp_baseline_results.csv
```
**Expected Output:**  
- The overall **Execution Accuracy (SQP)** should be approximately **65.0%**.  
- The **F1-Score (MMP)** should be approximately **54.0%**.
2. **Run the Enriched (Semantics-Aware) Evaluation:**  
Restore the schema comments file to enable semantics-awareness:

```bash
mv dynaquery/config/schema_comments.json.bak dynaquery/config/schema_comments.json
```
Now, run the benchmark scripts again:

```bash
# Run SQP Enriched
python -m experiments.rq3.run_sqp_benchmark \
    --benchmark_file artifacts/rq3/benchmark_suites/sqp_benchmark_suite.csv \
    --results_file outputs/rq3/sqp_enriched_results.csv

# Run MMP Enriched
python -m experiments.rq3.run_mmp_benchmark \
    --benchmark_file artifacts/rq3/benchmark_suites/mmp_benchmark_suite.csv \
    --results_file outputs/rq3/mmp_enriched_results.csv
```
**Expected Output:**  
- The overall **Execution Accuracy (SQP)** should be approximately **95.0%**.  
- The **F1-Score (MMP)** should be approximately **93.0%**.
### 2.3.3 Verifying the Results

The summary results printed to the console by the scripts directly correspond to **Tables 9 and 10** in the paper.  
For a more detailed, row-by-row verification of our findings, we provide the complete outputs from our original experimental run as **verifiable artifacts**.

These are located in the `artifacts/rq3/verifiable_logs/` directory and include:

- **Detailed Result CSVs** (e.g., `results-sqp-enriched.csv`):  
  These files contain the generated SQL for each query (for SQP) or the final set of accepted IDs (for MMP), allowing for a direct audit of our accuracy calculations.

- **Verbose Log Files** (e.g., `output-sqp-enriched.log`):  
  These files contain the complete console trace for each run, showing the step-by-step reasoning of the LLM components.


---
## 3. Licenses and Acknowledgements

The original code for the DynaQuery framework is licensed under the **[MIT License](LICENSE)**.

This work builds upon, includes derivative works of, and uses data from several excellent open-source projects. We are grateful to their creators for making their work available.

-   **Spider Benchmark Tooling:** Our evaluation scripts in `/experiments/analysis/` are derivative works of the official Spider benchmark tooling and are used in compliance with the **Apache License 2.0**. The full license text is available in `LICENSES/LICENSE-SPIDER-EVAL.txt`.

-   **BIRD Benchmark Tooling:** The evaluation scripts in `vendor/bird_eval/` are derivative works of the official BIRD benchmark tooling and are used in compliance with the **MIT License**. The full license text is available in `LICENSES/LICENSE-BIRD-EVAL.txt`.

-   **Public Datasets:** This project requires several public datasets for evaluation. Please see `LICENSES/DATA_LICENSES.md` for information regarding their respective licenses.