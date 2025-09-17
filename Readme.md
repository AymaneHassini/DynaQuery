# DynaQuery: A Self-Adapting Framework for Querying Structured and Multimodal Data

This repository contains the source code and experimental artifacts for the paper:

**"DynaQuery: A Self-Adapting Framework for Querying Structured and Multimodal Data"**

The contents of this artifact support full reproduction of the main experimental results presented in the paper. The contents support full reproduction of the main experimental results for Research Questions 1, 2, and 3.

## Repository Structure

-   `/dynaquery`: Core source code for the DynaQuery framework.
-   `/experiments`: Self-contained Python scripts used to generate and evaluate all experimental results.
-   `/data_samples`: Curated data samples used for evaluation.
-   `download_data.sh`: Script to automatically download external datasets.
-   `requirements.txt`: Python package dependencies required to run the code.
-   `LICENSE`: License for this repository.
-   `/LICENSES`: Licenses for all third-party code and data used in this work.

## 1. Setup Instructions

**Prerequisites:**

-   A Unix-like environment (Linux, macOS, WSL) with `bash`, `wget`/`curl`, `unzip`, and `git`.
-   Python 3.9+
-   `gdown` for downloading from Google Drive (`pip install gdown`).

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

### Step 3. Configure API Keys

Create a `.env` file by copying the provided template.

```bash
cp .env.example .env
```
Then, edit the .env file and add your GOOGLE_API_KEY and database credentials.
### Step 4. Download Benchmark Data

This script will download all external datasets required for the experiments into a new `external_data/` directory. This directory is not part of the repository and will be created by the script. The download size is approximately 550MB.

```bash
bash download_data.sh
```

## 2. Reproducing Paper Results

This section details how to reproduce the key results from the paper.

**IMPORTANT:** All commands must be run from the root directory of the DynaQuery repository.

### 2.1: Spider Evaluation

The following three-stage process reproduces our results on the Spider benchmark, as presented in Tables [2] and [3] of the paper.

#### Stage 1: Direct Schema Linking Performance 

This first experiment evaluates the component-level performance (Precision, Recall, and F1-Score) of our SILE linker against the RAG baseline on a reproducible 500-entry sample.

**Command:**
```bash
python -m experiments.run_rq3_spider_linking
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
python -m experiments.run_rq3_spider_e2e
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
python -m experiments.analyze_hardness \
    --pred_file outputs/spider/predictions_dynaquery-schema-linking.sql \
    --gold_file outputs/spider/dev_gold_sample-schema-linking.sql \
    --db_dir external_data/spider/database/
```
##### **Analyze RAG**
```bash
python -m experiments.analyze_hardness \
    --pred_file outputs/spider/predictions_rag-schema-linking.sql \
    --gold_file outputs/spider/dev_gold_sample-schema-linking.sql \
    --db_dir external_data/spider/database/
```
**Expected Output:**
Each command will run quickly and print a formatted table breaking down the Execution Accuracy by difficulty. The overall EA for DynaQuery should be **80.0%** and for the RAG baseline should be **57.1%**.

### 2.2 BIRD Generalization Evaluation

This four-stage process reproduces our results on the BIRD benchmark, as presented in Tables [4] and [5] of the paper.

#### Stage 1: Generate Prediction Files and Sample Data

This first script is the main entry point for the BIRD experiment. It performs two key tasks:
1. It creates a reproducible, 500-entry stratified random sample of the BIRD development set.
2. It runs the full DynaQuery and RAG pipelines on this sample, generating "sparse" prediction files that are compatible with the official BIRD evaluation tools.

**Command:**
```bash
python -m experiments.run_rq3_bird_e2e
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
python -m experiments.calculate_true_ves
```
**Output:**
- Prints the final, stratified VES breakdown tables for both models.

#### Stage 4: Programmatically Analyze Failures (Table [5])

This final script takes the failure reports generated in Stage 3a and uses `sqlglot` to automatically categorize every error, producing the final analysis for the paper.

**Command:**
```bash
python -m experiments.analyze_rq3_bird_failures
```
**Output:**
- Prints the two failure analysis summary tables (one for DynaQuery, one for RAG).  
- Creates detailed `..._analyzed.json` reports in the `outputs/bird/` directory.


## 3. Licenses and Acknowledgements

The original code for the DynaQuery framework, contained in the `/dynaquery` and `/experiments` directories, is licensed under the **[MIT License](LICENSE)**.

This work builds upon, and our artifact includes derivative works of, several excellent open-source projects. We are grateful to their creators for making their work available.

-   **Spider Benchmark Tooling:** Our evaluation scripts in `experiments/analysis/` are derivative works of the official Spider benchmark tooling. They are used in compliance with the **Apache License 2.0**. The full license text is available in `LICENSES/LICENSE-APACHE-2.0-SPIDER.txt`.

-   **BIRD Benchmark Tooling:** The evaluation scripts in `vendor/bird_eval/` are derivative works of the official BIRD benchmark tooling. They are used in compliance with the **MIT License**. The full license text is available in `LICENSES/LICENSE-BIRD-EVAL.txt`.

-   **Public Datasets:** This project requires several public datasets for evaluation. Please see `LICENSES/DATA_LICENSES.md` for information regarding their respective licenses.