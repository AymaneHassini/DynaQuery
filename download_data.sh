#!/bin/bash
# Download all external data for DynaQuery using gdown (macOS friendly)

set -e

echo "--- Creating directory for external data ---"
mkdir -p external_data

# --- Step 1: Download Official Spider DATA (Google Drive) ---
SPIDER_DATA_ZIP="external_data/spider.zip"
SPIDER_DATA_DIR="external_data/spider"

if [ ! -d "$SPIDER_DATA_DIR" ]; then
    echo "--> Downloading Spider dataset from Google Drive (~200 MB)..."
    FILE_ID="1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"

    gdown "https://drive.google.com/uc?id=${FILE_ID}" -O "${SPIDER_DATA_ZIP}"

    echo "    --> Extracting Spider dataset..."
    unzip -q "${SPIDER_DATA_ZIP}" -d external_data/

    # --- Rename folder if needed ---
    if [ -d "external_data/spider_data" ]; then
        mv external_data/spider_data external_data/spider
    fi

    rm "${SPIDER_DATA_ZIP}"
    echo "    Spider dataset is ready in ${SPIDER_DATA_DIR}."
else
    echo "--> Spider data directory already exists. Skipping download."
fi

# --- Step 2: Download the Spider Schema Linking Dataset ---
LINKING_ZIP="external_data/spider-linking.zip"
LINKING_DIR="external_data/spider-schema-linking-dataset-main"

if [ ! -d "$LINKING_DIR" ]; then
    echo "--> Downloading Spider Schema Linking dataset from GitHub..."
    curl -L -C - "https://github.com/yasufumy/spider-schema-linking-dataset/archive/refs/heads/main.zip" -o "$LINKING_ZIP"

    echo "    --> Extracting the linking dataset..."
    unzip -q "$LINKING_ZIP" -d external_data/
    rm "$LINKING_ZIP"
else
    echo "--> Spider Schema Linking data directory already exists. Skipping download."
fi
# --- Step 3: Download and Correctly Unpack BIRD Development DATA ---
BIRD_DATA_DIR="external_data/bird"
if [ ! -d "$BIRD_DATA_DIR" ]; then
    echo "--> Downloading BIRD development dataset from official source (~330MB)..."
    BIRD_DATA_ZIP="external_data/bird_dev.zip"
    
    curl -L "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip" -o "$BIRD_DATA_ZIP"
    
    echo "    --> Verifying BIRD download..."
    if ! unzip -t "$BIRD_DATA_ZIP" > /dev/null; then
        echo "ERROR: Failed to download a valid BIRD dev.zip file."
        exit 1
    fi
    
    echo "    --> Extracting BIRD dataset..."
    unzip -q "$BIRD_DATA_ZIP" -d external_data/
    
    # The zip file creates a directory with a versioned name, e.g., 'dev_20240627'.
    # We will find this directory and rename it to a standard name, 'bird'.
    # This makes the script robust to future changes in the zip file's internal structure.
    EXTRACTED_BIRD_DIR=$(unzip -Z1 "$BIRD_DATA_ZIP" | head -n1 | cut -d'/' -f1)
    if [ -d "external_data/${EXTRACTED_BIRD_DIR}" ]; then
        mv "external_data/${EXTRACTED_BIRD_DIR}" "$BIRD_DATA_DIR"
    else
        echo "ERROR: Could not find the extracted BIRD directory."
        exit 1
    fi

    # --- THE FINAL, AUTOMATED FIX IS HERE ---
    # Unzip the nested databases
    BIRD_DB_ZIP="${BIRD_DATA_DIR}/dev_databases.zip"
    if [ -f "$BIRD_DB_ZIP" ]; then
        echo "    --> Extracting nested BIRD databases..."
        unzip -q "$BIRD_DB_ZIP" -d "$BIRD_DATA_DIR/"
        rm "$BIRD_DB_ZIP"
    fi
    
    # Automatically rename dev.sql to dev_gold.sql to match the script's expectation
    BIRD_GOLD_SQL_ORIGINAL="${BIRD_DATA_DIR}/dev.sql"
    BIRD_GOLD_SQL_TARGET="${BIRD_DATA_DIR}/dev_gold.sql"
    if [ -f "$BIRD_GOLD_SQL_ORIGINAL" ]; then
        echo "    --> Renaming BIRD gold file for compatibility..."
        mv "$BIRD_GOLD_SQL_ORIGINAL" "$BIRD_GOLD_SQL_TARGET"
    fi
    # -----------------------------------------------------
    
    rm "$BIRD_DATA_ZIP"
    echo "    BIRD dataset is now ready in ${BIRD_DATA_DIR}."
else
    echo "--> BIRD data directory already exists. Skipping download."
fi

echo ""
echo "--- All external data has been successfully downloaded and set up. ---"
