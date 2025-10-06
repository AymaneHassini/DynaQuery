# split_data.py
import pandas as pd
from sklearn.model_selection import train_test_split

print("Splitting the dataset into reproducible train/test files...")

# Load the full dataset
df = pd.read_csv("dataset.csv")  # DATASET_PATH = "external_data/dynaquery_eval_5k_benchmark/dataset.csv" - this is the path when successfully downloading dataset using our script.


train_df, test_df = train_test_split(
    df, 
    test_size=0.2, 
    random_state=42, # Use a fixed random state for reproducibility
    stratify=df['labels']
)

# Save the splits to new CSV files 
train_df.to_csv("train_split.csv", index=False)
test_df.to_csv("test_split.csv", index=False)

print("Data split successfully!")
print(f"Training set size: {len(train_df)}")
print(f"Test set size: {len(test_df)}")
print("\nTraining set distribution:\n", train_df['labels'].value_counts(normalize=True))
print("\nTest set distribution:\n", test_df['labels'].value_counts(normalize=True))