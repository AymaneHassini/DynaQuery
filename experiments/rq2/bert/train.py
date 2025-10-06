import torch
import pandas as pd
import os
import re
import argparse
from transformers import BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def get_parser():
    """Defines command-line arguments for the training script."""
    parser = argparse.ArgumentParser(description="Train a BERT model for sequence classification.")
    
    parser.add_argument("--train_file", type=str, default="train_split.csv", help="Path to the training data CSV file.")
    parser.add_argument("--test_file", type=str, default="test_split.csv", help="Path to the test data CSV file.")
    parser.add_argument("--output_dir", type=str, default="bert_finetuned_checkpoint", help="Directory to save the final model checkpoint.")
    
    # --- Model and Tokenizer ---
    parser.add_argument("--checkpoint", type=str, default="bert-base-cased", help="Hugging Face model checkpoint.")
    
    # --- Default Training Hyperparameters ---
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Initial learning rate for the AdamW optimizer.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for the AdamW optimizer.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per device during training.")
    
    return parser


def compute_metrics(pred):
    """
    Computes macro-averaged and per-class accuracy, F1, precision, and recall.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0
    )
    
    per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
        labels, preds, average=None, labels=[0, 1, 2], zero_division=0
    )
    
    acc = accuracy_score(labels, preds)
    
    metrics = {
        'Accuracy': acc,
        'F1_macro': macro_f1,
        'Precision_macro': macro_precision,
        'Recall_macro': macro_recall,
        'F1_class_0': per_class_f1[0], 'Precision_class_0': per_class_precision[0], 'Recall_class_0': per_class_recall[0],
        'F1_class_1': per_class_f1[1], 'Precision_class_1': per_class_precision[1], 'Recall_class_1': per_class_recall[1],
        'F1_class_2': per_class_f1[2], 'Precision_class_2': per_class_precision[2], 'Recall_class_2': per_class_recall[2],
    }
    
    return metrics

class CustomDataset(Dataset):
    """Custom Dataset class for tokenized text and labels."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]).clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels.iloc[idx]).clone().detach()
        return item

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":

    # Device selection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Parse arguments
    args = get_parser().parse_args()
    checkpoint = args.checkpoint
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    output_dir = args.output_dir

    # Load pre-split data
    print(f"Loading training data from {args.train_file}")
    train_df = pd.read_csv(args.train_file)
    print(f"Loading test data from {args.test_file}")
    test_df = pd.read_csv(args.test_file)
    
     # Preprocess text
    def clean_text(text):
        text = str(text).replace('\n', ' ') # Replace newlines with spaces
        text = re.sub(r'\s+', ' ', text)    # Collapse multiple whitespace chars into one
        return text.strip()                # Remove leading/trailing whitespace

    train_df['text'] = train_df['text'].apply(clean_text)
    test_df['text'] = test_df['text'].apply(clean_text)


    # Assign texts and labels from the dataframes
    train_texts, test_texts = train_df['text'], test_df['text']
    train_labels, test_labels = train_df['labels'], test_df['labels']
    print(f"Train size: {len(train_texts)} | Test size: {len(test_texts)}")

    labels = sorted(train_df['labels'].unique())
    id2label = {int(i): str(label) for i, label in enumerate(labels)}
    label2id = {str(label): int(i) for i, label in enumerate(labels)}

    # Tokenize data
    tokenizer = BertTokenizerFast.from_pretrained(checkpoint, max_length=512)
    train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, return_tensors="pt")
    test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, return_tensors="pt")

    train_dataset = CustomDataset(train_encodings, train_labels)
    test_dataset = CustomDataset(test_encodings, test_labels)

    # Load model
    model = BertForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    ).to(device)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        seed=42,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=weight_decay,        
        eval_strategy="epoch", 
        save_strategy="epoch",
        logging_strategy="epoch",    
        load_best_model_at_end=True,
        metric_for_best_model="F1_macro",
        greater_is_better=True,
        report_to=[]
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset, # Use the test set for evaluation during training
        compute_metrics=compute_metrics
    )

    # Train
    print("\n--- Starting Training ---")
    trainer.train()
    print("\n--- Final Evaluation on the Held-Out Test Set ---")
    final_metrics = trainer.evaluate(eval_dataset=test_dataset)
    
    print("\nDefinitive Test Metrics for the Best Model :")
    # Pretty print the final results
    for key, value in final_metrics.items():
        if "eval_" in key:
            metric_name = key.replace("eval_", "")
            print(f"{metric_name:<20}: {value:.4f}")

    # Save the best model
    best_model_dir = os.path.join(output_dir, "best-model")
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    print(f"\nBest model and tokenizer saved at {best_model_dir}")