# models/classifier.py
"""
BERT classifier models initialization and configuration.
"""
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertForSequenceClassification,
    BertTokenizerFast,
    Trainer,
    TrainingArguments
)
from dynaquery.config.settings import CHECKPOINT_PATH

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Common training arguments
training_args = TrainingArguments(
    "tmp",
    disable_tqdm=True,
    run_name='advanced-reasoning',
)

def load_classifier_distilbert():
    """
    Load and configure a DistilBERT classifier.
    Returns:
        tuple: (trainer, tokenizer) for the model
    """
    print(f"Using device: {device}")
    # Prepare the model
    classifier = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT_PATH,
        num_labels=3
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)
    trainer = Trainer(
        classifier,
        args=training_args,
    )
    return trainer, tokenizer

def load_classifier_bert():
    """
    Load and configure a BERT classifier.
    Returns:
        tuple: (trainer, tokenizer) for the model
    """
    # Prepare the model
    classifier = BertForSequenceClassification.from_pretrained(
        CHECKPOINT_PATH,
        num_labels=3,
    )
    tokenizer = BertTokenizerFast.from_pretrained(CHECKPOINT_PATH, max_length=512)
    trainer = Trainer(
        classifier,
        args=training_args,
    )
    return trainer, tokenizer