# preprocessing/text.py
"""
Text preprocessing utilities for tokenization.
"""
from datasets import Dataset

def tokenize(text, tokenizer):
    """
    Tokenize text using the provided tokenizer.
    
    Args:
        text: Text to tokenize
        tokenizer: Tokenizer instance
        
    Returns:
        Dataset: A HuggingFace Dataset containing tokenized inputs
    """
    # Tokenize the input text
    dataset = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )
    
    # Convert to Dataset format
    return Dataset.from_dict({
        "input_ids": dataset["input_ids"], 
        "attention_mask": dataset["attention_mask"]
    })