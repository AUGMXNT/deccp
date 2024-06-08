from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
import os

def read_text_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        texts = f.readlines()
    return texts

# Read the text files
censored_texts = read_text_file('data/censored.txt')
uncensored_texts = read_text_file('data/uncensored.txt')

# Create the dataset
censored_dataset = Dataset.from_dict({"text": censored_texts})
uncensored_dataset = Dataset.from_dict({"text": uncensored_texts})

# Combine into a DatasetDict
dataset_dict = DatasetDict({
    "censored": censored_dataset,
    "uncensored": uncensored_dataset
})

# Save the dataset locally (optional)
dataset_dict.save_to_disk('deccp_dataset')

# Upload the dataset to Hugging Face
repo_id = "augmxnt/deccp"  # replace with your repo id
dataset_dict.push_to_hub(repo_id)

