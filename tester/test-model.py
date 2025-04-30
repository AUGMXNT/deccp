#!/usr/bin/env python3
# Script to test HF models against augmxnt/deccp dataset for refusals

import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Union
import pandas as pd
from tqdm import tqdm
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from vllm import LLM, SamplingParams

class MinosRefusalClassifier:
    def __init__(self, model_path_or_name="NousResearch/Minos-v1"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path_or_name,
            num_labels=2,
            id2label={0: "Non-refusal", 1: "Refusal"},
            label2id={"Non-refusal": 0, "Refusal": 1}
        ).to(self.device)
        
        self.model.eval()
        print("Model loaded successfully")

    def predict(self, user_text, assistant_text):
        """Single turn prediction for simple use cases"""
        return self.predict_multi_turn([{"user": user_text, "assistant": assistant_text}])
    
    def predict_multi_turn(self, conversation_turns):
        """
        Process multiple conversation turns
        
        Args:
            conversation_turns: List of dictionaries, each with 'user' and 'assistant' keys
            
        Returns:
            Dictionary with prediction results
        """
        # Format the conversation
        formatted_text = ""
        for i, turn in enumerate(conversation_turns):
            formatted_text += f"<|user|>\n{turn['user']}\n<|assistant|>\n{turn['assistant']}"
            if i < len(conversation_turns) - 1:
                formatted_text += "\n"  # Add newline between turns
        
        inputs = self.tokenizer(
            formatted_text,
            return_tensors="pt",
            truncation=True,
            max_length=8192,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(probabilities, dim=-1)
            
        return {
            "text": formatted_text,
            "prediction": self.model.config.id2label[prediction.item()],
            "confidence": probabilities[0][prediction.item()].item()
        }

def load_model(model_name, gpu_memory_utilization=0.9):
    """Load the model using vLLM"""
    print(f"Loading model: {model_name}")
    llm = LLM(model=model_name, gpu_memory_utilization=gpu_memory_utilization, max_model_len=16384)
    return llm

def load_deccp_dataset():
    """Load the augmxnt/deccp dataset"""
    print("Loading augmxnt/deccp dataset")
    dataset = load_dataset("augmxnt/deccp")
    return dataset["censored"]

def get_model_response(llm, prompt, max_tokens=1024, temperature=0.0, top_p=1.0):
    """Get model response using vLLM"""
    # Create a system prompt + user prompt conversation
    conversation = [
        {
            "role": "system",
            "content": "You are a helpful assistant"
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    # Set sampling parameters
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    
    # Get response
    outputs = llm.chat(conversation, sampling_params)
    response = outputs[0].outputs[0].text
    return response

def run_tests(model_name, save_dir="results", num_samples=None, max_tokens=1024):
    """Run the tests on the dataset and save results"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Load model, dataset, and classifier
    llm = load_model(model_name)
    dataset = load_deccp_dataset()
    classifier = MinosRefusalClassifier()
    
    # Limit samples if specified
    if num_samples is not None and num_samples > 0:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    results = []
    refusal_count = 0
    
    # Process each dataset example
    for i, example in enumerate(tqdm(dataset, desc="Testing prompts")):
        prompt = example["text"]
        
        # Get model response
        response = get_model_response(llm, prompt, max_tokens=max_tokens)
        
        # Classify the response
        classification = classifier.predict(prompt, response)
        
        # Check if it's a refusal
        is_refusal = classification["prediction"] == "Refusal"
        if is_refusal:
            refusal_count += 1
        
        # Store result
        result = {
            "id": i,
            "prompt": prompt,
            "response": response,
            "classification": classification["prediction"],
            "confidence": classification["confidence"],
            "is_refusal": is_refusal,
        }
        results.append(result)
    
    # Calculate refusal rate
    refusal_rate = refusal_count / len(dataset) if len(dataset) > 0 else 0
    
    # Create a summary
    summary = {
        "model": model_name,
        "dataset": "augmxnt/deccp",
        "samples_tested": len(dataset),
        "refusal_count": refusal_count,
        "refusal_rate": refusal_rate,
        "timestamp": datetime.now().isoformat(),
    }
    
    # Save results as CSV and JSON
    base_filename = f"{model_name.replace('/', '_')}_deccp_test"
    
    # Create a DataFrame and save as CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(save_dir, f"{base_filename}.csv")
    df.to_csv(csv_path, index=False)
    
    # Save detailed results as JSON
    json_path = os.path.join(save_dir, f"{base_filename}.json")
    with open(json_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 50)
    print(f"Test Summary for {model_name}:")
    print(f"Samples tested: {len(dataset)}")
    print(f"Refusal count: {refusal_count}")
    print(f"Refusal rate: {refusal_rate:.2%}")
    print(f"Results saved to: {csv_path}")
    print("=" * 50)
    
    return {"summary": summary, "results": results}

def main():
    parser = argparse.ArgumentParser(description="Test HF LLMs against deccp dataset for refusals")
    parser.add_argument("model", type=str, help="HuggingFace model ID to test")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to test (default: all)")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum tokens for response generation")
    parser.add_argument("--save-dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization for vLLM")
    
    args = parser.parse_args()
    
    run_tests(
        model_name=args.model,
        save_dir=args.save_dir,
        num_samples=args.num_samples,
        max_tokens=args.max_tokens
    )

if __name__ == "__main__":
    main()
