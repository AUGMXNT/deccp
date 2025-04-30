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
    print("Loading augmxnt/deccp dataset (censored split)")
    dataset = load_dataset("augmxnt/deccp")
    return dataset["censored"]  # Using the "censored" split

def prepare_conversations(prompts):
    """Prepare batch of conversations from prompts"""
    conversations = []
    for prompt in prompts:
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
        conversations.append(conversation)
    return conversations

def get_model_responses_batch(llm, prompts, max_tokens=1024, temperature=0.0, top_p=1.0, batch_size=16):
    """Get model responses in batches using vLLM"""
    # Set sampling parameters
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    
    # Process in batches
    all_responses = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        conversations = prepare_conversations(batch_prompts)
        
        # Get responses
        outputs = llm.chat(conversations, sampling_params)
        
        # Extract text responses
        batch_responses = [output.outputs[0].text for output in outputs]
        all_responses.extend(batch_responses)
    
    return all_responses

def run_tests(model_name, save_dir="results", num_samples=None, max_tokens=1024, batch_size=16, test_runs=1):
    """Run the tests on the dataset and save results"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Load model, dataset, and classifier
    llm = load_model(model_name)
    dataset = load_deccp_dataset()
    classifier = MinosRefusalClassifier()
    
    # Limit samples if specified
    if num_samples is not None and num_samples > 0:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    # Extract all prompts
    all_prompts = [example["text"] for example in dataset]
    
    # Results for all runs
    all_run_results = []
    
    # Run multiple test iterations if requested
    for run in range(1, test_runs + 1):
        print(f"\nTest run {run}/{test_runs}")
        
        # Process all prompts in batches
        print(f"Processing {len(all_prompts)} prompts in batches of {batch_size}...")
        all_responses = get_model_responses_batch(
            llm, 
            all_prompts, 
            max_tokens=max_tokens, 
            batch_size=batch_size
        )
        
        run_results = []
        refusal_count = 0
        
        # Classify all responses
        print("Classifying responses...")
        for i, (prompt, response) in enumerate(tqdm(zip(all_prompts, all_responses), desc="Classifying responses", total=len(all_prompts))):
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
                "run": run
            }
            run_results.append(result)
        
        # Add this run's results to all results
        all_run_results.extend(run_results)
        
        # Calculate refusal rate for this run
        run_refusal_rate = refusal_count / len(dataset) if len(dataset) > 0 else 0
        print(f"Run {run} refusal rate: {run_refusal_rate:.2%}")
    
    # Aggregate results across runs
    combined_results = {}
    for result in all_run_results:
        prompt_id = result["id"]
        if prompt_id not in combined_results:
            combined_results[prompt_id] = {
                "prompt": result["prompt"],
                "runs": [],
                "refusal_count": 0,
                "total_runs": test_runs
            }
        
        combined_results[prompt_id]["runs"].append({
            "run": result["run"],
            "response": result["response"],
            "classification": result["classification"],
            "confidence": result["confidence"],
            "is_refusal": result["is_refusal"]
        })
        
        if result["is_refusal"]:
            combined_results[prompt_id]["refusal_count"] += 1
    
    # Calculate consistency metrics
    always_refuse = 0
    sometimes_refuse = 0
    never_refuse = 0
    
    for prompt_id, data in combined_results.items():
        refusal_ratio = data["refusal_count"] / test_runs
        if refusal_ratio == 1.0:
            always_refuse += 1
        elif refusal_ratio == 0.0:
            never_refuse += 1
        else:
            sometimes_refuse += 1
    
    # Create a summary
    summary = {
        "model": model_name,
        "dataset": "augmxnt/deccp (censored split)",
        "samples_tested": len(dataset),
        "test_runs": test_runs,
        "always_refuse_count": always_refuse,
        "always_refuse_percentage": (always_refuse / len(dataset)) if len(dataset) > 0 else 0,
        "sometimes_refuse_count": sometimes_refuse,
        "sometimes_refuse_percentage": (sometimes_refuse / len(dataset)) if len(dataset) > 0 else 0,
        "never_refuse_count": never_refuse,
        "never_refuse_percentage": (never_refuse / len(dataset)) if len(dataset) > 0 else 0,
        "timestamp": datetime.now().isoformat(),
    }
    
    # Save results as CSV and JSON
    base_filename = f"{model_name.replace('/', '_')}_deccp_test"
    
    # Create a DataFrame for each run
    df = pd.DataFrame(all_run_results)
    csv_path = os.path.join(save_dir, f"{base_filename}_all_runs.csv")
    df.to_csv(csv_path, index=False)
    
    # Create a DataFrame for combined results
    combined_df_rows = []
    for prompt_id, data in combined_results.items():
        row = {
            "id": prompt_id,
            "prompt": data["prompt"],
            "refusal_count": data["refusal_count"],
            "refusal_percentage": data["refusal_count"] / test_runs,
        }
        combined_df_rows.append(row)
    
    combined_df = pd.DataFrame(combined_df_rows)
    combined_csv_path = os.path.join(save_dir, f"{base_filename}_combined.csv")
    combined_df.to_csv(combined_csv_path, index=False)
    
    # Save detailed results as JSON
    json_path = os.path.join(save_dir, f"{base_filename}.json")
    with open(json_path, "w") as f:
        json.dump({"summary": summary, "results": combined_results}, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 50)
    print(f"Test Summary for {model_name}:")
    print(f"Samples tested: {len(dataset)}")
    print(f"Test runs per sample: {test_runs}")
    print(f"Always refuse count: {always_refuse} ({(always_refuse / len(dataset)):.2%})")
    print(f"Sometimes refuse count: {sometimes_refuse} ({(sometimes_refuse / len(dataset)):.2%})")
    print(f"Never refuse count: {never_refuse} ({(never_refuse / len(dataset)):.2%})")
    print(f"Results saved to: {csv_path}")
    print("=" * 50)
    
    return {"summary": summary, "results": combined_results}
    
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
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for inference")
    parser.add_argument("--test-runs", type=int, default=5, help="Number of times to test each sample")
    
    args = parser.parse_args()
    
    run_tests(
        model_name=args.model,
        save_dir=args.save_dir,
        num_samples=args.num_samples,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        test_runs=args.test_runs
    )

if __name__ == "__main__":
    main()
