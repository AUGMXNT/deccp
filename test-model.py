import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

model_path = "modified_model"
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
prompt = "Once upon a time"
output = generator(prompt, max_length=200, num_return_sequences=1)
print(output[0]['generated_text'])
