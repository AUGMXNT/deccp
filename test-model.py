import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

model_path = "modified_model"
model_path = "augmxnt/Qwen2-7B-Instruct-deccp"
model_path = "Qwen/Qwen2-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

streamer = TextStreamer(tokenizer)

with open("harmful.txt", "r") as f:
    harmful = f.readlines()

    for prompt in harmful:
        print('===')
        print(prompt)
        print('---')
        conversation=[]
        conversation.append({"role": "user", "content": prompt})
        toks = tokenizer.apply_chat_template(conversation=conversation,
            add_generation_prompt=True, return_tensors="pt")

        gen = model.generate(toks.to(model.device), streamer=streamer, max_new_tokens=200)

        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
        conversation.append({"role": "assistant", "content": decoded})
