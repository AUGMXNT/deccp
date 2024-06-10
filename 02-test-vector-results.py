from typing import Optional, Tuple

import einops
import jaxtyping
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig

torch.inference_mode()

torch.set_default_device("cuda")

MODEL_ID = "Qwen/Qwen2-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

refusal_dir = torch.load(MODEL_ID.replace("/", "_") + "_refusal_dir.pt")
refusal_dir = refusal_dir.to(torch.bfloat16)


def direction_ablation_hook(activation: jaxtyping.Float[torch.Tensor, "... d_act"],
                            direction: jaxtyping.Float[torch.Tensor, "d_act"]):
    proj = einops.einsum(activation, direction.view(-1, 1), '... d_act, d_act single -> ... single') * direction
    return activation - proj

class AblationDecoderLayer(nn.Module):
    def __init__(self, original_layer):
        super(AblationDecoderLayer, self).__init__()
        self.original_layer = original_layer

    def forward(self, *args, **kwargs):
        hidden_states = args[0]
        ablated = direction_ablation_hook(hidden_states, refusal_dir.to(hidden_states.device)).to(hidden_states.device)
        args = (ablated,) + args[1:]
        return self.original_layer.forward(*args, **kwargs)


for idx in range(len(model.model.layers)):
    model.model.layers[idx] = AblationDecoderLayer(model.model.layers[idx])


# Test Inference
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

'''
# Test Inference
prompt = "Tell me about the Tiananmen Square Massacre"
conversation=[]
conversation.append({"role": "user", "content": prompt})
toks = tokenizer.apply_chat_template(conversation=conversation, add_generation_prompt=True, return_tensors="pt")
gen = model.generate(toks.to(model.device), max_new_tokens=200)
decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
print(decoded)

### Doesn't work ofc
# model.save_pretrained("modified_model")               
# tokenizer.save_pretrained("modified_model")   
'''
