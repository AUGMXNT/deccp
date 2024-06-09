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


for layer in model.model.layers:
    def ablation_forward(hidden_states, **kwargs):
        ablated = direction_ablation_hook(hidden_states, refusal_dir.to(hidden_states.device)).to(hidden_states.device)
        return layer.forward(ablated, **kwargs)

    layer.forward = ablation_forward

model.save_pretrained("modified_model")
tokenizer.save_pretrained("modified_model")
