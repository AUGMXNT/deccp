from typing import Optional, Tuple

import einops
import jaxtyping
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.inference_mode()
torch.set_default_device("cuda")

MODEL_ID = "Qwen/Qwen2-7B-Instruct"
OUTPUT_DIR = "modified_model"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

refusal_dir = torch.load(MODEL_ID.replace("/", "_") + "_refusal_dir.pt")
refusal_dir = refusal_dir.to(torch.bfloat16)

def direction_ablation_hook(activation: jaxtyping.Float[torch.Tensor, "... d_act"],
                            direction: jaxtyping.Float[torch.Tensor, "d_act"]):
    proj = einops.einsum(activation, direction.view(-1, 1), '... d_act, d_act single -> ... single') * direction
    return activation - proj

class AblationDecoderLayer(nn.Module):
    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        assert not output_attentions

        ablated = direction_ablation_hook(hidden_states, refusal_dir.to(hidden_states.device)).to(hidden_states.device)

        outputs = (ablated,)

        if use_cache:
            outputs += (past_key_value,)

        return outputs

for idx in reversed(range(len(model.model.layers))):  # for qwen 1 this needs to be changed to model.transformer.h
    model.model.layers.insert(idx, AblationDecoderLayer())

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
