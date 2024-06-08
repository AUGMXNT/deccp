from typing import Optional, Tuple

import einops
import jaxtyping
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

torch.inference_mode()
torch.set_default_device("cuda")

MODEL_ID = "Qwen/Qwen2-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

refusal_dirs = {}
for layer_idx in range(2, len(model.model.layers) - 2):
    refusal_dir = torch.load(f"{MODEL_ID.replace('/', '_')}_refusal_dir.{layer_idx}.pt")
    refusal_dirs[layer_idx] = refusal_dir.to(torch.bfloat16)

def direction_ablation_hook(activation: jaxtyping.Float[torch.Tensor, "... d_act"],
                            direction: jaxtyping.Float[torch.Tensor, "d_act"]):
    proj = einops.einsum(activation, direction.view(-1, 1), '... d_act, d_act single -> ... single') * direction
    return activation - proj

class AblationDecoderLayer(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx

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

        if self.layer_idx in refusal_dirs:
            refusal_dir = refusal_dirs[self.layer_idx].to(hidden_states.device)
            ablated = direction_ablation_hook(hidden_states, refusal_dir).to(hidden_states.device)
        else:
            ablated = hidden_states

        outputs = (ablated,)

        if use_cache:
            outputs += (past_key_value,)

        return outputs

for idx in range(len(model.model.layers)):
    if idx in range(2, len(model.model.layers) - 2):
        model.model.layers[idx] = AblationDecoderLayer(idx)

streamer = TextStreamer(tokenizer)

while True:
    conversation = []
    prompt = input()
    conversation.append({"role": "user", "content": prompt})
    toks = tokenizer.apply_chat_template(conversation=conversation,
                                         add_generation_prompt=True, return_tensors="pt")

    gen = model.generate(toks.to(model.device), streamer=streamer, max_new_tokens=1337, use_cache=False)

    decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
    conversation.append({"role": "assistant", "content": decoded})
