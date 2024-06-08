import jaxtyping
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
import einops
from tqdm import tqdm

torch.inference_mode()
torch.set_default_device("cpu")

MODEL_ID = "Qwen/Qwen2-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

instructions = 500
# layer_range = range(int(len(model.model.layers) * 0.5), int(len(model.model.layers) * 0.7))
layer_range = range(2, len(model.model.layers) - 2)
pos = -1

print("Instruction count: " + str(instructions))
print("Layer range: " + str(layer_range))

with open("harmful.txt", "r") as f:
    harmful = f.readlines()

with open("harmless.txt", "r") as f:
    harmless = f.readlines()

harmful_instructions = random.sample(harmful, len(harmful))
harmless_instructions = random.sample(harmless, instructions)

harmful_toks = [
    tokenizer.apply_chat_template(conversation=[{"role": "user", "content": insn}], add_generation_prompt=True,
                                  return_tensors="pt") for insn in harmful_instructions]
harmless_toks = [
    tokenizer.apply_chat_template(conversation=[{"role": "user", "content": insn}], add_generation_prompt=True,
                                  return_tensors="pt") for insn in harmless_instructions]

max_its = instructions*2
bar = tqdm(total=max_its)

def generate(toks):
    bar.update(n=1)
    return model.generate(toks.to(model.device), use_cache=False, max_new_tokens=1, return_dict_in_generate=True, output_hidden_states=True)

harmful_outputs = [generate(toks) for toks in harmful_toks]
harmless_outputs = [generate(toks) for toks in harmless_toks]

bar.close()

def calculate_refusal_dir(layer_idx):
    harmful_hidden = [output.hidden_states[0][layer_idx][:, pos, :] for output in harmful_outputs]
    harmless_hidden = [output.hidden_states[0][layer_idx][:, pos, :] for output in harmless_outputs]

    harmful_mean = torch.stack(harmful_hidden).mean(dim=0)
    harmless_mean = torch.stack(harmless_hidden).mean(dim=0)

    refusal_dir = harmful_mean - harmless_mean
    refusal_dir = refusal_dir / refusal_dir.norm()

    return refusal_dir

for layer_idx in layer_range:
    refusal_dir = calculate_refusal_dir(layer_idx)
    torch.save(refusal_dir, f"{MODEL_ID.replace('/', '_')}_refusal_dir.{layer_idx}.pt")
