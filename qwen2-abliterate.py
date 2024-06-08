import abliterator
from transformers import AutoModelForCausalLM

model = "Qwen/Qwen2-7B-Instruct"
dataset = [abliterator.get_harmful_instructions(), abliterator.get_harmless_instructions()]
cache_fname = 'my_cached_point.pth'

chat_template = None                        # optional: defaults to Llama-3 instruction template. You can use a format string e.g. ("<system>{instruction}<end><assistant>") or a custom class with format function -- it just needs an '.format(instruction="")` function. See abliterator.ChatTemplate for a very basic structure.
negative_toks = [4250]                      # optional, but highly recommended: ' cannot' in Llama's tokenizer. Tokens you don't want to be seeing. Defaults to my preset for Llama-3 models
positive_toks = [23371, 40914]              # optional, but highly recommended: ' Sure' and 'Sure' in Llama's tokenizer. Tokens you want to be seeing, basically. Defaults to my preset for Llama-3 models

my_model = abliterator.ModelAbliterator(
  model,
  dataset,
  device='cuda',
  n_devices=None,
  cache_fname=cache_fname,
  activation_layers=['resid_pre', 'resid_post', 'attn_out', 'mlp_out'],
  chat_template="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n",
  # positive_toks=positive_toks,
  # negative_toks=negative_toks
)

# Cache activations/sample dataset
my_model.cache_activations(N=95,reset=True,preserve_harmless=True)

# Test refusal directions
my_amazing_dir = find_best_refusal_dir()[0]

# Blacklist layers
my_model.blacklist_layer(0)
my_model.blacklist_layer(1)
my_model.blacklist_layer(26)
my_model.blacklist_layer(27)


my_model.apply_refusal_dirs([my_amazing_dir],layers=None)


# Save the resulting model
output_dir = "model_deccp"
model_to_save = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=None,
    config=my_model.model.cfg,
    state_dict=my_model.model.state_dict()
)
model_to_save.save_pretrained(output_dir)
