from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2-7B-Instruct"

SKIP_BEGIN_LAYERS=1
SKIP_END_LAYERS=0
SCALE_FACTOR = 1.0

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Reload the model in CPU memory with bfloat16 data type
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map='cpu'
)
model.requires_grad_(False)

# Load your pre-computed direction tensor
refusal_dir = torch.load(MODEL_ID.replace("/", "_") + "_refusal_dir.pt")
refusal_dir = refusal_dir.to(model.device)

# Get the language model component and check it's as expected.
lm_model = model.model
assert hasattr(lm_model, 'layers'), "The model does not have the expected structure."

# Check the ranges are valid.
num_layers = len(lm_model.layers)
assert SKIP_BEGIN_LAYERS >= 0, "SKIP_BEGIN_LAYERS must be >= 0."
assert SKIP_END_LAYERS >= 0, "SKIP_END_LAYERS must be >= 0."
assert SKIP_BEGIN_LAYERS + SKIP_END_LAYERS < num_layers, "SKIP_BEGIN_LAYERS + SKIP_END_LAYERS must be < num_layers."

bar_layers = tqdm(total= (num_layers - (SKIP_BEGIN_LAYERS + SKIP_END_LAYERS)) * 2, desc = "Modifying tensors")

# Cast any ops performed on CPU up to float32... If you have newer CPU might be able to use bfloat16 for this.
# NOTE: Use a negative scale_factor to "induce" and a positive scale_factor of < 1 to "ablate" less.
def modify_tensor(tensor_data, refusal_dir, scale_factor: float = 1.0):
    assert scale_factor <= 1.0, "Using a scale_factor of > 1 doesn't make sense..."
    tensor_float32 = tensor_data.to(torch.float32)
    refusal_dir_float32 = refusal_dir.to(torch.float32)
    # Ensure refusal_dir is a 1-dimensional tensor
    if refusal_dir_float32.dim() > 1:
        refusal_dir_float32 = refusal_dir_float32.view(-1)
    tensor_float32 -= scale_factor * torch.matmul(torch.outer(refusal_dir_float32, refusal_dir_float32), tensor_float32)
    tensor_modified = tensor_float32.to(torch.bfloat16)
    bar_layers.update(1)
    return torch.nn.Parameter(tensor_modified)

# Modify the 'self_attn.o_proj.weight' and 'mlp.down_proj.weight' in each chosen layer.
# NOTE: These tensors names are specific to "Qwen2" and may need changing.
#       - See here for others: https://github.com/arcee-ai/mergekit/tree/main/mergekit/_data/architectures
for layer_idx in range(SKIP_BEGIN_LAYERS, num_layers - SKIP_END_LAYERS):
    lm_model.layers[layer_idx].self_attn.o_proj.weight = modify_tensor(
        lm_model.layers[layer_idx].self_attn.o_proj.weight.data, refusal_dir, SCALE_FACTOR
    )
    lm_model.layers[layer_idx].mlp.down_proj.weight = modify_tensor(
        lm_model.layers[layer_idx].mlp.down_proj.weight.data, refusal_dir, SCALE_FACTOR
    )

bar_layers.close()

# Save the modified model and original tokenizer
print("Saving modified model (with original tokenizer)...")
model.save_pretrained("modified_model")
tokenizer.save_pretrained("modified_model")
