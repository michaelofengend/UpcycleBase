import os, torch
import safetensors.torch as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from upcycle_heads import upcycle_heads_inplace
from train_upcycled import enable_moh_training, auto_device, auto_dtype

ckpt = "/Users/michaelofengenden/Desktop/UpcycleBase/gemma3-1b-it-upcycled-heads-trained/checkpoint-650"
base_id = "google/gemma-3-1b-it"
device = auto_device()
dtype = auto_dtype(device)

# 1) Load base, attach upcycling modules
tok = AutoTokenizer.from_pretrained(base_id)
model = AutoModelForCausalLM.from_pretrained(
    base_id, dtype=dtype, device_map="auto", trust_remote_code=True, attn_implementation="eager"
)
upcycle_heads_inplace(model, factor=2, top_k=None)

# 2) Load the checkpoint weights into the upcycled model
sd = st.load_file(os.path.join(ckpt, "model.safetensors"))
missing, unexpected = model.load_state_dict(sd, strict=False)
# optional: print(missing, unexpected)
# 3) Enable routed path (or pass identity_shortcut=True for baseline)
enable_moh_training(model, identity_shortcut=False)
print("loaded_missing:", len(missing), "loaded_unexpected:", len(unexpected))
probe = next(m for _, m in model.named_modules() if hasattr(m, "moh_router"))
print("moh_router.weight.norm:", probe.moh_router.weight.norm().item())
print("has q_proj_extra:", hasattr(probe, "q_proj_extra"))
print("identity_shortcut:", int(probe._moh_identity_shortcut.item()))  

# 4) Compare baseline vs routed deterministically
prompt = "Explain photosynthesis in one sentence."
inputs = tok(prompt, return_tensors="pt").to(next(model.parameters()).device)

# Baseline (identity on)
for _, attn in model.named_modules():
    if hasattr(attn, "set_moh"):
        attn.set_moh(enabled=True, identity_shortcut=True)
with torch.no_grad():
    out_base = model.generate(**inputs, max_new_tokens=48, do_sample=False)
base_txt = tok.decode(out_base[0], skip_special_tokens=True)

# Routed (identity off)
for _, attn in model.named_modules():
    if hasattr(attn, "set_moh"):
        attn.set_moh(enabled=True, identity_shortcut=False)
with torch.no_grad():
    out_routed = model.generate(**inputs, max_new_tokens=48, do_sample=False)
routed_txt = tok.decode(out_routed[0], skip_special_tokens=True)

print("baseline == routed:", base_txt == routed_txt)
print("routed:", routed_txt)