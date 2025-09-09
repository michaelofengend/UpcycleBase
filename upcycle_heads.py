import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-3-1b-it"
SAVE_DIR = "./gemma3-1b-it-upcycled-heads"

def _find_attn_modules(model):
    # Works for HF Gemma-3: look for modules that expose q_proj/k_proj/v_proj/o_proj
    attn_modules = []
    for name, module in model.named_modules():
        has = all(hasattr(module, attr) for attr in ["q_proj","k_proj","v_proj","o_proj"])
        if has:
            attn_modules.append((name, module))
    if not attn_modules:
        raise RuntimeError("Could not find attention modules with q_proj/k_proj/v_proj/o_proj")
    return attn_modules

def upcycle_heads_inplace(model, factor=2, top_k=None):
    """
    Attach a second bank of head parameters and a MoH-style router stub.
    By default the router selects the original H heads, so forward math is unchanged.
    """
    if factor != 2:
        raise ValueError("This minimalist version supports factor=2 only for clarity.")
    # Read config for info only
    H = getattr(model.config, "num_attention_heads", None)
    KV = getattr(model.config, "num_key_value_heads", None)
    if H is None or KV is None:
        raise ValueError("Model config missing num_attention_heads or num_key_value_heads")

    if top_k is None:
        top_k = H  # keep same top-K as before

    for name, attn in _find_attn_modules(model):
        # Duplicate parameters as a second, currently-unused bank.
        # We store them as buffers to avoid changing forward compute or optimizer param lists.
        # Create extra bank only for Q; keep K/V on original path (MQA-safe, leaner)
        proj = getattr(attn, "q_proj")
        q_extra = nn.Linear(
            proj.in_features,
            proj.out_features,
            bias=(proj.bias is not None),
            device=proj.weight.device,
            dtype=proj.weight.dtype,
        )
        with torch.no_grad():
            q_extra.weight.copy_(proj.weight)
            if proj.bias is not None:
                q_extra.bias.copy_(proj.bias)
        setattr(attn, "q_proj_extra", q_extra)

        # Control buffers
        attn.register_buffer("_moh_top_k", torch.tensor(int(top_k)))  # scalar
        attn.register_buffer("_moh_enabled", torch.tensor(1, dtype=torch.uint8))
        attn.register_buffer("_moh_identity_shortcut", torch.tensor(1, dtype=torch.uint8))
        attn.register_buffer("_moh_heads", torch.tensor(H, dtype=torch.int), persistent=False)

        # Add a tiny learnable router: hidden_size -> 2H logits, biased to pick original H at init
        hidden_size = getattr(attn.q_proj, 'in_features', getattr(model.config, 'hidden_size', None))
        if hidden_size is None:
            hidden_size = getattr(getattr(model.config, 'text_config', model.config), 'hidden_size')
        attn.moh_router = nn.Linear(
            hidden_size,
            2 * H,
            bias=True,
            device=attn.q_proj.weight.device,
            dtype=attn.q_proj.weight.dtype,
        )
        with torch.no_grad():
            nn.init.zeros_(attn.moh_router.weight)
            # Interleave bias so extras are selected from step 1 (ensures gradients flow to new heads)
            bias = torch.zeros(2 * H, dtype=attn.q_proj.weight.dtype, device=attn.q_proj.weight.device)
            for i in range(H):
                bias[i] = 1.00 - (i % 2) * 1e-2
                bias[H + i] = 1.01 - ((i + 1) % 2) * 1e-2
            attn.moh_router.bias.copy_(bias)

        # tiny helper so users can flip routing later if they want
        def enable_moh(self, enabled: bool = True):
            val = torch.tensor(1 if enabled else 0, dtype=torch.uint8, device=self._moh_enabled.device)
            self._moh_enabled.copy_(val)
        attn.enable_moh = enable_moh.__get__(attn, attn.__class__)

        def set_moh(self, enabled: bool = True, identity_shortcut: bool = True, top_k: int = None):
            self._moh_enabled.copy_(torch.tensor(1 if enabled else 0, dtype=torch.uint8, device=self._moh_enabled.device))
            self._moh_identity_shortcut.copy_(torch.tensor(1 if identity_shortcut else 0, dtype=torch.uint8, device=self._moh_identity_shortcut.device))
            if top_k is not None:
                self._moh_top_k.copy_(torch.tensor(int(top_k), device=self._moh_top_k.device))
        attn.set_moh = set_moh.__get__(attn, attn.__class__)

        # Minimal head routing via projection hooks, preserving the rest of attention (rotary, caches, etc.)
        def make_gated_hook(attn_module, proj_name):
            extra = getattr(attn_module, f"{proj_name}_extra")

            def hook(mod, inputs, output):
                if int(attn_module._moh_enabled.item()) == 0 or int(attn_module._moh_identity_shortcut.item()) == 1:
                    return output
                x = inputs[0]
                y_orig = output
                y_extra = extra(x)

                H_local = int(attn_module._moh_heads.item())
                head_dim = y_orig.shape[-1] // H_local

                y_orig = y_orig.view(*y_orig.shape[:-1], H_local, head_dim)
                y_extra = y_extra.view(*y_extra.shape[:-1], H_local, head_dim)

                # Per-head soft blend (orig vs extra) so router gets gradients; preserves H heads alignment
                logits = attn_module.moh_router(x).view(*x.shape[:-1], H_local, 2)  # [B, T, H, 2]
                probs = F.softmax(logits, dim=-1)
                p_orig = probs[..., 0:1]  # [B, T, H, 1]
                p_extra = probs[..., 1:1+1]
                y_mix = p_orig * y_orig + p_extra * y_extra  # [B, T, H, Hd]
                y_out = y_mix.reshape(*y_mix.shape[:-2], -1)
                return y_out

            return hook

        attn.q_proj.register_forward_hook(make_gated_hook(attn, "q_proj"))
        # Keep K and V on the original path to respect Gemma-3's MQA layout

    return model

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # On Apple Silicon (MPS), bfloat16 is not supported. Use float16 on MPS, else bfloat16.
    dtype = torch.float16 if torch.backends.mps.is_available() else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto"
    )

    # Keep a frozen copy for equality test
    base = copy.deepcopy(model).eval()

    # Factor of 2 means we have 2 banks of heads
    upcycle_heads_inplace(model, factor=2, top_k=None)
    model.eval()

    # Deterministic prompt and settings
    torch.manual_seed(7)
    prompt = "Plants create energy through a process known as"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out1 = base.generate(**inputs, max_new_tokens=16, do_sample=False, use_cache=True)
        out2 = model.generate(**inputs, max_new_tokens=16, do_sample=False, use_cache=True)

    text1 = tokenizer.decode(out1[0], skip_special_tokens=True)
    text2 = tokenizer.decode(out2[0], skip_special_tokens=True)

    print("BASE:     ", text1)
    print("UPCYCLED: ", text2)
    print("IDENTICAL:", text1 == text2)

    os.makedirs(SAVE_DIR, exist_ok=True)
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"Saved upcycled checkpoint to {SAVE_DIR}")

if __name__ == "__main__":
    main()
