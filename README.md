## UpcycleBase — Minimal MoH-style head upcycling for Gemma-3

This repo adds a second bank of attention heads to Gemma-3 (1B IT), with a tiny router that selects top-K heads per token. It initializes to pick the original H heads so outputs are identical at start, then lets you gradually relax the selection during fine-tuning.

### Requirements
- Python 3.10+
- PyTorch ≥ 2.1
- transformers ≥ 4.56.1
- accelerate, datasets
- lm-eval == 0.4.2 (for MMLU)

Install (example):
```bash
pip install "torch" "transformers>=4.56.1" accelerate datasets "lm-eval==0.4.2"
```

### Apple Silicon (MPS) notes
- MPS rejects bfloat16. The code auto-uses float16 on MPS, bfloat16 on CPU.
- lm-eval accepts `--device mps` with PyTorch ≥ 2.1.

### Scripts

1) Upcycle and verify identical outputs
File: `upcycle_heads.py`
```bash
python upcycle_heads.py
```
- Loads Gemma-3 1B IT, adds a second bank of Q/K/V per attention block, and a small router.
- Router defaults to enabled but with an identity shortcut and top_k = H so generation remains identical.
- Saves a copy to `./gemma3-1b-it-upcycled-heads`.

2) Minimal training with double heads (still picks top_k = H)
File: `train_upcycled.py`
```bash
python train_upcycled.py \
  --model_id google/gemma-3-1b-it \
  --output_dir ./gemma3-1b-it-upcycled-heads-trained \
  --dataset wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --split train \
  --seq_len 512 \
  --batch_size 2 \
  --grad_accum 8 \
  --lr 5e-5 \
  --max_steps 1000
```
- Routing is active by default (identity shortcut disabled) and keeps `top_k = H`, so the router still selects the original heads initially.
- To force baseline behavior (no routed heads), add `--identity_shortcut`.
- Freezes base Q/K/V; trains only the extra banks and the router.
- Uses `use_cache=False` during training. Device and dtype are auto-detected (fp16 on MPS).
 - Routing gates Q only; K/V remain on the original path to respect Gemma‑3's MQA. The router bias is interleaved so some extra heads are selected from step 1.

3) Evaluate on MMLU
File: `eval_mmlu.py`
```bash
python eval_mmlu.py --device mps --dtype float16 --limit 100
```
- Compares baseline vs upcycled using EleutherAI lm-eval-harness.
- You can omit `--limit` for a full run.

### Toggling routing during training
To explicitly control routing from Python:
```python
H = int(getattr(getattr(model.config, "text_config", model.config), "num_attention_heads"))
for _, attn in model.named_modules():
    if hasattr(attn, "set_moh"):
        # Enable routing path, keep same number of selected heads
        attn.set_moh(enabled=True, identity_shortcut=False, top_k=H)
```
- For a clean “additive capacity” phase, freeze original Q/K/V and train only `*_extra_*` params plus `moh_router`.
- To sparsify more, gradually reduce `top_k` below H or anneal the router bias toward 0.

### Baseline path vs routed path (identity shortcut)
- `--identity_shortcut` enabled: bypasses routed heads and uses the original attention path; outputs are identical to stock Gemma and extra params do not receive gradients.
- Default (no flag): routed path active; Q/K/V are doubled and the router selects `top_k = H` heads among `2H`. The router is biased to the original heads initially, so behavior starts near baseline while gradients flow to the new capacity.

### Notes & limitations
- The routed forward falls back to the original path when caches are used; training uses `use_cache=False`.
- We do not change the model config head counts; all changes live in-module for minimalism and portability.

### Sanity checks and useful flags
- `--sanity_once`: prints a single raw HF loss on one batch and exits. Expect a single-digit loss (≈2–6) when things are wired correctly; significantly higher implies routing/math issues.
- `--identity_shortcut`: enables the baseline path (bypass routing). Use this to confirm baseline loss looks sane.
- `--mps_fp32`: forces float32 and disables autocast on MPS to rule out precision issues.

Example sanity run:
```bash
python train_upcycled.py \
  --model_id google/gemma-3-1b-it \
  --dataset wikitext --dataset_config wikitext-2-raw-v1 \
  --split train --seq_len 512 --batch_size 2 --sanity_once
```


