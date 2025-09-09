import os
import math
import argparse
import contextlib
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm
from itertools import cycle


def auto_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def auto_dtype(device: str):
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float16


def iterate_attention_modules(model):
    for name, module in model.named_modules():
        if all(hasattr(module, p) for p in ["q_proj", "k_proj", "v_proj", "o_proj"]):
            yield name, module


def enable_moh_training(model, identity_shortcut: bool = False):
    # Determine H from config
    text_cfg = getattr(model.config, "text_config", model.config)
    num_heads = int(getattr(text_cfg, "num_attention_heads"))
    # Use routed path, but keep top_k = H
    for _, attn in iterate_attention_modules(model):
        if hasattr(attn, "set_moh"):
            attn.set_moh(enabled=True, identity_shortcut=identity_shortcut, top_k=num_heads)


def freeze_base_qkv(model):
    for name, param in model.named_parameters():
        train_this = (
            "q_proj_extra." in name or
            "moh_router." in name
        )
        param.requires_grad = train_this


def build_text_dataloader(tokenizer, dataset_name: str, split: str, seq_len: int, batch_size: int, config_name: str | None = None, num_proc: int = 4):
    ds = load_dataset(dataset_name, config_name, split=split) if config_name else load_dataset(dataset_name, split=split)

    def tokenize(example):
        return tokenizer(example["text"], truncation=False, add_special_tokens=False)

    tokenized = ds.map(tokenize, batched=True, remove_columns=ds.column_names, num_proc=num_proc)

    # Concatenate and chunk into fixed-length sequences
    def group_texts(examples):
        concatenated = sum(examples["input_ids"], [])
        total_len = (len(concatenated) // seq_len) * seq_len
        concatenated = concatenated[:total_len]
        input_ids = [concatenated[i:i + seq_len] for i in range(0, total_len, seq_len)]
        attention_mask = [[1] * seq_len for _ in range(len(input_ids))]
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    lm_ds = tokenized.map(group_texts, batched=True, num_proc=num_proc)

    return DataLoader(lm_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch, num_workers=num_proc)


def collate_batch(batch):
    input_ids = torch.tensor([b["input_ids"] for b in batch], dtype=torch.long)
    attention_mask = torch.tensor([b["attention_mask"] for b in batch], dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids.clone()}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="google/gemma-3-1b-it")
    ap.add_argument("--output_dir", default="./gemma3-1b-it-upcycled-heads-trained")
    ap.add_argument("--dataset", default="wikitext")
    ap.add_argument("--dataset_config", default="wikitext-2-raw-v1")
    ap.add_argument("--split", default="train")
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--max_steps", type=int, default=1000)  # optimizer steps
    ap.add_argument("--save_every", type=int, default=0)
    ap.add_argument("--no_autocast", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--identity_shortcut", action="store_true", help="Enable identity shortcut (bypass routed heads)")
    ap.add_argument("--num_proc", type=int, default=4, help="Number of processes for dataset mapping")
    ap.add_argument("--sanity_once", action="store_true", help="Print one-batch raw HF loss and exit")
    ap.add_argument("--mps_fp32", action="store_true", help="Force FP32 and disable autocast on MPS for numerics triage")
    args = ap.parse_args()

    device = auto_device()
    dtype = auto_dtype(device)
    if device == "mps" and args.mps_fp32:
        dtype = torch.float32
        args.no_autocast = True

    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=dtype,
        device_map=None if device in ("cuda", "mps", "cpu") else "auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    if device in ("cuda", "mps", "cpu"):
        model.to(device)

    # Import and apply upcycling from local helper
    from upcycle_heads import upcycle_heads_inplace
    upcycle_heads_inplace(model, factor=2, top_k=None)

    # Route through MoH while still selecting top_k = H
    enable_moh_training(model, identity_shortcut=args.identity_shortcut)

    # Freeze base Q/K/V, train extras + router only
    freeze_base_qkv(model)

    # Ensure no caches are used during training
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable_params)
    print(f"Trainable params: {n_train/1e6:.2f}M across {len(trainable_params)} tensors")
    # Optional: separate weight decay
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.endswith('.bias') or 'norm' in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    optimizer = torch.optim.AdamW([
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=args.lr)

    # Build data loader
    dataset_name = args.dataset
    dataset_config = args.dataset_config if args.dataset_config else None
    dl = build_text_dataloader(tokenizer, dataset_name, args.split, args.seq_len, args.batch_size, config_name=dataset_config, num_proc=args.num_proc)

    # Optional: one-batch sanity loss
    if args.sanity_once:
        batch = next(iter(dl))
        if device in ("cuda", "mps", "cpu"):
            for k in batch:
                batch[k] = batch[k].to(device)
        model.eval()
        with torch.no_grad():
            out = model(**batch, use_cache=False)
        print("raw HF loss:", float(out.loss))
        return

    # Learning rate schedule
    num_update_steps_per_epoch = max(1, math.ceil(len(dl) / args.grad_accum))
    total_steps = args.max_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    model.train()
    micro = 0
    updates = 0
    running_loss = 0.0

    autocast_enabled = (not args.no_autocast) and device in ("cuda", "mps")
    use_fp16_cuda = device == "cuda" and dtype == torch.float16 and not args.no_autocast
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16_cuda)

    data_iterator = cycle(dl)
    pbar = tqdm(total=total_steps, desc="Training")

    while updates < total_steps:
        batch = next(data_iterator)

        if device in ("cuda", "mps", "cpu"):
            for k in batch:
                batch[k] = batch[k].to(device)

        with torch.autocast(device_type=device, dtype=dtype, enabled=autocast_enabled):
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss / args.grad_accum

        if use_fp16_cuda:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        running_loss += loss.item()
        micro += 1

        if micro % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            if use_fp16_cuda:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1

            # progress/logging: running_loss already sums micro-batch losses post-division
            avg_loss = running_loss
            pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
            pbar.update(1)
            running_loss = 0.0

            if args.save_every and updates % args.save_every == 0:
                save_dir = os.path.join(args.output_dir, f"checkpoint-{updates}")
                os.makedirs(save_dir, exist_ok=True)
                model.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)

    pbar.close()

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved trained upcycled checkpoint to {args.output_dir}")


if __name__ == "__main__":
    main()


