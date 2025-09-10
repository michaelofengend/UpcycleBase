import argparse
import math
import csv
from datetime import datetime
from typing import List, Tuple

import torch
from datasets import load_dataset, get_dataset_config_names
from transformers import AutoModelForCausalLM, AutoTokenizer

from upcycle_heads import upcycle_heads_inplace
from train_upcycled import enable_moh_training, auto_device, auto_dtype


def load_baseline(model_id: str, device: str, dtype: torch.dtype):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    return tok, model


def load_upcycled_from_checkpoint(model_id: str, ckpt_path: str, device: str, dtype: torch.dtype):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    # Attach upcycling modules and load weights
    upcycle_heads_inplace(model, factor=2, top_k=None)
    import safetensors.torch as st
    state = st.load_file(f"{ckpt_path}/model.safetensors")
    model.load_state_dict(state, strict=False)
    enable_moh_training(model, identity_shortcut=False)
    return tok, model


def format_mmlu_prompt(q: str, choices: List[str]) -> str:
    letters = ["A", "B", "C", "D"]
    body = "\n".join([f"{letters[i]}. {choices[i]}" for i in range(len(choices))])
    return f"Question: {q}\n{body}\nAnswer:"


@torch.no_grad()
def option_logprob_sum(model, tok, prompt: str, option_text: str) -> float:
    # Compute logprob of option_text given prompt
    prompt_ids = tok(prompt, return_tensors="pt").to(next(model.parameters()).device)
    opt_ids = tok(option_text, return_tensors="pt", add_special_tokens=False).to(next(model.parameters()).device)
    # Concatenate
    input_ids = torch.cat([prompt_ids.input_ids, opt_ids.input_ids], dim=1)
    attn = torch.ones_like(input_ids)
    out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
    logits = out.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    # Only sum the part corresponding to option tokens
    opt_len = opt_ids.input_ids.shape[1]
    logits_opt = logits[:, -opt_len:, :]
    targets_opt = targets[:, -opt_len:]
    logprobs = torch.log_softmax(logits_opt, dim=-1)
    gather = torch.gather(logprobs, -1, targets_opt.unsqueeze(-1)).squeeze(-1)
    return float(gather.sum().item())


LETTERS = ["A", "B", "C", "D"]

def _normalize_row(ex):
    if all(k in ex for k in ("input", "A", "B", "C", "D", "target")):
        return ex["input"], [ex["A"], ex["B"], ex["C"], ex["D"]], ex["target"]
    if "question" in ex and "choices" in ex:
        ans = ex["answer"]
        if isinstance(ans, int):
            ans = LETTERS[ans]
        return ex["question"], ex["choices"], ans
    raise KeyError("Unknown MMLU row schema")

def _load_mmlu(task_name: str, split: str):
    tried = []
    for path, kwargs in [
        ("lukaemon/mmlu", dict(name=task_name, split=split, trust_remote_code=True)),
        ("cais/mmlu", dict(name=task_name, split="test", trust_remote_code=True)),
        ("hendrycks_test", dict(name=task_name, split="test", trust_remote_code=True)),
    ]:
        try:
            return load_dataset(path, **kwargs)
        except Exception as e:
            tried.append((path, str(e)))
    raise RuntimeError("Could not load MMLU. Tried:\n" + "\n".join(f"- {p}: {msg}" for p, msg in tried))

def _all_mmlu_tasks() -> List[str]:
    tried = []
    for path in ["lukaemon/mmlu", "cais/mmlu", "hendrycks_test"]:
        try:
            return get_dataset_config_names(path)
        except Exception as e:
            tried.append((path, str(e)))
    raise RuntimeError("Could not list MMLU tasks. Tried:\n" + "\n".join(f"- {p}: {msg}" for p, msg in tried))

def _build_fewshot_examples(task_name: str, shots: int, seed: int) -> List[Tuple[str, List[str], str]]:
    if shots <= 0:
        return []
    # prefer validation; fallback to train
    try:
        ds_ctx = _load_mmlu(task_name, "validation")
    except Exception:
        ds_ctx = _load_mmlu(task_name, "train")
    n = min(shots, len(ds_ctx))
    # deterministic selection
    import random
    rng = random.Random(seed)
    indices = list(range(len(ds_ctx)))
    rng.shuffle(indices)
    picks = indices[:n]
    ctx = []
    for i in picks:
        ex = ds_ctx[i]
        q, choices, gold = _normalize_row(ex)
        ctx.append((q, choices, gold))
    return ctx

def _apply_fewshot(prompt_base: str, ctx: List[Tuple[str, List[str], str]]) -> str:
    if not ctx:
        return prompt_base
    shots_text = []
    for (q_i, choices_i, gold_i) in ctx:
        shots_text.append(f"{format_mmlu_prompt(q_i, choices_i)} {gold_i}\n")
    return "\n".join(shots_text) + "\n" + prompt_base

def eval_task(model, tok, task_name: str, limit: int, shots: int, seed: int, progress_every: int = 100) -> Tuple[int, int]:
    ds = _load_mmlu(task_name, "test")
    ctx = _build_fewshot_examples(task_name, shots, seed)
    correct = 0
    total = 0
    n = len(ds) if (limit is None or limit <= 0) else min(limit, len(ds))
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    for idx, ex in enumerate(ds.select(range(n)), start=1):
        q, choices, gold_letter = _normalize_row(ex)
        prompt = format_mmlu_prompt(q, choices)
        prompt = _apply_fewshot(prompt, ctx)
        scores = [option_logprob_sum(model, tok, prompt + " ", choices[i]) for i in range(4)]
        pred_idx = int(torch.tensor(scores).argmax().item())
        gold_idx = LETTERS.index(gold_letter)
        correct += int(pred_idx == gold_idx)
        total += 1
        if progress_every and (idx % progress_every == 0):
            print(f"[{task_name}] {idx}/{n} done | running acc={correct/total:.3f}")
    return correct, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_id", default="google/gemma-3-1b-it")
    ap.add_argument("--up_ckpt", required=True, help="Path to upcycled checkpoint folder (contains model.safetensors)")
    ap.add_argument("--tasks", default="all", help="Comma-separated task names or 'all' for every MMLU subject")
    ap.add_argument("--limit", type=int, default=100, help="Max questions per task; set 0 for full benchmark")
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--csv", default=None, help="Optional path to save per-task results as CSV")
    ap.add_argument("--html", default=None, help="Optional path to save results as an HTML table")
    ap.add_argument("--progress_every", type=int, default=100, help="Print progress every N examples per subject (0 to disable)")
    args = ap.parse_args()

    device = auto_device()
    dtype = auto_dtype(device)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if len(tasks) == 1 and tasks[0].lower() == "all":
        tasks = _all_mmlu_tasks()

    # Baseline
    tok_b, model_b = load_baseline(args.baseline_id, device, dtype)
    # Upcycled
    tok_u, model_u = load_upcycled_from_checkpoint(args.baseline_id, args.up_ckpt, device, dtype)

    print("Evaluating tasks:", tasks)

    rows = []
    base_correct = base_total = up_correct = up_total = 0
    macro_baseline: List[float] = []
    macro_up: List[float] = []
    for t in tasks:
        print(f"\n>>> Subject: {t}")
        b_c, b_t = eval_task(model_b, tok_b, t, args.limit, args.shots, args.seed, args.progress_every)
        u_c, u_t = eval_task(model_u, tok_u, t, args.limit, args.shots, args.seed, args.progress_every)
        base_correct += b_c; base_total += b_t
        up_correct += u_c; up_total += u_t
        b_acc = (b_c / b_t) if b_t else 0.0
        u_acc = (u_c / u_t) if u_t else 0.0
        macro_baseline.append(b_acc)
        macro_up.append(u_acc)
        print(f"{t}: baseline {b_c}/{b_t} = {b_acc:.3f} | upcycled {u_c}/{u_t} = {u_acc:.3f}")
        rows.append([t, b_c, b_t, f"{b_acc:.4f}", u_c, u_t, f"{u_acc:.4f}"])

    if base_total:
        print(f"\nMicro avg baseline: {base_correct}/{base_total} = {base_correct/base_total:.3f}")
    if up_total:
        print(f"Micro avg upcycled: {up_correct}/{up_total} = {up_correct/up_total:.3f}")
    if macro_baseline:
        print(f"Macro avg baseline: {sum(macro_baseline)/len(macro_baseline):.3f}")
    if macro_up:
        print(f"Macro avg upcycled: {sum(macro_up)/len(macro_up):.3f}")

    if args.csv:
        try:
            with open(args.csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["task","baseline_correct","baseline_total","baseline_acc","upcycled_correct","upcycled_total","upcycled_acc"])
                w.writerows(rows)
            print(f"Wrote per-task results to {args.csv}")
        except Exception as e:
            print("Failed to write CSV:", e)

    if args.html:
        try:
            html_rows = "\n".join(
                f"<tr><td>{t}</td><td>{bc}</td><td>{bt}</td><td>{ba}</td><td>{uc}</td><td>{ut}</td><td>{ua}</td></tr>"
                for t, bc, bt, ba, uc, ut, ua in rows
            )
            summary = []
            if base_total:
                summary.append(f"<p>Micro avg baseline: {base_correct}/{base_total} = {base_correct/base_total:.3f}</p>")
            if up_total:
                summary.append(f"<p>Micro avg upcycled: {up_correct}/{up_total} = {up_correct/up_total:.3f}</p>")
            if macro_baseline:
                summary.append(f"<p>Macro avg baseline: {sum(macro_baseline)/len(macro_baseline):.3f}</p>")
            if macro_up:
                summary.append(f"<p>Macro avg upcycled: {sum(macro_up)/len(macro_up):.3f}</p>")
            html = f"""
<!doctype html>
<html><head><meta charset='utf-8'><title>MMLU Results</title>
<style>table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:6px 10px}}thead{{background:#f5f5f5}}</style>
</head><body>
<h2>MMLU Results ({datetime.now().isoformat(timespec='seconds')})</h2>
{''.join(summary)}
<table><thead><tr><th>task</th><th>baseline_correct</th><th>baseline_total</th><th>baseline_acc</th><th>upcycled_correct</th><th>upcycled_total</th><th>upcycled_acc</th></tr></thead>
<tbody>
{html_rows}
</tbody></table>
</body></html>
"""
            with open(args.html, "w") as f:
                f.write(html)
            print(f"Wrote HTML report to {args.html}")
        except Exception as e:
            print("Failed to write HTML:", e)


if __name__ == "__main__":
    main()


