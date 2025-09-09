# eval_mmlu.py
# Run MMLU on baseline vs upcycled copy with EleutherAI lm-eval-harness.
# pip install "lm-eval==0.4.2" transformers accelerate datasets
import argparse
import torch
from lm_eval import evaluator

def _auto_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def _auto_dtype(device: str) -> str:
    # MPS rejects bfloat16; use float16 on MPS, else prefer bfloat16 on CPU
    return "float16" if device == "mps" else "bfloat16"

def run_eval(model_id, dtype=None, device=None):
    # Use the HF causal model wrapper
    device = device or _auto_device()
    dtype = dtype or _auto_dtype(device)
    model_args = f"pretrained={model_id},dtype={dtype},trust_remote_code=True"
    # "mmlu" is the 57-task aggregate in lm-eval. You can also try "hendrycksTest"
    tasks = ["mmlu"]
    results = evaluator.simple_evaluate(
        model="hf-causal",
        model_args=model_args,
        tasks=tasks,
        batch_size="auto",
        device=device,
        limit=None,                  # set small int to do a smoke test
        gen_kwargs={"do_sample": False}
    )
    # Print the MMLU average for quick comparison
    mmlu = results.get("results", {}).get("mmlu", {})
    print(model_id, "→ acc:", mmlu.get("acc"), "acc_norm:", mmlu.get("acc_norm"))
    return results

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="google/gemma-3-1b-it")
    ap.add_argument("--upcycled", default="./gemma3-1b-it-upcycled-heads")
    ap.add_argument("--dtype", default=None, help="Override dtype; default auto (fp16 on MPS, bf16 otherwise)")
    ap.add_argument("--device", default=None, help="Override device; default auto (mps>cuda>cpu)")
    args = ap.parse_args()

    print("Running MMLU on baseline…")
    run_eval(args.baseline, dtype=args.dtype, device=args.device)
    print("Running MMLU on upcycled copy…")
    run_eval(args.upcycled, dtype=args.dtype, device=args.device)
