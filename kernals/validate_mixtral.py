"""
validate_mixtral.py — MSML 605 (Gokul)
Validates Mixtral loads and patches correctly with Triton kernels.

Usage:
    python validate_mixtral.py
"""
import torch
from patch_models import load_mixtral, validate_patch


def find_mixtral_router(model):
    print("\n=== Mixtral Router Modules ===")
    count=0
    for name, module in model.named_modules():
        if type(module).__name__=="MixtralSparseMoeBlock":
            print(f"  {name:80s} {type(module).__name__}")
            count+=1
    print(f"  total MoE layers: {count}")


if __name__=="__main__":
    print("="*60)
    print("  Mixtral Validation")
    print("="*60)

    # baseline — no kernels, fp16
    model, tok=load_mixtral(precision="gptq", kernels=False)
    _, base_logits=validate_patch(model, tok, "Mixtral baseline")
    find_mixtral_router(model)
    del model; torch.cuda.empty_cache()

    # patched — with Triton kernels
    model, tok=load_mixtral(precision="gptq", kernels=True)
    _, patch_logits=validate_patch(model, tok, "Mixtral patched")
    print("  note: Mixtral GPTQ uses RMSNorm kernel replacements only; router softmax patching is disabled for numeric correctness.")
    max_err=(patch_logits-base_logits).abs().max().item()
    print(f"  max logit diff: {max_err:.4f}  {'PASS' if max_err<0.1 else 'FAIL'}")
    del model; torch.cuda.empty_cache()
