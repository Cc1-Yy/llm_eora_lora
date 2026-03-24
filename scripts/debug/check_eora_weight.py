import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel


def align_to_fp_shape(mat: torch.Tensor, fp_shape):
    """
    Try to align a matrix to the full-precision weight shape.
    Return (aligned_mat, used_transpose: bool)
    """
    if tuple(mat.shape) == tuple(fp_shape):
        return mat, False
    if tuple(mat.T.shape) == tuple(fp_shape):
        return mat.T.contiguous(), True
    raise ValueError(f"Cannot align shape {tuple(mat.shape)} to fp shape {tuple(fp_shape)}")


def main():
    fp_dir = "outputs/lm/exp0/optimized_small/model"
    q_dir = "outputs/lm/exp0/quantized_optimized_small_gptq4"
    adapter_dir = "outputs/lm/exp3/1_eora_recover_q4_r16_ar1_tm-ap/adapter"

    print("Loading models...")
    fp_model = AutoModelForCausalLM.from_pretrained(fp_dir)
    q_model = AutoModelForCausalLM.from_pretrained(q_dir)
    eora_model = PeftModel.from_pretrained(q_model, adapter_dir)

    fp_layer_name = "transformer.h.0.attn.c_attn"
    eora_layer_name = "base_model.model.transformer.h.0.attn.c_attn"

    fp_layer = dict(fp_model.named_modules())[fp_layer_name]
    eora_layer = dict(eora_model.named_modules())[eora_layer_name]
    base = eora_layer.base_layer

    print("=== STEP: dequantize and compare ΔW vs BA ===")

    # full-precision weight
    W_fp = fp_layer.weight.data.float()

    # dequantized quantized weight
    W_q_raw = base.dequantize_weight()
    if not isinstance(W_q_raw, torch.Tensor):
        raise TypeError(f"dequantize_weight() did not return a tensor, got: {type(W_q_raw)}")
    W_q_raw = W_q_raw.float().cpu()

    # LoRA / EoRA low-rank update
    A = eora_layer.lora_A["default"].weight.data.float().cpu()   # [r, in]
    B = eora_layer.lora_B["default"].weight.data.float().cpu()   # [out, r]
    BA_raw = B @ A                                               # [out, in]

    print("fp weight.shape     :", tuple(W_fp.shape))
    print("q raw shape         :", tuple(W_q_raw.shape))
    print("A shape             :", tuple(A.shape))
    print("B shape             :", tuple(B.shape))
    print("BA raw shape        :", tuple(BA_raw.shape))

    # align both W_q and BA to full-precision weight shape
    W_q, q_used_T = align_to_fp_shape(W_q_raw, W_fp.shape)
    BA, ba_used_T = align_to_fp_shape(BA_raw, W_fp.shape)

    print("aligned q shape     :", tuple(W_q.shape), " transpose_used =", q_used_T)
    print("aligned BA shape    :", tuple(BA.shape), " transpose_used =", ba_used_T)

    delta = W_fp.cpu() - W_q

    d_flat = delta.flatten()
    ba_flat = BA.flatten()

    cos = torch.nn.functional.cosine_similarity(d_flat, ba_flat, dim=0).item()
    rel_err = torch.norm(delta - BA) / (torch.norm(delta) + 1e-12)

    print("\n===== DIAGNOSTIC =====")
    print(f"||W_fp||            = {torch.norm(W_fp):.6f}")
    print(f"||W_q||             = {torch.norm(W_q):.6f}")
    print(f"||ΔW||              = {torch.norm(delta):.6f}")
    print(f"||BA||              = {torch.norm(BA):.6f}")
    print(f"cos(ΔW, BA)         = {cos:.6f}")
    print(f"relative_error      = {rel_err:.6f}")


if __name__ == "__main__":
    main()