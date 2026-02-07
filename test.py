from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

mdir="outputs/optimized_lm_small/model"
tok=AutoTokenizer.from_pretrained(mdir)
model=AutoModelForCausalLM.from_pretrained(mdir).cuda().eval()

x=tok("Hello, my name is", return_tensors="pt").to("cuda")
with torch.no_grad():
    out=model.generate(**x, max_new_tokens=20)
print(tok.decode(out[0], skip_special_tokens=True))
