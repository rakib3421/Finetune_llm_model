# merge_lora_fixed.py
import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL  = "stabilityai/stablelm-2-zephyr-1_6b"
LORA_PATH   = r"stablelm-2-zephyr-1_6b-eade-finetuned/checkpoint-20"  # your path
OUT_PATH    = r"./stablelm-2-zephyr-1_6b-finetuned"

# --- environment sanity for Windows/CPU ---
# bitsandbytes isn't used on CPU; avoid 8-bit paths
os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")

has_cuda = torch.cuda.is_available()
dtype = torch.float16 if has_cuda else torch.float32   # FP32 on CPU

print("🔹 Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, trust_remote_code=True)

print(f"🔹 Loading base model on {'CUDA' if has_cuda else 'CPU'} (no auto-offload)...")
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=dtype,
    low_cpu_mem_usage=False,      # avoid meta/offload shenanigans
    trust_remote_code=True,       # StableLM is fine with this
)

if has_cuda:
    base.to("cuda")

print("🔹 Loading LoRA adapter...")
model = PeftModel.from_pretrained(base, LORA_PATH, is_trainable=False)

print("🔹 Merging LoRA into base (this can take a bit)...")
# safe_merge=True validates shapes; set False if you’re sure
model = model.merge_and_unload()

print("🔹 Saving merged model...")
os.makedirs(OUT_PATH, exist_ok=True)
tok.save_pretrained(OUT_PATH)
model.save_pretrained(OUT_PATH)
print(f"✅ Done! Final model saved at: {OUT_PATH}")