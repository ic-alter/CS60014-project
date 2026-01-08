# src/translate_compare.py
import json
import random
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "src" / "base_model"
LORA_DIR = ROOT / "src" / "lora_model"
TEST_CCH = ROOT / "data" / "SCUT-C2MChn" / "test" / "test.cch"
TEST_MCH = ROOT / "data" / "SCUT-C2MChn" / "test" / "test.mch"
OUT = ROOT / "results" / "compare_samples.jsonl"


def build_prompt(cch):
    return (
        "你是一个精通文言文的翻译助手。\n"
        "请将下面的文言文翻译成现代汉语。\n\n"
        f"文言文：{cch}\n\n现代汉语："
    )


def load_model(path, lora=False):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        path,
        device_map="auto",
        quantization_config=bnb,
        trust_remote_code=True,
    )
    if lora:
        model = PeftModel.from_pretrained(model, LORA_DIR)
    model.eval()
    return model


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_DIR, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = load_model(BASE_DIR)
    lora_model = load_model(BASE_DIR, lora=True)

    cch_lines = [x.strip() for x in open(TEST_CCH, encoding="utf-8") if x.strip()]
    mch_lines = [x.strip() for x in open(TEST_MCH, encoding="utf-8") if x.strip()]

    idxs = random.sample(range(len(cch_lines)), 20)

    OUT.parent.mkdir(exist_ok=True)

    with open(OUT, "w", encoding="utf-8") as f:
        for i in idxs:
            prompt = build_prompt(cch_lines[i])
            inputs = tokenizer(prompt, return_tensors="pt").to(base_model.device)

            with torch.no_grad():
                base_out = base_model.generate(**inputs, max_new_tokens=128)
                lora_out = lora_model.generate(**inputs, max_new_tokens=128)

            base_txt = tokenizer.decode(base_out[0], skip_special_tokens=True).split("现代汉语：")[-1].strip()
            lora_txt = tokenizer.decode(lora_out[0], skip_special_tokens=True).split("现代汉语：")[-1].strip()

            f.write(json.dumps({
                "cch": cch_lines[i],
                "reference": mch_lines[i],
                "base_pred": base_txt,
                "lora_pred": lora_txt
            }, ensure_ascii=False) + "\n")

    print(f"[OK] Comparison saved to {OUT}")


if __name__ == "__main__":
    main()
