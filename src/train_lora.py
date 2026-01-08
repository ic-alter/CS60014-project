# src/train_lora.py
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = ROOT / "src" / "base_model"
LORA_OUT = ROOT / "src" / "lora_model"


def format_example(ex):
    return {
        "text": (
            "你是一个精通文言文的翻译助手。\n"
            "请将下面的文言文翻译成现代汉语。\n\n"
            f"文言文：{ex['input']}\n\n"
            f"现代汉语：{ex['output']}"
        )
    }


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        quantization_config=bnb,
        trust_remote_code=True,
    )

    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_cfg)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    ds = load_dataset(
        "json",
        data_files={
            "train": str(ROOT / "data" / "train.jsonl"),
        "validation": str(ROOT / "data" / "valid.jsonl"),
        },
    )

    ds = ds.map(format_example)

    def tokenize(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )
        out["labels"] = out["input_ids"].copy()
        return out

    ds = ds.map(tokenize, batched=True, remove_columns=ds["train"].column_names)

    args = TrainingArguments(
        output_dir=LORA_OUT,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=50,
        save_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
    )

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("Training interrupted, saving LoRA...")

    trainer.save_model(LORA_OUT)
    tokenizer.save_pretrained(LORA_OUT)

    print(f"[OK] LoRA saved to {LORA_OUT}")


if __name__ == "__main__":
    main()
