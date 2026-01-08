# src/download_base.py
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
ROOT = Path(__file__).resolve().parents[1]
SAVE_DIR = ROOT / "src" / "base_model"


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    tokenizer.save_pretrained(SAVE_DIR)
    model.save_pretrained(SAVE_DIR)

    print(f"[OK] Base model saved to {SAVE_DIR}")


if __name__ == "__main__":
    main()
