# src/prepare_data.py
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
C2MCHN = ROOT / "data" / "SCUT-C2MChn"
OUT = ROOT / "data"


def convert(split, limit=None):
    base = C2MCHN / split
    cch_path = base / f"{split}.cch"
    mch_path = base / f"{split}.mch"

    out_path = OUT / f"{split}.jsonl"

    count = 0
    with open(cch_path, encoding="utf-8") as fc, \
         open(mch_path, encoding="utf-8") as fm, \
         open(out_path, "w", encoding="utf-8") as fo:

        for cch, mch in zip(fc, fm):
            cch, mch = cch.strip(), mch.strip()
            if not cch or not mch:
                continue

            item = {
                "instruction": "将下面文言文翻译成现代汉语",
                "input": cch,
                "output": mch
            }
            fo.write(json.dumps(item, ensure_ascii=False) + "\n")

            count += 1
            if limit and count >= limit:
                break

    print(f"[OK] {split}: {count} samples -> {out_path}")


if __name__ == "__main__":
    convert("train", limit=10000)   # 可调小
    convert("valid", limit=1000)
    convert("test", limit=500)
