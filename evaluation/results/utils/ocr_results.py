import argparse
import json
from pathlib import Path


LABEL_TO_ID = {"A": 0, "B": 1, "C": 2, "D": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mean character error rate (CER) of an OCR results JSONL."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL file")
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    cer_total = 0.0
    n_scored = 0
    n_missing = 0

    with args.input.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue


            item = json.loads(line)

            cer = item.get("cer")
            if cer is None:  # generation/scoring failed for this row
                n_missing += 1
                continue
            cer_total += cer
            n_scored += 1



    print("==== OCR CER ====")
    print(f"Input: {args.input}")
    print(f"Scored rows: {n_scored} (missing CER: {n_missing})")
    print(f"Mean CER: {cer_total / n_scored:.4f}" if n_scored else "Mean CER: n/a")


if __name__ == "__main__":
    main()
