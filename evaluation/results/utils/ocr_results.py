import argparse
import json
from pathlib import Path


LABEL_TO_ID = {"A": 0, "B": 1, "C": 2, "D": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple logprob accuracy: pick highest logprob and compare with ground truth."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL file")
    parser.add_argument(
        "--show-mismatches",
        type=int,
        default=10,
        help="How many mismatches to print",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    cer_total = 0
    mismatches = []

    with args.input.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue


            item = json.loads(line)

            cer = item.get("cer")
            cer_total += cer



    print("==== Simple Logprob Accuracy ====")
    print(f"Input: {args.input}")
    print(f"Cer: {cer_total}")


if __name__ == "__main__":
    main()
