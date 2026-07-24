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

    total = 0
    valid = 0
    correct = 0
    mismatches = []

    with args.input.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            total += 1
            item = json.loads(line)

            gt_label = item.get("ground_truth")
            lp = item.get("option_log_probs")

            pred_label = max(lp, key=lambda k: float(lp[k]))

            valid += 1

            gt_id = LABEL_TO_ID[gt_label]
            pred_id = LABEL_TO_ID[pred_label]

            if pred_id == gt_id:
                correct += 1
            elif len(mismatches) < args.show_mismatches:
                mismatches.append(
                    {
                        "idx": item.get("idx", line_no),
                        "ground_truth": gt_label,
                        "ground_truth_id": gt_id,
                        "predicted": pred_label,
                        "predicted_id": pred_id,
                    }
                )

    accuracy = (100.0 * correct / valid) if valid else 0.0

    print("==== Simple Logprob Accuracy ====")
    print(f"Input: {args.input}")
    print(f"Rows: total={total}, valid={valid}")
    print(f"Mapping: A=0, B=1, C=2, D=3")
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{valid})")

    if mismatches:
        print("\nSample mismatches:")
        for m in mismatches:
            print(
                f"- idx={m['idx']} gt={m['ground_truth']}({m['ground_truth_id']}) "
                f"pred={m['predicted']}({m['predicted_id']})"
            )


if __name__ == "__main__":
    main()
