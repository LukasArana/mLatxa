"""
Translate animal names from English to Basque using a language model with vLLM.
"""

import argparse
import json
import re
from pathlib import Path

from tqdm import tqdm
from vllm import LLM, SamplingParams


PROMPT = """Translate the following animal into Basque.

Animal: {animal}

Respond using only the common Basque name of the animal.
Do not include explanations, punctuation or extra words.
"""


def get_arguments():
    p = argparse.ArgumentParser()

    p.add_argument("--model_path", required=True)
    p.add_argument("--labels_path", required=True, help="Path to animals_name.txt")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_sample_dataset", type=int, default=None)
    p.add_argument("--output_file", default="animal_translation_results.jsonl")

    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--min_p", type=float, default=0.0)

    p.add_argument("--presence_penalty", type=float, default=2.0)
    p.add_argument("--repetition_penalty", type=float, default=2.0)

    p.add_argument("--max_new_tokens", type=int, default=20)

    return p.parse_args()


def load_names(file_path):
    """
    animals_name.txt format:

    antelope    antilopea
    badger      azkonarra;azkona
    bat         saguzarra;gauenara
    """

    mapping = {}

    names_file = Path(file_path)

    if not names_file.exists():
        raise FileNotFoundError(file_path)

    for line in names_file.read_text(encoding="utf-8").splitlines():

        line = line.strip()

        if not line:
            continue

        parts = line.split(maxsplit=1)

        if len(parts) != 2:
            continue

        en = parts[0].strip().lower()

        eu = [
            x.strip().lower()
            for x in parts[1].split(";")
            if x.strip()
        ]

        mapping[en] = {
            "en": [en],
            "eu": eu,
        }

    print(f"Loaded {len(mapping)} animal names.")

    return mapping


def build_dataset(names_map):
    dataset = []

    for animal in sorted(names_map.keys()):
        dataset.append(
            {
                "label": animal,
            }
        )

    print(f"Created dataset with {len(dataset)} animals.")

    return dataset


def prepare(batch):
    messages = []

    for ex in batch:
        messages.append(
            [
                {
                    "role": "user",
                    "content": PROMPT.format(
                        animal=ex["label"]
                    ),
                }
            ]
        )

    print(messages)
    return messages


def norm(text):
    text = text.lower().strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"[^a-zA-ZÀ-ÿ -]", "", text)
    text = re.sub(r"-{2,}", "-", text)
    text = text.strip(" -")
    return text.lower()


def normalize_basque(word):
    word = word.lower().strip()

    if word.endswith("ak"):
        word = word[:-2]

    elif word.endswith("rra"):
        word = word[:-3] + "r"

    elif word.endswith("ra"):
        word = word[:-2]

    elif word.endswith("a"):
        word = word[:-1]

    return word


def is_match(pred, accepted):
    pred_tokens = [
        normalize_basque(tok)
        for tok in pred.split()
    ]

    for acc in accepted:

        acc = normalize_basque(acc)

        if acc in pred_tokens:
            return True

    return False


def main():

    args = get_arguments()

    names_map = load_names(args.labels_path)

    dataset = build_dataset(names_map)

    if args.num_sample_dataset:
        dataset = dataset[: args.num_sample_dataset]

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        max_tokens=args.max_new_tokens,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
    )

    out = Path(args.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)

    correct = 0

    with open(out, "w", encoding="utf-8") as f:

        for i in tqdm(range(0, len(dataset), args.batch_size)):

            batch = dataset[i : i + args.batch_size]

            outputs = llm.chat(
                messages=prepare(batch),
                sampling_params=sampling_params,
                chat_template_kwargs={
                    "enable_thinking": False
                },
            )

            for ex, out_obj in zip(batch, outputs):

                pred = norm(out_obj.outputs[0].text)

                accepted = names_map[ex["label"]]["eu"]

                ok = is_match(pred, accepted)

                correct += ok

                row = {
                    "animal": ex["label"],
                    "accepted_answers": accepted,
                    "prediction": pred,
                    "correct": ok,
                }

                f.write(
                    json.dumps(row, ensure_ascii=False) + "\n"
                )

        accuracy = correct / len(dataset)

        summary = {
            "summary": True,
            "accuracy": accuracy,
            "correct": correct,
            "total": len(dataset),
        }

        f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()