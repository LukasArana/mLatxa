"""
Animal classification with a Vision-Language Model using vLLM.
"""

import argparse
import json
import re
from pathlib import Path

from PIL import Image
from tqdm import tqdm
from vllm import LLM, SamplingParams

PROMPT_EN = """What animal is shown in this image?

Respond using only the common English name of the animal.
Do not include explanations, punctuation or extra words.
"""

PROMPT_EU = """Zein animalia agertzen da irudi honetan?
Erantzun bakarrik animalia horren izen arrunta erabiliz."""


def get_arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--dataset_path", required=True, help="Path to the load animal dataset")
    p.add_argument("--labels_path", required=True, help="Path to the labels file for the animal dataset")
    p.add_argument("--language", choices=["en", "eu"], default="en", help="Language to prompt the model in and to evaluate against (English or Basque)")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_sample_dataset", type=int, default=None)
    p.add_argument("--output_file", default="animal_results.jsonl")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--min_p", type=float, default=0.0)
    p.add_argument("--presence_penalty", type=float, default=2.0, help="Presence penalty to encourage diversity in answers")
    p.add_argument("--repetition_penalty", type=float, default=2.0, help="Repetition penalty to discourage repetition")
    p.add_argument("--max_new_tokens", type=int, default=20)
    return p.parse_args()


def load_names(file_path):
    """
    Parses animals_name.txt with lines like:
        antelope\tantilope
        badger\tazkonar;azkon
        bat\tsaguzar;gauenara

    Returns a dict:
        {
          "antelope": {"en": ["antelope"], "eu": ["antilope"]},
          "badger":   {"en": ["badger"],   "eu": ["azkonar", "azkon"]},
          ...
        }
    """
    mapping = {}
    names_file = Path(file_path)
    if not names_file.exists():
        return mapping

    for line in names_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) < 2:
            continue
        en_name = parts[0].strip().lower()
        eu_variants = [v.strip().lower() for v in parts[1].split(";") if v.strip()]
        mapping[en_name] = {
            "en": [en_name],
            "eu": eu_variants if eu_variants else [en_name],
        }
    if mapping:
        print(f"Loaded {len(mapping)} names from {file_path}")
    else:
        print(f"No names loaded from {file_path}")
    print(mapping)
    return mapping


def load_dataset(root):
    samples = []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    not_include = ["pelecaniformes", "possum", "hornbill"]

    for img_path in root.rglob("*"):
        if img_path.suffix.lower() not in exts:
            continue
        if img_path.parent.name.lower() in not_include:
            continue
        samples.append(
            {
                "image_path": img_path,
                "label": img_path.parent.name.lower(),
            }
        )

    print(f"Loaded {len(samples)} images.")
    return samples


def prepare(batch, prompt):
    msgs = []
    for ex in batch:
        ex["image_path"]
        img = Image.open(ex["image_path"]).convert("RGB")
        msgs.append([{"role": "user", "content": [
            {"type": "image_pil", "image_pil": img},
            {"type": "text", "text": prompt}
        ]}])
    return msgs


def norm(x):
    x = x.lower().strip()
    x = re.sub(r"<think>.*?</think>", "", x, flags=re.S)
    x = re.sub(r"[^a-z -]", "", x)   # strip disallowed chars first
    x = re.sub(r"-{2,}", "-", x)     # collapse multiple hyphens
    x = x.strip(" -")                # remove leading/trailing hyphen/space
    return x.strip()


def get_accepted_answers(label, names_map, language):
    """
    Returns the normalized list of accepted answers for a given English
    folder-name label, in the requested language. Falls back to the raw
    label itself if it's not present in names.txt.
    """
    entry = names_map.get(label)
    if entry is None:
        return [norm(label)]
    return [norm(v) for v in entry[language]]
 
 
def normalize_basque(word):
    word = word.lower().strip()

    # Plural definite
    if word.endswith("ak"):
        word = word[:-2]

    # Singular definite after consonant: hamsterra -> hamster
    elif word.endswith("rra"):
        word = word[:-3] + "r"

    # Singular definite after other consonants:
    # txakurra -> txakur
    elif word.endswith("ra"):
        word = word[:-2]

    # Singular definite after vowel:
    # otsoa -> otso
    # katua -> katu
    elif word.endswith("a"):
        word = word[:-1]

    return word

def is_match(pred, accepted):
    pred_tokens = [normalize_basque(tok) for tok in pred.split()]

    for acc in accepted:
        acc = normalize_basque(acc)

        if acc in pred_tokens:
            return True

    return False

def main():
    args = get_arguments()
    dsroot = Path(args.dataset_path)

    dataset = load_dataset(dsroot)
    names_map = load_names(args.labels_path)

    if args.num_sample_dataset:
        dataset = dataset[:args.num_sample_dataset]

    prompt = PROMPT_EN if args.language == "en" else PROMPT_EU

    llm = LLM(model=args.model_path,
              tensor_parallel_size=args.tensor_parallel_size,
              mm_encoder_tp_mode="data",
              trust_remote_code=True)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
    )

    out = Path(args.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    correct = 0

    with open(out, "w") as f:
        for i in tqdm(range(0, len(dataset), args.batch_size)):
            batch = dataset[i:i + args.batch_size]
            outputs = llm.chat(
                messages=prepare(batch, prompt),
                sampling_params=sampling_params,
                chat_template_kwargs={"enable_thinking": False}
            )
            for ex, o in zip(batch, outputs):
                pred = norm(o.outputs[0].text)
                accepted = get_accepted_answers(ex["label"], names_map, args.language)
                ok = is_match(pred, accepted)
                correct += ok
                row = {
                    "image": str(ex["image_path"]),
                    "label": ex["label"],
                    "accepted_answers": accepted,
                    "prediction": pred,
                    "correct": ok,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        accuracy = correct/len(dataset) if len(dataset) > 0 else 0.0
        summary = {
            "summary": True,
            "accuracy": accuracy,
            "correct": correct,
            "total": len(dataset),
        }
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")

            
    print(f"Accuracy: {correct/len(dataset):.4f}")
    print(f"Saved to {out}")


if __name__ == "__main__":            
    main()